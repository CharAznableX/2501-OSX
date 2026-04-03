# Error Handling Audit

## How It Works

Errors propagate through 4 layers: Python engine → VMLXService → ChatEngine → UI/API.

### Error Types

**File:** `Services/Inference/VMLXProcessManager.swift` (lines 461-482)

```swift
enum VMLXError: Error, LocalizedError {
    case engineStartTimeout        // 120s health check timeout
    case engineCrashed(model:stderr:) // Process died during startup
    case portAllocationFailed      // socket bind failed
    case engineNotRunning(model:)  // HTTP error from engine
    case noModelLoaded             // No installed models found
}
```

### Error Sources and Propagation

| Error Source | Error Type | Where Caught | User Sees |
|--------------|-----------|--------------|-----------|
| Engine won't start (120s timeout) | `VMLXError.engineStartTimeout` | VMLXService.ensureEngineRunning → ChatEngine → ChatSession.send() | Error in chat? |
| Engine crashes during startup | `VMLXError.engineCrashed(model:stderr:)` | Same path | Last stderr line |
| Engine returns non-200 HTTP | `VMLXError.engineNotRunning` | VMLXService.streamWithTools() line 148 | HTTP status code |
| Engine dies mid-stream | URLSession bytes throws | VMLXService catch block line 266 | Generic error |
| Port allocation fails | `VMLXError.portAllocationFailed` | launchEngine() | System error |
| No models installed | `VMLXError.noModelLoaded` | resolveModel() | "No model loaded" |
| Tool call interrupt | `ServiceToolInvocation` | ChatEngine / HTTPHandler | Tool call response |

### ChatEngine Error Path

**File:** `Services/Chat/ChatEngine.swift` (lines 199-214)

In `wrapStreamWithLogging()`:
- `ServiceToolInvocation`: re-thrown, counts as `finishReason: .toolCalls`
- `CancellationError`: normal cancel, `continuation.finish()` (no error)
- Other errors: `finishReason: .error`, logged via `InsightsService`, `continuation.finish(throwing: error)`

### VMLXService Error Path

**File:** `Services/Inference/VMLXService.swift` (lines 266-274)

On stream error:
1. Close unclosed `<think>` tag if needed (line 267-268)
2. Reset idle timer (line 271)
3. Call `InferenceProgressManager.shared.generationDidFinishAsync()` (line 272)
4. `continuation.finish(throwing: error)` (line 273)

### HTTPHandler Error Path

External API clients receive errors as:
- Non-streaming: JSON `{"error": {"message": "...", "type": "...", "code": ...}}`
- Streaming: SSE error event, then connection close

### Process Monitor Error Path

**File:** `Services/Inference/VMLXProcessManager.swift` (lines 332-382)

`startMonitor()` detects unexpected process exit:
1. Log last stderr line
2. Unregister from gateway
3. Clean up process/timers
4. If non-zero exit: auto-restart with exponential backoff (2s, 4s, 6s), max 3 retries
5. After max retries: log "giving up (OOM or bad model?)"

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| E1 | **User-facing error messages** — When engine crashes or times out, what does the user actually see in the chat UI? Is it a clear error message or a generic spinner? Trace the error from VMLXService → ChatEngine → ChatSession → ChatTurn → UI. |
| E2 | **Partial response on crash** — If engine dies mid-generation (OOM), is the partial response preserved in the ChatTurn? Or is it lost? The error path at VMLXService line 266-274 doesn't explicitly save partial content. |
| E3 | **generationDidFinish on all paths** — Is `InferenceProgressManager.generationDidFinishAsync()` called on EVERY error path? If not, `isGenerating` stays true forever, stats chip shows stale data. |
| E4 | **Error differentiation** — Can the user tell the difference between: model too large (OOM), bad model files (corrupt), network error (engine HTTP), timeout (slow load)? All currently show as generic errors. |

### Engine Errors

| # | Issue | Notes |
|---|-------|-------|
| E5 | **OOM during generation** — Engine gets killed by macOS memory pressure. No warning before launch. Process monitor detects exit, logs stderr, but user's in-flight message gets an error. |
| E6 | **Engine returns HTTP error** — VMLXService line 148: `throw VMLXError.engineNotRunning(model: "HTTP \(statusCode) from engine")`. Could be 400 (bad request), 500 (engine error), 503 (overloaded). All conflated into one error type. |
| E7 | **Health check false positive** — Engine responds 200 on `/health` but is actually in a bad state (e.g., model partially loaded). Subsequent requests fail. |
| E8 | **Restart loop detection** — After 3 failed restarts, the monitor gives up. But the user can trigger another launch by sending a message. `restartCounts` is only consulted by the monitor, not by `launchEngine()`. |

### API Errors

| # | Issue | Notes |
|---|-------|-------|
| E9 | **Consistent error format** — OpenAI, Ollama, Anthropic formats all have different error structures. Are errors formatted correctly for each dialect? |
| E10 | **Streaming error mid-response** — If error occurs after SSE headers are sent, can the writer properly close the stream? Or does the client hang? |
| E11 | **Request validation** — What happens with: missing model field, empty messages array, invalid temperature, malformed tools JSON? |

### Cleanup

| # | Issue | Notes |
|---|-------|-------|
| E12 | **Idle timer on error** — VMLXService line 271 calls `resetIdleTimer` even on error. This is correct (engine is still running). But what if the error is because the engine died? Timer targets a dead engine. |
| E13 | **Gateway stale entry** — If engine dies between requests and no monitor fires, gateway has stale port. Next request to that port fails. Need to clean up gateway on connection failure. |
| E14 | **Multiple concurrent errors** — If two requests hit a crashed engine simultaneously, both get errors. Both might try to restart the engine. The `launching` guard prevents duplicate launches but the first requester waits 120s for the second's launch. |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| VMLXError types | Working | 5 specific error types |
| Crash detection during startup | Working | Process liveness check in health loop |
| Crash restart with backoff | Working | 3 retries, exponential |
| Idle timer reset on error | Working | Correct behavior |
| Error logging | Working | InsightsService, os_log |
| User-facing error messages | **Needs verify** | May be generic |
| Partial response preservation | **Needs verify** | May lose content |
| Error differentiation | **Missing** | All errors look the same to user |
| OOM pre-launch warning | **Missing** | No memory budget check |
| Gateway stale entry cleanup | **Missing** | Dead port stays registered |
