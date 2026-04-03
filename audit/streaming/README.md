# Streaming Audit

## How It Works

Streaming flows through a 4-layer pipeline:

```
Python vmlx-engine (SSE over HTTP, ~80+ tok/s)
  -> VMLXService.streamWithTools() (SSE line parsing, batching, tag wrapping)
    -> ChatEngine.wrapStreamWithLogging() (Task.detached producer, token counting)
      -> StreamingDeltaProcessor (adaptive flush, <think> tag parsing, UI sync throttle)
        -> ChatTurn (appendContent / appendThinking)
          -> NSTableView (row reconfigure, scroll anchor)
```

### Layer 1: Python Engine SSE Output

- Engine emits `data: {json}\n\n` lines at rate controlled by `--stream-interval` (default 3 = every 3rd token)
- Each line is an OpenAI-format chunk: `choices[0].delta.content`, `delta.reasoning_content`, `delta.tool_calls`
- `stream_options.include_usage: true` causes usage stats in every chunk
- Keep-alive comments (`: keep-alive`) sent during long pauses

### Layer 2: VMLXService SSE Consumer

**File:** `Services/Inference/VMLXService.swift` (lines 105-283)

**Functions:**
- `streamWithTools()` — main entry, builds HTTP POST, opens `URLSession.shared.bytes(for:)`, iterates `bytes.lines`
- `buildRequestBody()` — serializes messages, strips `<think>` blocks from history, encodes tools
- `resolveModel()` — maps requested model string to (name, path)
- `ensureEngineRunning()` — checks gateway, launches engine if needed

**Content Batching** (lines 162-232):
- Accumulates content deltas into `contentBatch` and `reasoningBatch` strings
- Yields at 30ms intervals max (~33 yields/sec) or on `finish_reason`
- Reasoning content wrapped in `<think>...</think>` before yielding
- `CFAbsoluteTimeGetCurrent()` used for timing (no Date() syscall overhead)

**Tool Call Accumulation** (lines 195-247):
- `AccumulatedToolCall` struct: `id`, `name`, `arguments` (string concatenation)
- Tool call deltas indexed by SSE `index` field
- On `finish_reason == "tool_calls"`: emits `StreamingToolHint.encode()` sentinels, throws `ServiceToolInvocation` for first tool only

**Cancellation** (lines 279-281):
- `continuation.onTermination` cancels the inner `streamTask`
- Closing URLSession bytes stream causes Python engine to detect disconnect and abort

### Layer 3: ChatEngine Stream Wrapper

**File:** `Services/Chat/ChatEngine.swift` (lines 151-256)

**Functions:**
- `wrapStreamWithLogging()` — wraps inner stream in `Task.detached(priority: .userInitiated)` to avoid actor isolation deadlock
- Uses `producerTask` reference + `continuation.onTermination` for cancellation

**Token counting:**
- Counts output tokens heuristically: `max(1, delta.count / 4)` per delta
- Skips `StreamingToolHint` sentinels
- Logs via `InsightsService.logInference()` on completion (chat UI only)

### Layer 4: StreamingDeltaProcessor

**File:** `Utils/StreamingDeltaProcessor.swift` (lines 1-279)

**Functions:**
- `receiveDelta()` — buffers delta, checks flush conditions
- `flush()` — parses `<think>` tags, routes to `appendContent()` or `appendThinking()`
- `finalize()` — drains all buffers, syncs to UI
- `parseAndRoute()` — state machine for `<think>`/`</think>` tag parsing with partial-prefix lookahead
- `syncIfNeeded()` — throttles UI updates: 100ms (short), 150ms (medium), 200-250ms (long)
- `recomputeFlushTuning()` — adjusts buffer size/interval based on total output length

**Adaptive tuning:**
- 50ms flush / 256 char buffer initially
- Scales to 150ms / 1024 chars for outputs >20k chars
- `deltasSinceLastCheck` counter: only calls `Date()` every 4th delta
- `longestFlushMs > 50` triggers 1.5x interval increase
- Fallback 100ms timer for push-based consumers

**Tag parsing:**
- Partial prefix arrays: `["<think", "<thin", "<thi", "<th", "<t", "<"]` and close equivalents
- `pendingTagBuffer` holds partial tags across chunk boundaries
- Case-insensitive matching

### Layer 5: UI Rendering

**Files:**
- `Views/Chat/MessageThreadView.swift` — `NSTableView` wrapper
- `Views/Chat/MessageTableRepresentable.swift` — coordinator, height invalidation, auto-scroll
- `Views/Chat/NativeThinkingView.swift` — collapsible thinking box (pure AppKit)
- `Views/Chat/ScrollAnchorManager.swift` — `isPinnedToBottom` tracking

**Scroll behavior:**
- Auto-scroll when pinned to bottom
- User scroll-up disables auto-scroll
- Re-enables on: scroll back to bottom, or new message sent
- Height invalidation on content changes triggers re-layout

---

## What Needs Checking

### Critical

| # | Issue | File | Line |
|---|-------|------|------|
| S1 | **UTF-8 splitting in batching** — `contentBatch += content` concatenates strings, but `bytes.lines` delivers complete lines. However, if a JSON string contains multi-byte UTF-8, does JSONSerialization handle it? Need to verify no character splitting at SSE → JSON → String boundaries. | VMLXService.swift | 179 |
| S2 | **Reasoning content to external API clients** — VMLXService wraps `reasoning_content` in `<think>` tags before yielding. External HTTP clients via HTTPHandler receive these `<think>` tags in `delta.content` instead of `delta.reasoning_content`. This is a **known bug** — external clients get polluted content. | VMLXService.swift | 216-222 |
| S3 | **`<think>` tag split across 30ms batches** — If reasoning ends and content begins within one 30ms window, the `</think>` close tag is yielded followed by content in the same batch. StreamingDeltaProcessor handles this, but verify the ordering is always correct. | VMLXService.swift | 224-230 |
| S4 | **Stream cancellation → Python engine** — When user presses stop, `streamTask.cancel()` closes the URLSession. Does the Python engine actually detect this and stop generation? Or does it continue generating tokens into the void? | VMLXService.swift | 279-281 |
| S5 | **OOM mid-stream** — If Python engine dies mid-generation, `bytes.lines` will throw. Is the error caught? Is `generationDidFinish()` called? Is partial response preserved? Error path at line 266 calls both, but verify. | VMLXService.swift | 266-274 |

### Performance

| # | Issue | File | Line |
|---|-------|------|------|
| S6 | **30ms batch interval vs stream-interval** — If engine already batches at `--stream-interval 3`, Swift batching on top means double-buffering. At 80 tok/s with interval=3, engine sends ~27 events/sec. 30ms batch catches ~1 event per flush. Is this redundant? | VMLXService.swift | 167 |
| S7 | **syncIfNeeded cadence** — At 80+ tok/s, with 30ms flush and 100-250ms sync, the UI updates 4-10 times/sec. Is this smooth enough? The NSTableView `reconfigureRows` is O(visible rows). | StreamingDeltaProcessor.swift | 205-221 |
| S8 | **`recomputeFlushTuning()` called on every large-buffer flush** — Minor overhead, but called in hot path. Could cache the computation. | StreamingDeltaProcessor.swift | 223-240 |
| S9 | **Date() syscall in flush()** — `flush()` creates two `Date()` objects (lines 125, 133). Called ~33 times/sec. Could use `CFAbsoluteTimeGetCurrent()` like VMLXService does. | StreamingDeltaProcessor.swift | 125, 133 |

### Edge Cases

| # | Issue | File | Line |
|---|-------|------|------|
| S10 | **Empty content deltas** — Engine can send chunks with only `usage` and no `content`. These trigger `shouldFlush` on finish but yield nothing. Verify no empty strings are yielded to StreamingDeltaProcessor. | VMLXService.swift | 214 |
| S11 | **Very long tool call arguments** — If tool call JSON is >64KB, `AccumulatedToolCall.arguments` grows via string concatenation. No memory limit. Could this cause issues with very large function arguments? | VMLXService.swift | 198 |
| S12 | **Multiple tool calls in one response** — Only first tool is dispatched via `ServiceToolInvocation` (line 241). Remaining tools are emitted as `StreamingToolHint` sentinels but never executed. **Known limitation.** | VMLXService.swift | 236-247 |
| S13 | **`hasEmittedThinkOpen` state on error** — If error occurs while thinking is open, line 267 emits `</think>`. But if the error is `ServiceToolInvocation`, this runs after the `return` at line 246. Verify tool calls during thinking don't leave unclosed tags. | VMLXService.swift | 234-247 |
| S14 | **Keep-alive during long thinking** — Engine sends `: keep-alive` comments during long pauses. `VMLXSSEParser.parse()` returns `nil` for these. But does URLSession's `bytes.lines` timeout before keep-alive arrives? Default URLSession timeout is 60s. | VMLXSSEParser.swift | 60 |
| S18 | **No inactivity timeout** — `URLSession.shared.bytes(for:)` uses default 300s timeout. If engine hangs (alive but silent), stream blocks 5 minutes. Need inactivity watchdog (30-60s). Reference: `RemoteProviderService` has `StreamInactivityHandler`. | VMLXService.swift | 144 |
| S19 | **SSE stream ends without [DONE]** — If connection drops before `[DONE]`, `for try await` loop exits normally. No error thrown, response may be incomplete. Need truncation indicator. | VMLXService.swift | 173 |

### Stats Flow

| # | Issue | File | Line |
|---|-------|------|------|
| S15 | **TTFT includes engine launch time** — `generationDidStartAsync()` is called at line 154, inside the stream task. But `ensureEngineRunning()` (line 124) may have already spent 0-120s launching. TTFT measures from stream open to first token, which is correct for warm engines but misleading for cold starts. | VMLXService.swift | 124, 154 |
| S16 | **TPS calculation accuracy** — `InferenceProgressManager.updateStats()` computes TPS from `completion` token count growth. But token counts come from the engine's running total, not per-chunk deltas. If usage arrives irregularly, TPS could be spiky. | InferenceProgressManager.swift | 111-128 |
| S17 | **Stats not reset on error** — `generationDidFinishAsync()` is called on error path (line 272), which sets `isGenerating = false`. But final stats from the partial generation remain visible. Is this desired? | VMLXService.swift | 272 |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| SSE parsing | Working | All chunk types: content, reasoning, tool calls, usage, finish_reason |
| Content batching | Working | 30ms intervals, reduces MainActor pressure |
| Reasoning wrapping | Working | `<think>` tags injected, StreamingDeltaProcessor parses them |
| Tool call accumulation | Working | Two-chunk protocol handled via AccumulatedToolCall |
| Stream cancellation | Working | onTermination → task cancel → URLSession close |
| Inference stats | Working | TPS, TTFT, cache hits displayed in FloatingInputCard |
| Adaptive flush tuning | Working | Scales with output length |
| External API reasoning_content | **Bug** | Leaked as `<think>` tags in content instead of separate field |
| Engine error chunks | **Bug** | Parser silently ignores error SSE chunks (no `choices` key) |
| Parallel tool calls | **Limitation** | Only first tool dispatched |

### Engine Compatibility Items (from cross-reference with Python source)

| # | Issue | File | Line |
|---|-------|------|------|
| S20 | **Engine error SSE chunks silently swallowed** — Engine emits `data: {"error":{"message":"...","type":"server_error"}}` on stream failure (`server.py:5009-5024`). VMLXSSEParser looks for `choices` array, finds none, returns `nil`. Error is lost. Stream appears to end normally. | VMLXSSEParser.swift | 75-77 |
| S21 | **Engine partial usage on error also lost** — Engine sends usage chunk after error chunk (`server.py:5025-5039`) with partial token counts. Also ignored since no `choices`. | VMLXSSEParser.swift | 75-77 |
| S22 | **No request timeout sent to engine** — Swift doesn't send `timeout` in request body. Engine falls back to `_default_timeout` (CLI flag or None). Combined with S18 (no Swift inactivity watchdog), a hung engine blocks indefinitely. | VMLXService.swift | 131 |
