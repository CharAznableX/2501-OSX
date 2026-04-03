# Streaming Rule Book

Rules governing the 4-layer streaming pipeline: Python engine SSE → VMLXService batching → StreamingDeltaProcessor → UI.

---

## Layer 1: SSE Protocol (VMLXSSEParser)

### RULE S-SSE-1: Line Format
Every SSE data line MUST match `data: {json}\n` or `data: [DONE]`. Lines starting with `:` are keep-alive comments and MUST return `nil`. Empty lines MUST return `nil`.

**Verified:** `VMLXSSEParser.swift:57-61` — trims whitespace, skips empty/comment lines.

### RULE S-SSE-2: JSON Parsing
JSON parsing MUST use `JSONSerialization` (not `JSONDecoder`) because the schema is dynamic (optional fields vary per chunk). Parsing failures MUST return `nil` (skip), never throw.

**Verified:** `VMLXSSEParser.swift:73-78` — `try?` JSONSerialization, returns `nil` on failure.

### RULE S-SSE-3: Field Extraction
Every chunk MUST extract these fields when present:
- `choices[0].delta.content` → `chunk.content`
- `choices[0].delta.reasoning_content` → `chunk.reasoningContent`
- `choices[0].delta.tool_calls` → `chunk.toolCalls` (array of incremental deltas)
- `choices[0].finish_reason` → `chunk.finishReason`
- `usage` (top-level) → `chunk.usage` (prompt_tokens, completion_tokens, cached_tokens, cache_detail)

**Verified:** `VMLXSSEParser.swift:82-127`.

### RULE S-SSE-4: Tool Call Deltas
Tool calls arrive incrementally: first chunk has `id` + `name` + partial `arguments`. Continuation chunks have only `index` + more `arguments`. Parser MUST handle both (using empty string defaults for missing id/name).

**Verified:** `VMLXSSEParser.swift:104-111`.

### RULE S-SSE-5: [DONE] Sentinel
`data: [DONE]` MUST be recognized as stream termination. Parser MUST return `VMLXSSEChunk(isDone: true)`.

**Verified:** `VMLXSSEParser.swift:68-69`.

### RULE S-SSE-6: UTF-8 Safety
`bytes.lines` delivers complete UTF-8 lines. `JSONSerialization` handles multi-byte UTF-8 in JSON strings. No character splitting can occur at the SSE→JSON→String boundary.

**Status:** SAFE — `URLSession.AsyncBytes.lines` guarantees complete line delivery.

---

## Engine Compatibility (Python ↔ Swift Contract)

### RULE S-ENGINE-1: ensure_ascii JSON Encoding
Python engine serializes SSE JSON with `ensure_ascii=True` (`server.py:4611`). All multi-byte UTF-8 (emoji, CJK) is escaped as `\uXXXX`. This means the JSON payload over SSE is pure ASCII — S1 (UTF-8 splitting) is mitigated engine-side. `JSONSerialization` on the Swift side correctly unescapes `\uXXXX` back to native Unicode.

### RULE S-ENGINE-2: exclude_none Serialization
Engine uses `exclude_none=True` in `model_dump()`. Fields like `reasoning_content`, `tool_calls` are omitted entirely when `None` rather than appearing as `null`. Swift parser MUST handle missing keys (optional fields) — which it does via `as?` casts.

**Verified:** `VMLXSSEParser.swift:93-111` — all field extractions use `as?` optional casts.

### RULE S-ENGINE-3: reasoning_content Computed Field
Engine stores reasoning internally as `delta.reasoning` but serializes it to JSON as `reasoning_content` via Pydantic `@computed_field` (`api/models.py:735-739`). Swift MUST read `delta["reasoning_content"]` (NOT `delta["reasoning"]`).

**Verified:** `VMLXSSEParser.swift:98` — reads `delta["reasoning_content"]`.

### RULE S-ENGINE-4: Keep-Alive at 15s Intervals
Engine emits `: keep-alive\n\n` (SSE comment) every 15 seconds during silence (`_SSE_KEEPALIVE_INTERVAL = 15.0`, `server.py:4621`). This prevents proxies and URLSession from timing out during long prefills. Swift parser correctly skips these (comment lines start with `:`).

**Verified:** `VMLXSSEParser.swift:60` — lines starting with `:` return `nil`.

### RULE S-ENGINE-5: Error Chunks (NEW BUG FOUND)
Engine emits error SSE chunks on stream failure (`server.py:5009-5024`):
```json
data: {"id":"...","object":"chat.completion.chunk","error":{"message":"...","type":"server_error","code":"internal_error"}}
```
**BUG:** Swift `VMLXSSEParser` does NOT handle these error chunks. It looks for `choices` array, finds none, returns `nil`. Engine-side stream errors are **silently swallowed**. The stream appears to end normally.

**Fix required:** Parser should check for `error` key in JSON and either:
- (a) Return a new `VMLXSSEChunk` field `.error: VMLXStreamError?`
- (b) Or at minimum log the error message

### RULE S-ENGINE-6: Engine-Side Tool Call Parsing
Engine has its own tool call marker detection using `_TOOL_CALL_MARKERS` (`server.py:4864-4882`). When markers are found, engine buffers content and later emits `tool_calls` in the delta. Tool call arguments arrive incrementally. The Swift side accumulates them via `AccumulatedToolCall`.

**Contract:** Engine emits `finish_reason: "tool_calls"` with the final tool call chunk. Swift relies on this to dispatch.

### RULE S-ENGINE-7: Usage in Every Chunk
When `stream_options.include_usage` is `true` (Swift always sends this, `VMLXService.swift:398`), the engine includes `usage` in **every** chunk — not just the final one (`server.py:4906`). This enables real-time TPS tracking.

**Verified:** Swift reads `usage` from every chunk at `VMLXService.swift:185-192`.

### RULE S-ENGINE-8: Engine-Side Reasoning Suppression
When `enable_thinking: false` is sent and the model is a thinking model, the engine suppresses reasoning chunks (`server.py:4908-4916`). Since Swift NEVER sends `enable_thinking` (`VMLXService.swift:391-394`), the engine always emits reasoning when the parser detects it. The Swift `showThinking` flag controls UI display only.

### RULE S-ENGINE-9: Engine Request Timeout
Engine has a per-request timeout via `request.timeout` field, falling back to `_default_timeout` (`server.py:4804`). Swift does NOT send a `timeout` field in the request body. The engine's default timeout applies. If the engine has no default timeout configured, generation runs indefinitely.

**Gap:** No Swift-side inactivity timeout AND no explicit engine timeout being set. Both sides rely on defaults.

---

## Layer 2: Content Batching (VMLXService)

### RULE S-BATCH-1: Yield Interval
Content deltas MUST be batched at minimum 30ms intervals (`minYieldIntervalSec = 0.03`). At 80+ tok/s this reduces async hops from ~80/sec to ~33/sec. Timer uses `CFAbsoluteTimeGetCurrent()` (no Date syscall).

**Verified:** `VMLXService.swift:167,210-211`.

### RULE S-BATCH-2: Finish Flush
When `finish_reason` is non-nil, all accumulated content MUST be flushed immediately regardless of the 30ms interval.

**Verified:** `VMLXService.swift:212` — `|| chunk.finishReason != nil`.

### RULE S-BATCH-3: Think Tag Wrapping Order
Reasoning content MUST be wrapped in `<think>...</think>` tags. The yield order MUST be:
1. If reasoning accumulated and no `<think>` emitted yet → yield `<think>`
2. Yield reasoning batch
3. If content accumulated and `<think>` was open → yield `</think>`
4. Yield content batch

This ensures thinking and content never interleave within a single yield.

**Verified:** `VMLXService.swift:214-232`.

### RULE S-BATCH-4: Think Tag Close on Stream End
On normal stream end, any unclosed `<think>` tag MUST be closed with `</think>`.

**Verified:** `VMLXService.swift:259-261`.

### RULE S-BATCH-5: Think Tag Close on Error
On error (including tool call termination via `ServiceToolInvocation`), any unclosed `<think>` tag MUST be closed.

**Verified:** `VMLXService.swift:267-268`. **ISSUE:** If `finish_reason == "tool_calls"`, the function returns at line 246 BEFORE the error handler at 267. Think tags opened during tool-call responses will NOT be closed. **(S13 — confirmed bug)**

### RULE S-BATCH-6: Show Thinking Gate
Reasoning content MUST only be accumulated when `showThinking == true`. When `showThinking == false`, `reasoning_content` is silently discarded (not generated — engine still generates it, but UI doesn't display it).

**Verified:** `VMLXService.swift:176` — `if ... showThinking { reasoningBatch += reasoning }`.

### RULE S-BATCH-7: Tool Call Accumulation
Tool call deltas MUST be accumulated into `AccumulatedToolCall` structs indexed by `delta.index`. Arguments are concatenated via `+=`.

**Verified:** `VMLXService.swift:195-207`.

### RULE S-BATCH-8: Tool Call Dispatch
On `finish_reason == "tool_calls"`, ALL accumulated tools are emitted as `StreamingToolHint` sentinels, but only the FIRST tool is dispatched as `ServiceToolInvocation`.

**Known limitation:** Parallel tool calls not supported (S12).

**Verified:** `VMLXService.swift:235-247`.

### RULE S-BATCH-9: Stats Updates
Every chunk with `usage` data MUST update `InferenceProgressManager.shared.updateStatsAsync()`. This is lightweight (no UI sync) and MUST NOT be throttled.

**Verified:** `VMLXService.swift:185-192`.

### RULE S-BATCH-10: Idle Timer Reset
Idle timer MUST be reset AFTER stream completion (not before), to prevent triggering sleep during active generation.

**Verified:** `VMLXService.swift:263` (normal) and `271` (error).

---

## Layer 3: Stream Cancellation

### RULE S-CANCEL-1: Consumer Cancellation Path
When consumer cancels the stream, `continuation.onTermination` MUST cancel `streamTask`, which closes URLSession, causing `bytes.lines` to throw `CancellationError`.

**Verified:** `VMLXService.swift:279-281`.

### RULE S-CANCEL-2: Engine Disconnect Detection
Python engine MUST detect client disconnect via `request.is_disconnected()` and call `engine.abort_request()` to stop token generation.

**Verified:** `server.py:4815-4818` — engine checks disconnect and aborts.

---

## Layer 4: StreamingDeltaProcessor

### RULE S-PROC-1: Think Tag State Machine
`parseAndRoute()` MUST track `isInsideThinking` state. Content before `<think>` goes to `appendContent()`. Content between `<think>` and `</think>` goes to `appendThinking()`. Tags are case-insensitive.

**Verified:** `StreamingDeltaProcessor.swift:248-278`.

### RULE S-PROC-2: Partial Tag Buffering
If a chunk ends with a partial tag prefix (e.g., `<thi`), the partial MUST be saved in `pendingTagBuffer` and prepended to the next chunk. Partial prefixes checked longest-first.

**Verified:** `StreamingDeltaProcessor.swift:245-246,255-258,268-271`.

### RULE S-PROC-3: Adaptive Flush Tuning
Flush interval and buffer size MUST scale with total output length:
- 0-2K chars: 50ms / 256 chars
- 2K-8K: 75ms / 512
- 8K-20K: 100ms / 768
- 20K+: 150ms / 1024

If longest flush exceeds 50ms, interval MUST increase by 1.5x (capped at 200ms).

**Verified:** `StreamingDeltaProcessor.swift:223-240`.

### RULE S-PROC-4: UI Sync Throttling
UI updates via `syncToTurn()` → `turn.notifyContentChanged()` MUST be throttled:
- 0-2K chars: 100ms sync interval
- 2K-5K: 150ms
- 5K-10K: 200ms
- 10K+: 250ms

First content MUST always sync immediately (`syncCount == 0`).

**Verified:** `StreamingDeltaProcessor.swift:205-221`.

### RULE S-PROC-5: Fallback Timer
A 100ms fallback timer MUST fire when deltas stop arriving but buffer is non-empty. This prevents content from getting stuck in the buffer when no more deltas trigger inline flushes.

**Verified:** `StreamingDeltaProcessor.swift:106-117`.

### RULE S-PROC-6: Finalize Drains All
`finalize()` MUST drain all remaining content from `deltaBuffer` and `pendingTagBuffer`. Routing follows current `isInsideThinking` state.

**Verified:** `StreamingDeltaProcessor.swift:138-153`.

---

## Layer 5: External API Forwarding

### RULE S-EXT-1: Reasoning Content Separation (BUG — S2)
External API clients MUST receive `reasoning_content` in a separate `delta.reasoning_content` field, NOT as `<think>` tags inside `delta.content`. Currently broken — VMLXService wraps reasoning in `<think>` tags and HTTPHandler forwards raw deltas to `writeContent()` without separating them.

**File:** `HTTPHandler.swift:2128-2138` — `for try await delta in stream` → `writeContent(delta, ...)`.

**Fix required:** Either:
- (a) VMLXService yields structured data (content vs reasoning) instead of tag-wrapped strings
- (b) HTTPHandler parses `<think>` tags out and puts them in a separate `reasoning_content` field
- (c) Add a separate external-facing stream path that uses `VMLXSSEParser` data directly

### RULE S-EXT-2: Stream End Marker
External SSE MUST terminate with `data: [DONE]\n\n`. Currently handled by `writeEnd()`.

**Verified:** `ResponseWriters.swift` — `writeEnd()` sends `[DONE]`.

---

## Known Issues

| ID | Rule Violated | Severity | Status |
|----|---------------|----------|--------|
| S2 | S-EXT-1 | **HIGH** | Bug — reasoning_content leaked as `<think>` tags to external clients |
| S5 | — | Medium | OOM mid-stream: error path verified (line 266-274), partial response preserved |
| S12 | S-BATCH-8 | Medium | Only first parallel tool call dispatched |
| S13 | S-BATCH-5 | Low | Think tags not closed on tool_calls finish_reason |
| S18 | — | **HIGH** | No inactivity timeout — engine hang blocks 300s (URLSession default) |
| S19 | — | Medium | Connection drop without [DONE] exits loop silently, no truncation indicator |
| **NEW** | S-ENGINE-5 | **HIGH** | Engine error SSE chunks silently swallowed — parser ignores chunks without `choices` |

---

## Invariants (must always hold)

1. **No async hop per token** — Batching must aggregate content; yielding per SSE line defeats the purpose.
2. **Think tags always balanced** — Every `<think>` MUST have a matching `</think>` before stream completion.
3. **Stats always finalized** — `generationDidFinishAsync()` MUST be called on ALL exit paths (normal, error, cancellation).
4. **Idle timer always reset** — `resetIdleTimer()` MUST be called on ALL exit paths except cancellation (engine might still be running).
5. **Engine always notified of cancellation** — Closing URLSession MUST propagate to engine disconnect detection.
