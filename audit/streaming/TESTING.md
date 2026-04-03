# Streaming Testing Suite

Test cases for every streaming audit item (S1-S19) and every rule in the rule book.

---

## Unit Tests: VMLXSSEParser

### T-SSE-1: Parse content delta
```
Input:  data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}
Expect: chunk.content == "hello", chunk.finishReason == nil, chunk.isDone == false
```

### T-SSE-2: Parse reasoning_content delta
```
Input:  data: {"choices":[{"delta":{"reasoning_content":"thinking...","content":null},"finish_reason":null}]}
Expect: chunk.reasoningContent == "thinking...", chunk.content == nil
```

### T-SSE-3: Parse [DONE] sentinel
```
Input:  data: [DONE]
Expect: chunk.isDone == true
```

### T-SSE-4: Parse keep-alive comment
```
Input:  : keep-alive
Expect: nil (skip)
```

### T-SSE-5: Parse empty line
```
Input:  ""
Expect: nil (skip)
```

### T-SSE-6: Parse finish_reason "stop"
```
Input:  data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
Expect: chunk.finishReason == "stop"
```

### T-SSE-7: Parse finish_reason "tool_calls"
```
Input:  data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
Expect: chunk.finishReason == "tool_calls"
```

### T-SSE-8: Parse tool call first chunk (id + name + partial args)
```
Input:  data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","function":{"name":"get_weather","arguments":"{\"loc"}}]}}]}
Expect: chunk.toolCalls[0].index == 0, .id == "call_123", .functionName == "get_weather", .arguments == "{\"loc"
```

### T-SSE-9: Parse tool call continuation chunk (args only)
```
Input:  data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"NYC\"}"}}]}}]}
Expect: chunk.toolCalls[0].index == 0, .id == "", .functionName == "", .arguments == "ation\":\"NYC\"}"
```

### T-SSE-10: Parse usage stats
```
Input:  data: {"choices":[{"delta":{},"finish_reason":null}],"usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60,"prompt_tokens_details":{"cached_tokens":30,"cache_detail":"prefix_cache"}}}
Expect: chunk.usage.promptTokens == 50, .completionTokens == 10, .cachedTokens == 30, .cacheDetail == "prefix_cache"
```

### T-SSE-11: Parse malformed JSON (graceful skip)
```
Input:  data: {invalid json
Expect: nil (no crash, no throw)
```

### T-SSE-12: Parse content with multi-byte UTF-8
```
Input:  data: {"choices":[{"delta":{"content":"こんにちは🌍"}}]}
Expect: chunk.content == "こんにちは🌍"
```

### T-SSE-13: Parse line without "data: " prefix
```
Input:  event: message
Expect: nil (skip non-data lines)
```

### T-SSE-14: parseBuffer with multiple lines
```
Input:  "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n"
Expect: 2 chunks, content "a" and "b"
```

---

## Unit Tests: StreamingDeltaProcessor

### T-PROC-1: Think tag routing — content outside tags
```
Input:  receiveDelta("Hello world")
Expect: turn.content contains "Hello world", turn.thinking is empty
```

### T-PROC-2: Think tag routing — thinking inside tags
```
Input:  receiveDelta("<think>Let me reason</think>Answer")
After finalize:
Expect: turn.thinking contains "Let me reason", turn.content contains "Answer"
```

### T-PROC-3: Partial tag across chunks
```
Input:  receiveDelta("prefix<thi"), receiveDelta("nk>inside</think>after")
After finalize:
Expect: turn.content contains "prefix" + "after", turn.thinking contains "inside"
```

### T-PROC-4: Case insensitive tags
```
Input:  receiveDelta("<THINK>reasoning</THINK>content")
After finalize:
Expect: turn.thinking contains "reasoning", turn.content contains "content"
```

### T-PROC-5: Finalize with open thinking tag
```
Input:  receiveDelta("<think>still thinking"), finalize()
Expect: turn.thinking contains "still thinking" (no crash, content routed by isInsideThinking state)
```

### T-PROC-6: Adaptive tuning — interval scales with output
```
Scenario: Send 3000 characters of content
Expect: flushIntervalMs changed from 50 to 75, maxBufferSize from 256 to 512
```

### T-PROC-7: Sync throttling — first content syncs immediately
```
Input:  receiveDelta("first")
Expect: syncCount increments to 1 after first flush (syncCount == 0 triggers immediate sync)
```

### T-PROC-8: Reset clears all state
```
Setup: Process some deltas, then call reset(turn: newTurn)
Expect: deltaBuffer == "", isInsideThinking == false, pendingTagBuffer == "", contentLength == 0
```

### T-PROC-9: Empty delta ignored
```
Input:  receiveDelta("")
Expect: No state change, no buffer growth
```

### T-PROC-10: Fallback timer fires for stale buffer
```
Input:  receiveDelta("small"), wait 150ms
Expect: Content flushed and synced to turn by fallback timer
```

---

## Integration Tests: VMLXService.streamWithTools()

### T-STREAM-1: Normal content stream
```
Setup: Mock engine returns 5 SSE chunks with content "Hello ", "world", "!", finish_reason "stop", [DONE]
Expect: Stream yields batched content (may be combined), total == "Hello world!"
```

### T-STREAM-2: Reasoning + content stream (showThinking = true)
```
Setup: Mock engine returns reasoning_content then content. modelOptions["disableThinking"] = false
Expect: Stream yields "<think>", reasoning text, "</think>", content text
```

### T-STREAM-3: Reasoning discarded (showThinking = false, default)
```
Setup: Mock engine returns reasoning_content + content. No disableThinking option set
Expect: Stream yields only content, no <think> tags
```

### T-STREAM-4: Tool call accumulation and dispatch
```
Setup: Mock engine returns tool_calls deltas then finish_reason "tool_calls"
Expect: Stream yields StreamingToolHint sentinels, then throws ServiceToolInvocation with first tool
```

### T-STREAM-5: Stream cancellation propagates to engine
```
Setup: Start stream, cancel consumer after 3 deltas
Expect: streamTask.cancel() called, URLSession connection closed
Verify: Engine log shows "Client disconnected, aborting request"
```

### T-STREAM-6: Engine crash mid-stream (S5)
```
Setup: Kill engine process after 3 deltas
Expect: bytes.lines throws, error caught at line 266, generationDidFinishAsync() called, partial content preserved in yielded deltas
```

### T-STREAM-7: Engine HTTP error (non-200)
```
Setup: Engine returns HTTP 500
Expect: VMLXError.engineNotRunning thrown with "HTTP 500 from engine"
```

### T-STREAM-8: Stats tracking through stream
```
Setup: Engine sends usage in every chunk (50 prompt, increasing completion)
Expect: InferenceProgressManager.completionTokens increases, TPS computed correctly
```

### T-STREAM-9: Idle timer reset on normal completion
```
Setup: Complete a stream normally
Expect: VMLXProcessManager.resetIdleTimer() called after stream finishes
```

### T-STREAM-10: Idle timer reset on error
```
Setup: Stream throws error
Expect: VMLXProcessManager.resetIdleTimer() called even on error path
```

---

## Bug Verification Tests

### T-BUG-S2: reasoning_content leaked to external API
```
Setup: External client sends POST /v1/chat/completions (stream=true). Model generates reasoning + content.
Expect (current — BROKEN): External client receives <think>reasoning</think>content in delta.content
Expect (fixed): External client receives reasoning in delta.reasoning_content, content in delta.content
```

### T-BUG-S12: Parallel tool calls — only first dispatched
```
Setup: Engine returns 2 tool calls: get_weather + get_time
Expect: StreamingToolHint emitted for both, ServiceToolInvocation thrown for get_weather only
Verify: get_time is NOT executed (known limitation)
```

### T-BUG-S13: Think tag not closed on tool_calls finish
```
Setup: Engine sends reasoning_content followed by finish_reason "tool_calls"
Expect (current — BUG): hasEmittedThinkOpen is true, return at line 246 skips </think> close
Expect (fixed): </think> emitted before tool call sentinels
```

---

## Edge Case Tests

### T-EDGE-1: Empty content deltas (S10)
```
Setup: Engine sends chunk with only usage, no content
Expect: No empty string yielded to consumer. shouldFlush condition requires !contentBatch.isEmpty || !reasoningBatch.isEmpty
```

### T-EDGE-2: Very long tool arguments (S11)
```
Setup: Engine sends 100KB of tool call arguments across many deltas
Expect: AccumulatedToolCall.arguments grows without error, ServiceToolInvocation contains full JSON
```

### T-EDGE-3: Mixed reasoning + content in one batch window (S3, S5)
```
Setup: Reasoning ends and content begins within same 30ms window
Expect: Yield order is: reasoning → </think> → content. StreamingDeltaProcessor correctly parses the transition.
```

### T-EDGE-4: Keep-alive during long thinking (S14)
```
Setup: Engine sends : keep-alive comments during 30s of reasoning
Expect: Parser returns nil for keep-alive lines, URLSession doesn't timeout (keep-alive resets timeout)
Note: URLSession default timeout is 60s. Keep-alive at ~15s intervals prevents timeout.
```

### T-EDGE-5: Stream ends without [DONE] (S19)
```
Setup: Close TCP connection without sending [DONE]
Expect: for-await loop exits normally (no throw), content up to that point is yielded
Current behavior: Silent truncation — no indicator to user
Desired: Detect truncation and indicate incomplete response
```

### T-EDGE-6: 30ms batch vs engine stream-interval double-buffering (S6)
```
Setup: Engine at --stream-interval 3, 80 tok/s → ~27 events/sec. Swift batches at 30ms (~33 max yields/sec)
Expect: ~27 events per second, each batch catches ~1 event. Effectively 1:1 passthrough with minimal latency added.
Measurement: Verify average batch size is 1-2 deltas, not accumulating excessively.
```

---

## Performance Tests

### T-PERF-1: Batching reduces async hops
```
Setup: Engine at 80 tok/s, 1000 tokens total
Measure: Count continuation.yield() calls
Expect: ~330 yields (at 33/sec for 10s), NOT ~1000 (per-token)
```

### T-PERF-2: syncIfNeeded throttles UI updates
```
Setup: 80 tok/s, 2000 characters total
Measure: Count syncToTurn() calls
Expect: ~40-100 syncs (at 4-10/sec), NOT ~330 (per-flush)
```

### T-PERF-3: Date() syscall reduction
```
Measure: In StreamingDeltaProcessor, Date() called every 4th delta (deltasPerTimeCheck = 4)
At 80 tok/s → ~20 Date() calls/sec instead of 80
VMLXService uses CFAbsoluteTimeGetCurrent() instead of Date() for batch timing.
```

### T-PERF-4: TTFT measurement accuracy (S15)
```
Setup: Cold start (engine launch takes 10s), then first token at 10.5s
Expect: generationDidStartAsync() at stream open (after engine ready), TTFT = 0.5s (not 10.5s)
Current: TTFT measures from generationDidStartAsync (line 154) to first token — correct for warm engine.
If engine was launched by ensureEngineRunning (line 124), the 0-120s launch time is NOT included. This is correct.
```

---

## Inactivity Timeout Test (S18 — MISSING FEATURE)

### T-TIMEOUT-1: Engine hangs — no tokens for 60s
```
Setup: Engine alive but stops emitting tokens (deadlock, infinite loop)
Current: URLSession blocks for 300s (default timeout)
Desired: Inactivity watchdog at 30-60s cancels stream and returns error
Reference: RemoteProviderService.nextByte() uses TaskGroup racing with Task.sleep
```

---

## Engine Compatibility Tests (Python ↔ Swift Contract)

### T-COMPAT-1: ensure_ascii round-trip
```
Setup: Engine sends content with emoji: data: {"choices":[{"delta":{"content":"\u3053\u3093\u306b\u3061\u306f\ud83c\udf0d"}}]}
Expect: Swift JSONSerialization decodes to "こんにちは🌍"
Note: Engine uses ensure_ascii=True (server.py:4611), so all multi-byte chars are \uXXXX escaped
```

### T-COMPAT-2: exclude_none — missing fields handled
```
Setup: Engine sends chunk with only content (no reasoning_content, no tool_calls, no usage):
  data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}
Expect: chunk.reasoningContent == nil, chunk.toolCalls == nil, chunk.usage == nil (not error)
```

### T-COMPAT-3: reasoning_content field name (NOT reasoning)
```
Setup: Engine sends: data: {"choices":[{"delta":{"reasoning_content":"thinking","content":null}}]}
Expect: chunk.reasoningContent == "thinking"
Verify: Parser reads "reasoning_content" key, NOT "reasoning" (engine serializes via @computed_field)
```

### T-COMPAT-4: Engine error chunk (NEW BUG)
```
Setup: Engine sends error: data: {"id":"xxx","object":"chat.completion.chunk","error":{"message":"OOM","type":"server_error","code":"internal_error"}}
Current (BROKEN): Parser returns nil (no "choices" key), error silently lost
Expected (fixed): Parser returns chunk with error info, VMLXService surfaces the error to consumer
```

### T-COMPAT-5: Keep-alive prevents URLSession timeout
```
Setup: Engine silent for 14s during prefill, then sends ": keep-alive\n\n"
Expect: Parser returns nil for keep-alive. URLSession does NOT timeout (default 60s, keep-alive at 15s resets the idle timer)
Verify: Complete stream after 30s prefill succeeds
```

### T-COMPAT-6: Usage in every chunk (not just final)
```
Setup: Engine sends usage in 3 consecutive content chunks with increasing completion_tokens: 5, 10, 15
Expect: InferenceProgressManager receives all 3 updates, TPS computed from token count growth
Verify: stream_options.include_usage is true in Swift request body
```

### T-COMPAT-7: Tool call marker detection engine-side
```
Setup: Model outputs text containing tool call markers (e.g., "<function=" or "to=get_weather code{")
Expect: Engine buffers and parses tool calls, emits finish_reason "tool_calls" with structured tool_calls array
Swift: Accumulates via AccumulatedToolCall, dispatches first tool
```

### T-COMPAT-8: Engine request timeout not set by Swift
```
Setup: Swift sends request body without "timeout" field
Expect: Engine uses _default_timeout (from CLI --timeout flag or default None)
Gap: If no timeout set, generation runs indefinitely. Swift has no inactivity watchdog either (S18).
```

### T-COMPAT-9: Engine error with partial usage (M8 feature)
```
Setup: Engine errors mid-stream after generating 50 tokens
Expect: Engine sends error chunk PLUS usage chunk with partial token counts (server.py:5025-5039)
Swift (current): Error usage chunk ignored (no "choices" key). Partial stats lost.
```

### T-COMPAT-10: Reasoning suppression when enable_thinking not sent
```
Setup: Swift NEVER sends enable_thinking (VMLXService.swift:391-394). Model is Gemma 4 (thinking is architectural).
Expect: Engine auto-detects and emits reasoning_content in delta. Swift showThinking flag controls display.
Verify: No enable_thinking in request body JSON
```

---

## Test Infrastructure Notes

- **Mock engine:** Create a lightweight FastAPI or `python3 -m http.server` that emits canned SSE responses
- **VMLXSSEParser tests:** Pure unit tests, no network needed. Feed strings to `parse(line:)`
- **StreamingDeltaProcessor tests:** Needs mock `ChatTurn` with `appendContent`/`appendThinking`/`notifyContentChanged` tracking
- **Integration tests:** Need either mock HTTP server or actual engine instance
- **Performance tests:** Use `XCTMetric` or manual timing with `CFAbsoluteTimeGetCurrent()`
