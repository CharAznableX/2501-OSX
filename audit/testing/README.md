# Testing Audit

## How It Works

### Existing Tests

Tests in `Packages/OsaurusCore/Tests/`:

**Deleted (MLX Swift removal):**
- `Tests/Service/MLXGenerationEngineTests.swift`
- `Tests/Service/KVCacheStoreTests.swift`
- `Tests/Service/StreamAccumulatorTests.swift`
- `Tests/Service/ToolDetectionTests.swift`
- `Tests/Model/ModelRuntimeMappingTests.swift`
- `Tests/Model/ModelRuntimeFallbackTests.swift`
- `Tests/Model/ModelRuntimePrefixTests.swift`

**Remaining:**
- `Tests/Networking/ServerConfigurationStoreTests.swift` — updated for new fields
- `Tests/Memory/PrefixHashTests.swift` — PrefixHash.compute

### Test Infrastructure

- `ChatEngine` takes injected `services` array and `installedModelsProvider` closure for testability
- `ChatEngineProtocol` allows `HTTPHandler` to be tested with mock engines
- `InferenceProgressManager._testMake()` creates isolated instances for testing

---

## What Needs Testing

### Unit Tests (New — Required)

| # | Test | Target | Priority |
|---|------|--------|----------|
| UT1 | **VMLXSSEParser** — parse content chunks, reasoning chunks, tool call chunks (first + continuation), usage stats, finish_reason, [DONE], keep-alive comments, malformed JSON, empty lines | VMLXSSEParser.swift | High |
| UT2 | **VMLXEngineConfig.buildArgs** — verify all 28+ settings map to correct CLI flags. Test: continuous batching off (no cache flags), per-model parser overrides global, KV quant "none" (no flag), speculative model, PLD | VMLXEngineConfig.swift | High |
| UT3 | **StreamingDeltaProcessor** — `<think>` tag parsing: open, close, split across chunks, partial tags, nested (malformed), case sensitivity. Adaptive flush tuning. Finalize with pending buffer. | StreamingDeltaProcessor.swift | High |
| UT4 | **VMLXService.stripThinkingBlocks** — complete blocks, multiline, multiple occurrences, `[THINK]` format, no blocks, incomplete tags, nested tags | VMLXService.swift | Medium |
| UT5 | **AccumulatedToolCall** — single tool, multiple tools, continuation chunks, empty args, large args | VMLXService.swift | Medium |
| UT6 | **VMLXGateway** — register (dual-key), unregister (removes all aliases), port lookup (exact, case-insensitive, suffix), count, allInstances | VMLXGateway.swift | Medium |
| UT7 | **InferenceProgressManager** — generationDidStart (reset), updateStats (TPS calculation, TTFT), generationDidFinish (keeps stats), showStats toggle | InferenceProgressManager.swift | Medium |
| UT8 | **ServerConfiguration decoder** — backward compat (missing fields get defaults), all fields roundtrip encode/decode | ServerConfiguration.swift | Medium |
| UT9 | **ModelOptionsStore** — save/load roundtrip, missing key returns nil, 3-key fallback resolution | ModelOptionsStore.swift | Low |

### Integration Tests

| # | Test | Scope | Priority |
|---|------|-------|----------|
| IT1 | **Engine launch + health check** — spawn real engine, verify /health returns 200, verify gateway registration | VMLXProcessManager + VMLXGateway | High |
| IT2 | **SSE streaming end-to-end** — launch engine, send request, verify SSE chunks arrive, content is correct | VMLXService | High |
| IT3 | **Tool call end-to-end** — send request with tools, verify engine returns tool_calls, verify ServiceToolInvocation thrown | VMLXService | High |
| IT4 | **Reasoning end-to-end** — send request to reasoning model, verify reasoning_content in SSE, verify `<think>` tags in stream | VMLXService | High |
| IT5 | **Engine crash + restart** — launch engine, kill process, verify monitor detects, verify restart | VMLXProcessManager | Medium |
| IT6 | **Idle sleep + wake** — launch engine, trigger soft/deep sleep, send new request, verify engine wakes | VMLXProcessManager | Medium |
| IT7 | **Model eviction** — load model A, switch to model B with strictSingleModel, verify A is stopped | VMLXService + VMLXProcessManager | Medium |

### End-to-End Tests per Model Family

| # | Model Family | Test Focus |
|---|-------------|------------|
| E2E1 | **Qwen 3/3.5** | Tool calling (qwen parser), reasoning (qwen3 parser), thinking mode, large context |
| E2E2 | **Gemma 4** | Architectural thinking (always on), tool calling (gemma4 parser), VLM images |
| E2E3 | **Llama 3.x** | Tool calling (llama parser), no reasoning, standard generation |
| E2E4 | **Mistral/Codestral** | Tool calling (mistral parser), reasoning (mistral parser), code generation |
| E2E5 | **DeepSeek R1** | Reasoning (deepseek_r1 parser), long thinking, `<think>` blocks in output |
| E2E6 | **MiniMax M2.5** | Tool calling (minimax parser), large context |
| E2E7 | **Nemotron** | Tool calling (nemotron parser), reasoning, speculative decoding |
| E2E8 | **JANG quantized** | TurboQuant auto-detection, correct weight loading, 2-3 bit precision |
| E2E9 | **VLM models** | Image input (base64 + URL), mixed text+image, multi-image |

### HTTP API Tests

| # | Test | Format |
|---|------|--------|
| API1 | **OpenAI streaming** — POST /chat/completions stream=true, verify SSE format, content chunks, [DONE] | OpenAI |
| API2 | **OpenAI non-streaming** — POST /chat/completions stream=false, verify JSON response | OpenAI |
| API3 | **OpenAI tool calls** — tools in request, verify tool_calls in response | OpenAI |
| API4 | **Ollama /chat** — POST /chat, verify NDJSON format, done field | Ollama |
| API5 | **Ollama /tags** — GET /tags, verify model list format | Ollama |
| API6 | **Anthropic /messages** — POST /messages, verify event sequence | Anthropic |
| API7 | **Open Responses** — POST /responses, verify event types | OpenResponses |
| API8 | **Auth** — verify loopback bypass, verify LAN requires token, verify invalid token rejected | All |
| API9 | **CORS** — verify headers present, verify preflight OPTIONS works | All |

### Performance Tests

| # | Test | Metric |
|---|------|--------|
| PT1 | **Streaming throughput** — measure tokens/sec at Swift layer vs Python engine direct | TPS |
| PT2 | **TTFT overhead** — measure time from send() to first token, compare warm vs cold engine | Latency |
| PT3 | **Memory usage** — measure Swift app memory during streaming at 80+ tok/s | Memory |
| PT4 | **MainActor contention** — profile MainActor task queue depth during high-speed streaming | Concurrency |

---

## Current Status

| Area | Status | Notes |
|------|--------|-------|
| Unit tests for new VMLX code | **Missing** | All MLX tests deleted, no VMLX replacements |
| ServerConfigurationStore tests | Working | Updated for new fields |
| PrefixHash tests | Working | SHA256 hashing |
| Integration tests | **Missing** | No end-to-end engine tests |
| API format tests | **Missing** | No SSE/NDJSON/Anthropic format verification |
| Performance benchmarks | **Missing** | No automated perf measurement |
