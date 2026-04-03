# API Endpoints Audit

## How It Works

Osaurus exposes an HTTP server via SwiftNIO on port 1337 (configurable). The server speaks multiple API dialects.

### Server Setup

**File:** `Networking/OsaurusServer.swift`

- `MultiThreadedEventLoopGroup` with `activeProcessorCount` threads
- `ServerBootstrap` with SO_REUSEADDR, TCP_NODELAY, backlog 256
- One `HTTPHandler` per child channel
- Started/stopped via `ServerController` (`@MainActor ObservableObject`)

### Route Table

**File:** `Networking/HTTPHandler.swift` (lines 216-370)

Path normalization strips `/v1`, `/api`, `/v1/api` prefixes.

| Method | Path | Handler | Format |
|--------|------|---------|--------|
| GET | `/` | inline | text/plain |
| GET | `/health` | inline | JSON |
| GET | `/models` | `handleModelsEndpoint` | OpenAI |
| GET | `/tags` | `handleTagsEndpoint` | Ollama compat |
| POST | `/show` | `handleShowEndpoint` | Ollama compat |
| POST | `/chat/completions` | `handleChatCompletions` | OpenAI SSE |
| POST | `/chat` | `handleChatNDJSON` | Ollama NDJSON |
| GET | `/mcp/health` | inline | JSON |
| GET | `/mcp/tools` | `handleMCPListTools` | JSON |
| POST | `/mcp/call` | `handleMCPCallTool` | JSON |
| POST | `/messages` | `handleAnthropicMessages` | Anthropic SSE |
| POST | `/audio/transcriptions` | `handleAudioTranscriptions` | OpenAI |
| POST | `/responses` | `handleOpenResponses` | Open Responses |
| POST | `/memory/ingest` | `handleMemoryIngest` | JSON |
| GET | `/agents` | `handleListAgents` | JSON |
| POST | `/agents/{id}/dispatch` | `handleDispatchEndpoint` | JSON |
| GET | `/tasks/{id}` | `handleTaskStatusEndpoint` | JSON |
| DELETE | `/tasks/{id}` | `handleTaskCancelEndpoint` | JSON |
| POST | `/tasks/{id}/clarify` | `handleTaskClarifyEndpoint` | JSON |
| POST | `/embeddings` or `/embed` | `handleEmbeddings` | OpenAI / Ollama |
| `*` | `/plugins/...` | `handlePluginRoute` | Plugin-defined |

### Response Writers

**File:** `Models/Chat/ResponseWriters.swift`

| Writer | Format | Events |
|--------|--------|--------|
| `SSEResponseWriter` | OpenAI SSE | `data: {json}`, `data: [DONE]` |
| `NDJSONResponseWriter` | Ollama NDJSON | `{json}\n` per chunk, `{"done":true}` |
| `AnthropicSSEResponseWriter` | Anthropic | `message_start`, `content_block_delta`, `message_stop` |
| `OpenResponsesSSEWriter` | Open Responses | `response.created`, `response.output_text.delta`, `response.completed` |

### Authentication

**File:** `Networking/HTTPHandler.swift` (lines 141-192)

- Loopback (`127.0.0.1` / `::1`): skip auth when `trustLoopback = true`
- LAN: require `Bearer osk-v1` token validated by `APIKeyValidator`
- Public paths: `/`, `/health` exempt
- Plugin routes: handle own auth

### CORS

- Headers set from `stateRef.value.corsHeaders`
- `allowedOrigins` in ServerConfiguration (empty = disabled, `"*"` = any)
- OPTIONS preflight handled (lines 193-214)

### Streaming Pipeline (handleChatCompletions)

1. Decode `ChatCompletionRequest` from body
2. Inject agent system prompt if `X-Osaurus-Agent-Id` header present
3. Create `ChatEngine(source: .httpAPI)` 
4. If `request.stream == true`: call `engine.streamChat()`, iterate stream, write SSE chunks via `SSEResponseWriter`
5. If not streaming: call `engine.completeChat()`, return JSON response
6. Tool calls: catch `ServiceToolInvocation`, write tool call deltas via writer

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| A1 | **reasoning_content not forwarded** — SSEResponseWriter writes `delta.content` only. If VMLXService wraps reasoning in `<think>` tags, external clients get think tags in content. Should have separate `delta.reasoning_content` field for OpenAI-compatible streaming. |
| A2 | **Anthropic thinking blocks** — `AnthropicSSEResponseWriter` sends `content_block_delta` with `type: text_delta`. Anthropic format uses `type: thinking` for reasoning. Currently not implemented. |
| A3 | **Router.swift dead code** — Legacy router's streaming endpoints return `.internalServerError`. All routes are now handled by HTTPHandler directly. Router.swift should be cleaned up or removed. |

### API Compatibility

| # | Issue | Notes |
|---|-------|-------|
| A4 | **OpenAI SSE format exactness** — Verify: `id` format (`chatcmpl-*`), `object: "chat.completion.chunk"`, `created` timestamp, `model` field reflects requested model, `system_fingerprint` present. |
| A5 | **Ollama `/chat` NDJSON** — Does it handle Ollama-specific fields like `context`, `keep_alive`, `options.num_predict`? |
| A6 | **Ollama `/tags`** — Must return model list in Ollama format: `{"models":[{"name":"...","model":"...","size":...}]}`. |
| A7 | **Open Responses API** — Full event sequence: `response.created`, `response.in_progress`, output items, text deltas, `response.completed`. Verify completeness. |
| A8 | **`/embeddings` endpoint** — Uses VMLXService or separate embedding model? How does it route? |
| A9 | **`/audio/transcriptions`** — Uses FluidAudio for local transcription. Does it handle all audio formats? Max file size? |

### Auth & Security

| # | Issue | Notes |
|---|-------|-------|
| A10 | **Loopback trust** — `trustLoopback = true` means any local process can access the API. This is standard for desktop apps but should be documented. |
| A11 | **API key storage** — Where are `osk-v1` tokens stored? Are they encrypted? |
| A12 | **CORS wildcard** — If `allowedOrigins` contains `"*"`, any website can make requests. Combined with loopback trust, this could be an attack vector. |

### Error Handling

| # | Issue | Notes |
|---|-------|-------|
| A13 | **Model not found error** — What HTTP status code? What error message? Is it consistent across OpenAI/Ollama/Anthropic formats? |
| A14 | **Engine crashed during stream** — If VMLXService throws mid-stream, does the SSE writer properly close? Or does the client hang? |
| A15 | **Request body size limit** — No apparent limit on request body size. Large message arrays or base64 images could exhaust memory. |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| OpenAI chat/completions | Working | Streaming + non-streaming |
| Ollama /chat NDJSON | Working | Basic compat |
| Ollama /tags | Working | Model list |
| Anthropic /messages | Working | Basic streaming |
| Open Responses | Working | Full event sequence |
| MCP endpoints | Working | List + call |
| Authentication | Working | Loopback trust + Bearer token |
| CORS | Working | Configurable origins |
| reasoning_content forwarding | **Bug** | Leaked as think tags |
| Anthropic thinking blocks | **Missing** | Not implemented |
| Router.swift cleanup | **Tech debt** | Dead streaming code |
