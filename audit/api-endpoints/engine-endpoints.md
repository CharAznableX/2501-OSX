# API Endpoints Audit

## Engine Endpoints (from server.py)

Complete list of endpoints exposed by vmlx-engine, with Swift integration status.

### Health & Admin

| Endpoint | Method | Swift Status | Notes |
|----------|--------|-------------|-------|
| `/health` | GET | **USED** | Startup polling in `VMLXProcessManager.waitForHealth()`. Not polled after startup. |
| `/admin/soft-sleep` | POST | **USED** | `VMLXProcessManager.softSleep()` via idle timer |
| `/admin/deep-sleep` | POST | **USED** | `VMLXProcessManager.deepSleep()` via idle timer |
| `/admin/wake` | POST | **NOT USED** | Engine auto-wakes on request. Could use for proactive pre-warm. |

### OpenAI-Compatible (v1)

| Endpoint | Method | Swift Status | Notes |
|----------|--------|-------------|-------|
| `/v1/chat/completions` | POST | **USED** | Primary inference endpoint. Streaming SSE. |
| `/v1/completions` | POST | **NOT USED** | Non-chat completion. Not needed for chat app. |
| `/v1/models` | GET | **NOT USED** | Returns loaded model info. Could validate post-startup. |
| `/v1/embeddings` | POST | **NOT USED** | Text embeddings. `EmbeddingService.swift` is stub. |

### Cache Management

| Endpoint | Method | Swift Status | Notes |
|----------|--------|-------------|-------|
| `/v1/cache/stats` | GET | **NOT USED** | Memory usage, hit rates, block counts. High value for stats UI. |
| `/v1/cache/entries` | GET | **NOT USED** | List cached prefixes. Debug tool. |
| `/v1/cache/warm` | POST | **NOT USED** | Pre-warm with system prompt. Could improve TTFT on first message. |
| `/v1/cache` | DELETE | **NOT USED** | Clear all caches. Useful for debug. |

### Ollama-Compatible (api/)

| Endpoint | Method | Swift Status | Notes |
|----------|--------|-------------|-------|
| `/api/tags` | GET | **NOT USED** | List models (Ollama format) |
| `/api/ps` | GET | **NOT USED** | Running model status |
| `/api/version` | GET | **NOT USED** | Engine version |
| `/api/show` | POST | **NOT USED** | Model details |
| `/api/chat` | POST | **NOT USED** | Chat (Ollama format) |
| `/api/generate` | POST | **NOT USED** | Generate (Ollama format) |
| `/api/embeddings` | POST | **NOT USED** | Embeddings (Ollama format) |
| `/api/embed` | POST | **NOT USED** | Embeddings alt (Ollama format) |
| `/api/pull` | POST | **NOT USED** | Pull model (stub) |
| `/api/delete` | POST | **NOT USED** | Delete model (stub) |
| `/api/copy` | POST | **NOT USED** | Copy model (stub) |
| `/api/create` | POST | **NOT USED** | Create model (stub) |

### Media (skipped in bundled copy)

| Endpoint | Method | Swift Status | Notes |
|----------|--------|-------------|-------|
| `/v1/images/edits` | POST | **NOT USED** | Image generation — stripped from bundled engine |
| `/v1/audio/transcriptions` | POST | **NOT USED** | Audio transcription — stripped |
| `/v1/audio/speech` | POST | **NOT USED** | TTS — stripped |
| `/v1/audio/voices` | GET | **NOT USED** | Voice list — stripped |

### MCP (skipped in bundled copy)

| Endpoint | Method | Swift Status | Notes |
|----------|--------|-------------|-------|
| `/v1/mcp/tools` | GET | **NOT USED** | MCP tool list — stripped |
| `/v1/mcp/servers` | GET | **NOT USED** | MCP server list — stripped |
| `/v1/mcp/execute` | POST | **NOT USED** | MCP tool execution — stripped |

## Key Functions

### Swift Side (HTTP calls)
- `VMLXService.streamWithTools()` — POST `/v1/chat/completions` (streaming)
- `VMLXProcessManager.waitForHealth()` — GET `/health` (polling)
- `VMLXProcessManager.softSleep()` — POST `/admin/soft-sleep`
- `VMLXProcessManager.deepSleep()` — POST `/admin/deep-sleep`

### What Should Be Wired

1. **`GET /v1/cache/stats`** — Poll after each generation, feed to InferenceProgressManager. Show cache hit rate, memory usage.
2. **`POST /v1/cache/warm`** — Call after engine startup with current chat's system prompt + tool definitions.
3. **`GET /v1/models`** — Call after startup to verify model loaded correctly. Could extract context length, supported features.
4. **`POST /admin/wake`** — Call proactively when user opens chat window (before they type) to pre-warm from deep sleep.
5. **`DELETE /v1/cache`** — Add button in ModelCacheInspectorView for manual cache clear.

## API Key Considerations

- **Upstream**: Most endpoints have `dependencies=[Depends(verify_api_key)]`
- **Bundled copy**: Admin endpoints (`soft-sleep`, `deep-sleep`) may NOT have api_key dependency (check server.py diff)
- **Swift side**: No API key sent in requests. Works because engine started without `--api-key` flag.
- **Risk**: If someone exposes port to network, no auth. Not a concern for loopback-only.
- **Action**: Verify bundled server.py admin endpoints match upstream re: api_key deps.
