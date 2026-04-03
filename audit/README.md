# Osaurus + vmlx-engine Production Audit

Systematic audit of every integration point between the Swift app and Python vmlx-engine.

## Branch Context

- **Branch:** `feature/osaurus-vmlx-py` (9 commits ahead of main v0.15.18)
- **Main is fully contained** — no missing features to port
- **`feature/vmlx`** (Swift-native approach) — NOT relevant, we are replacing it

## Structure

Each subfolder contains a comprehensive README covering:
- **How it works** — architecture, data flow, functions involved with file:line references
- **What needs checking** — prioritized issues with specific file/line references
- **Current status** — working / bug / missing / needs verify

## Categories

| Folder | Scope | Items |
|--------|-------|-------|
| [`streaming/`](streaming/README.md) | SSE parsing, content batching, think tag wrapping, adaptive flush, scroll | 17 items |
| [`process-management/`](process-management/README.md) | Subprocess spawning, health checks, crash restart, idle sleep, orphans | 12 items |
| [`caching/`](caching/README.md) | Prefix/paged/disk/block-disk cache, KV quantization, TurboQuant, stats | 13 items |
| [`tool-calling/`](tool-calling/README.md) | Tool call parsing, accumulation, dispatch, 15+ parsers, MCP | 13 items |
| [`reasoning-thinking/`](reasoning-thinking/README.md) | Reasoning parsers, thinking toggle, enable_thinking, history stripping | 13 items |
| [`configuration/`](configuration/README.md) | 28+ ServerConfiguration fields, persistence, CLI mapping, validation | 16 items |
| [`api-endpoints/`](api-endpoints/README.md) | OpenAI/Ollama/Anthropic/OpenResponses API, auth, CORS, MCP HTTP | 15 items |
| [`ui-ux/`](ui-ux/README.md) | Stats display, model picker, thinking toggle, parser chips, loading states | 15 items |
| [`error-handling/`](error-handling/README.md) | VMLXError types, crash recovery, partial response, user-facing messages | 14 items |
| [`model-management/`](model-management/README.md) | Discovery, JANG format, VLM, downloads, gateway routing, eviction | 14 items |
| [`lifecycle/`](lifecycle/README.md) | App startup, shutdown, session management, sleep/wake, memory pressure | 14 items |
| [`security/`](security/README.md) | API auth, process isolation, HF_TOKEN, CORS, network exposure | 13 items |
| [`testing/`](testing/README.md) | Unit tests, integration tests, E2E per model family, API format tests | 35 items |
| [`engine-sync/`](engine-sync/README.md) | Bundled vs upstream parity, CLI compat, version tracking, build system | 14 items |
| [`performance/`](performance/README.md) | Streaming throughput, MainActor contention, memory profiling, optimizations | 13 items |

**Total: ~200 audit items across 15 categories.**

## Priority

1. **Streaming** — timeout, crash detection, reasoning_content leak (worst UX when broken)
2. **Process Management** — health polling gap, app quit orphans, cold start UX
3. **Tool Calling** — parallel tool dispatch limitation
4. **Error Handling** — user-facing messages, partial response preservation
5. **UI/UX** — loading progress, stats polish, engine status accuracy
6. **Caching** — stats endpoint, memory slider UX, cache warm
7. **Configuration** — maxTokens save bug, validation warnings
8. **API Endpoints** — reasoning_content forwarding, Anthropic thinking blocks
9. **Engine Sync** — version tracking, automated sync process
10. Everything else

## Known Bugs (Across All Categories)

| Bug | Category | Severity |
|-----|----------|----------|
| reasoning_content leaked as `<think>` tags to external API clients | streaming, api-endpoints | High |
| Only first parallel tool call dispatched | tool-calling | Medium |
| maxTokens not saved from ConfigurationView | configuration | Medium |
| No periodic health monitor (stale green dot for crashed engine) | process-management, ui-ux | Medium |
| App quit may orphan Python processes (async stopAll in sync callback) | lifecycle | Medium |
| Crash restart doesn't pass modelOptions (parsers lost on restart) | process-management | Medium |
| No loading indicator during 0-120s engine cold start | ui-ux | Medium |

## Missing Features (Across All Categories)

| Feature | Category |
|---------|----------|
| Loading progress bar during model load | ui-ux |
| Memory budget check before loading models | model-management, lifecycle |
| Periodic health monitor between requests | process-management |
| Engine version tracking (upstream commit hash) | engine-sync |
| Automated engine sync process | engine-sync |
| Unit tests for VMLX code | testing |
| Anthropic thinking blocks in /messages API | api-endpoints |
| Engine capabilities discovery endpoint | engine-sync |
