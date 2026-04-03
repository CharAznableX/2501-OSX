# Osaurus — vmlx-engine Integration

## Architecture

Osaurus is a native macOS SwiftUI app. Inference is handled by **vmlx-engine**, a Python MLX backend running as a subprocess per model.

```
SwiftNIO gateway (port 1337) → ChatEngine → VMLXService → HTTP POST → Python engine (random port)
                                                          ← SSE stream ←
```

### Key Components

| Component | File | Role |
|-----------|------|------|
| VMLXService | Services/Inference/VMLXService.swift | HTTP client to Python engine, SSE parser, tool call accumulation, thinking tag wrapping, stats tracking |
| VMLXProcessManager | Services/Inference/VMLXProcessManager.swift | Process spawning, health polling (with crash detection), idle sleep, restart backoff (max 3), SIGKILL escalation |
| VMLXGateway | Services/Inference/VMLXGateway.swift | Model-to-port registry (actor) |
| VMLXEngineConfig | Services/Inference/VMLXEngineConfig.swift | Maps ServerConfiguration + per-model options → CLI args |
| VMLXSSEParser | Services/Inference/VMLXSSEParser.swift | Parses SSE lines: content, reasoning_content, tool_calls (incremental), usage stats, finish_reason, keep-alive |
| PrefixHash | Services/Inference/PrefixHash.swift | SHA256 hash for prefix cache keying (extracted from deleted ModelRuntime) |
| InferenceProgressManager | Managers/InferenceProgressManager.swift | Observable stats: TPS, TTFT, cache hits, token counts. Toggle via UserDefaults "showInferenceStats" |
| ChatEngine | Services/Chat/ChatEngine.swift | Routes requests to VMLXService or FoundationModelService |
| ServerConfiguration | Models/Configuration/ServerConfiguration.swift | All settings with defaults (28+ engine fields) |
| ConfigurationView | Views/Settings/ConfigurationView.swift | Settings UI for all engine options + stats toggle |
| ModelDetailView | Views/Model/ModelDetailView.swift | Model info card with per-model parser configuration |
| FloatingInputCard | Views/Chat/FloatingInputCard.swift | Chat input bar: model picker, thinking toggle, parser config chip, stats display |

### Python Engine Source

- `Resources/vmlx_engine/` — stripped engine source (no audio/mcp/image_gen/gradio/commands)
- `scripts/bundle-python.sh` — builds relocatable Python 3.12 + all deps
- `Resources/bundled-python/` — output of bundle script (gitignored, ~400-640MB)

### Process Lifecycle

1. User sends message → `VMLXService.ensureEngineRunning()`
2. `resolveModel()` maps requested model → `(name, path)` via `ModelManager.findInstalledModel`
3. Checks `VMLXGateway` for running instance (by name AND path)
4. If not running: `VMLXProcessManager.launchEngine()` spawns Python on random port
5. Health check polls `/health` every 2s for up to 120s — checks `process.isRunning` each iteration (surfaces crash immediately via `VMLXError.engineCrashed` with last stderr line)
6. Registers with `VMLXGateway`
7. HTTP POST to `http://127.0.0.1:<port>/v1/chat/completions`
8. SSE stream parsed and yielded as `AsyncThrowingStream<String>`

### Model Discovery

`ModelManager.scanLocalModels()` scans 3 directories:
1. Primary models dir (bookmark or `~/MLXModels`)
2. `UserModelDirectories` from UserDefaults (e.g. `~/.mlxstudio/models`, `~/jang/models`)
3. `~/MLXModels` (always, if exists)

Handles both flat (`root/ModelName/`) and nested (`root/org/repo/`) structures. Follows symlinks. Accepts `tokenizer_config.json` (JANG models) in addition to standard tokenizer files.

Model IDs are **filesystem paths** for locally discovered models. `MLXModel.localDirectory` detects absolute path IDs and returns them directly.

### Parser Configuration

Parsers are configured **per-model** (not global):
- **Chat input bar**: wrench icon chip → popover with Tool Parser + Reasoning Parser pickers → saved via `ModelOptionsStore`
- **Model detail card**: Parser Configuration section with same pickers → saved under `model.id`
- **Engine launch**: `VMLXEngineConfig.buildArgs()` reads per-model options, falls back to global config

Parser option key resolution in `ensureEngineRunning`:
1. Try `requestedModel` (picker item ID = filesystem path)
2. Try `resolved.path`
3. Try `resolved.name` (lowercased display name)

### Parser Auto-Detection

Both `--tool-call-parser auto` and `--reasoning-parser auto` (defaults) trigger Python's `model_config_registry` to detect the right parser from the model's `config.json` `model_type` field.

### Thinking / Reasoning

- `enable_thinking` is only sent when user explicitly sets `disableThinking` in model options
- Otherwise Python engine auto-detects per model (avoids breaking Gemma 4 where thinking is architectural)
- `reasoning_content` from SSE chunks is wrapped in `<think>...</think>` tags by VMLXService so `StreamingDeltaProcessor` routes it to `appendThinking()`
- Prior assistant messages have `<think>` and `[THINK]` blocks stripped in `buildRequestBody` to prevent history contamination

### Inference Stats

- `VMLXSSEParser` extracts `VMLXUsage` from every SSE chunk (prompt_tokens, completion_tokens, cached_tokens, cache_detail)
- `VMLXService` streaming loop feeds `InferenceProgressManager.shared.updateStatsAsync()` on each chunk
- `InferenceProgressManager` computes: TPS (token count growth over time), TTFT (first content token - generation start), cache hits
- `FloatingInputCard` displays stats chip (right-aligned in selector row) when `showStats` is enabled
- Toggle: Settings → Local Inference → Stats Display → "Show Inference Stats"
- Stats persist after generation until next generation starts

### Settings → CLI Flag Mapping

All 28+ settings in ServerConfiguration map to Python CLI flags via `VMLXEngineConfig.buildArgs()`. See the file for the complete mapping. Per-model parser options override global config.

### Idle Sleep

- Soft sleep: `POST /admin/soft-sleep` — clears GPU caches, model stays loaded
- Deep sleep: `POST /admin/deep-sleep` — unloads model from VRAM
- Auto-wake: next request triggers automatic reload
- Timer configured via `idleSleepMode` and `idleSleepMinutes` in ServerConfiguration

### Process Safety

- **Env isolation**: `PYTHONNOUSERSITE=1`, `PYTHONPATH=""`, `-s` flag, clean env with only HOME/PATH/TMPDIR/DYLD/Metal
- **HF_TOKEN**: Passed through for gated models (Llama 3, Gemma)
- **Orphan detection**: `lsof -ti :<port>` before launch, kills stale processes
- **Crash restart**: Max 3 retries with exponential backoff (2s, 4s, 6s), gives up after that
- **Shutdown**: SIGTERM → 1.5s grace → SIGKILL
- **Startup crash detection**: Health poller checks `process.isRunning` each iteration, surfaces stderr on crash

### Recommended Models

Curated list is the JANGQ collection: https://huggingface.co/collections/jangq/jang-quantized-gguf-for-mlx

Includes Qwen 3.5 (4B-397B), Gemma 4, Mistral Small 4, MiniMax M2.5, Nemotron Cascade/Super in various JANG quantization levels.

### Build

```bash
scripts/bundle-python.sh    # Build bundled Python (once, ~10 min)
xcodebuild -workspace osaurus.xcworkspace -scheme osaurus -configuration Debug \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO build
```

Dev mode: Falls back to `~/mlx/vllm-mlx/panel/bundled-python/python/bin/python3` if bundled Python not built.

## Known Issues / TODO

- [ ] Parallel tool calls: only first tool call is dispatched via `ServiceToolInvocation`; rest are emitted as hints but not executed
- [ ] `maxTokens` not saved from ConfigurationView (always uses default 32768)
- [ ] No periodic health monitor (crash detected only on next request or during startup)
- [ ] ConfigurationView has orphaned `@State` vars for removed parser pickers (dead code, harmless)

## Files Changed (from main branch)

### Deleted (MLX Swift removal)
- `Services/ModelRuntime/MLXGenerationEngine.swift`
- `Services/ModelRuntime/KVCacheStore.swift`
- `Services/ModelRuntime.swift`
- `Services/ModelRuntime/StreamAccumulator.swift`
- `Services/ModelRuntime/RuntimeConfig.swift`
- `Services/ModelRuntime/Events.swift`
- `Services/Inference/MLXService.swift`
- `Tests/Service/MLXGenerationEngineTests.swift`
- `Tests/Service/KVCacheStoreTests.swift`
- `Tests/Service/StreamAccumulatorTests.swift`
- `Tests/Service/ToolDetectionTests.swift`
- `Tests/Model/ModelRuntimeMappingTests.swift`
- `Tests/Model/ModelRuntimeFallbackTests.swift`
- `Tests/Model/ModelRuntimePrefixTests.swift`

### Created
- `Services/Inference/VMLXService.swift`
- `Services/Inference/VMLXProcessManager.swift`
- `Services/Inference/VMLXGateway.swift`
- `Services/Inference/VMLXEngineConfig.swift`
- `Services/Inference/VMLXSSEParser.swift`
- `Services/Inference/PrefixHash.swift`
- `Resources/vmlx_engine/` (full engine source, stripped)
- `Resources/vmlx_engine/pyproject.toml`
- `scripts/bundle-python.sh`

### Modified
- `Package.swift` — removed mlx-swift, mlx-swift-lm deps; kept swift-transformers for Hub
- `Models/Configuration/ServerConfiguration.swift` — removed old genKV fields, added 28+ vmlx engine fields
- `Models/Configuration/MLXModel.swift` — `localDirectory` handles absolute path IDs; `isDownloaded` accepts `tokenizer_config.json`
- `Managers/Model/ModelManager.swift` — multi-dir scanner, removed SDK allowlist filter, JANGQ curated models
- `Managers/InferenceProgressManager.swift` — added generation stats (TPS, TTFT, cache hits, toggle)
- `Views/Settings/ConfigurationView.swift` — new engine settings UI, stats toggle
- `Views/Chat/FloatingInputCard.swift` — parser config chip, inference stats chip
- `Views/Model/ModelDetailView.swift` — parser configuration card
- `Views/Model/ModelCacheInspectorView.swift` — shows running engine instances
- `Views/Chat/ChatView.swift` — VMLXService refs, removed ModelRuntime refs
- `Views/Common/SimpleComponents.swift` — VMLXGateway.shared.count
- `Services/Chat/ChatEngine.swift` — VMLXService
- `Networking/Router.swift` — VMLXService
- `Networking/HTTPHandler.swift` — VMLXService, PrefixHash.compute
- `Services/Inference/CoreModelService.swift` — VMLXService
- `Services/Plugin/PluginHostAPI.swift` — VMLXService
- `Managers/Chat/ChatWindowManager.swift` — removed ModelRuntime refs
- `AppDelegate.swift` — VMLXProcessManager.shared.stopAll() on quit
- `Services/Memory/EmbeddingService.swift` — comment update
- `Tests/Networking/ServerConfigurationStoreTests.swift` — updated for new fields
- `Tests/Memory/PrefixHashTests.swift` — PrefixHash.compute
