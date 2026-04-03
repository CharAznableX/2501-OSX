# Model Management Audit

## How It Works

### Model Discovery

**File:** `Managers/Model/ModelManager.swift`

`scanLocalModels()` scans 3 directories:
1. Primary models dir (bookmark or `~/MLXModels`)
2. `UserModelDirectories` from UserDefaults (e.g., `~/.mlxstudio/models`, `~/jang/models`)
3. `~/MLXModels` (always, if exists)

Handles both flat (`root/ModelName/`) and nested (`root/org/repo/`) structures. Follows symlinks.

Accepts `tokenizer_config.json` (JANG models) in addition to standard tokenizer files.

**Model IDs** are filesystem paths for local models. `MLXModel.localDirectory` detects absolute path IDs and returns them directly.

### Model Resolution

**File:** `Services/Inference/VMLXService.swift` (lines 407-421)

`resolveModel(requestedModel)`:
1. Trim whitespace
2. `ModelManager.findInstalledModel(named: trimmed)` → returns `(name, id)` if found
3. Fallback: first discovered model (`ModelManager.discoverLocalModels().first`)

Returns `(name: String, path: String)` where path = filesystem path (model ID).

### Model Picker

**File:** `Views/Model/ModelPickerView.swift`

- Groups models by source (local, remote providers)
- `SearchService` for fuzzy filtering
- Each row: model name, parameter count, quantization, VLM badge
- Selection writes to `ChatSession.selectedModel`

**Caching:** `ModelPickerItemCache` singleton builds and caches items. Posts `localModelsChanged` notification. `FloatingInputCard` snapshots items when picker opens to prevent refresh during streaming.

### Model Detail

**File:** `Views/Model/ModelDetailView.swift`

- Stats: parameter count, quantization, VLM badge
- Parser configuration card (tool + reasoning pickers)
- Download controls with progress from `ModelManager`
- Loads per-model options on appear

### JANG Format

JANG-quantized models from JANGQ-AI use:
- Custom 2-3 bit quantization via `jang_loader.py`
- `tokenizer_config.json` instead of standard tokenizer files
- TurboQuant: automatic 3-bit KV cache compression (engine auto-detects)
- Curated list: https://huggingface.co/collections/jangq/jang-quantized-gguf-for-mlx

### Hub Integration

**File:** `Services/HuggingFaceService.swift`

- `estimateTotalSize(repoId:patterns:)` — hits HF API to sum file sizes
- `fetchModelDetails(repoId:)` — author, downloads, likes, license, pipeline tag, model type, VLM detection
- Uses `Hub` product from `swift-transformers` (kept after mlx-swift removal)

### Gateway (Running Models)

**File:** `Services/Inference/VMLXGateway.swift`

- Dual-key registration: `modelName` AND `modelPath`
- Three-tier lookup: exact → case-insensitive → suffix match
- `allInstances()` returns all running `VMLXInstance`s

### Eviction Policy

**File:** `Models/Configuration/ServerConfiguration.swift` (lines 326-340)

```swift
enum ModelEvictionPolicy: String, Codable {
    case strictSingleModel  // "Strict (One Model)" — auto-unload others
    case manualMultiModel   // "Flexible (Multi Model)" — keep all loaded
}
```

In `VMLXService.ensureEngineRunning()` (line 449):
```swift
if config.modelEvictionPolicy == .strictSingleModel {
    await VMLXProcessManager.shared.stopAll()
}
```

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| M1 | **Model ID consistency** — Picker uses filesystem path as model ID. Gateway registers under both name and path. ModelOptionsStore saves under model ID. Verify all lookups use consistent keys. |
| M2 | **Spaces in model paths** — Model directory could have spaces (e.g., `/Users/eric/MLX Models/Qwen3`). Verify Process arguments handle this correctly (no shell word splitting). |
| M3 | **JANG format auto-detection** — How does the engine know a model is JANG format? Does it check weight file headers? Or config.json fields? |
| M4 | **Model download progress** — Hub downloads show progress. But engine model loading (weight loading, compilation) has no progress indicator. |

### Discovery

| # | Issue | Notes |
|---|-------|-------|
| M5 | **Symlink handling** — Scanner follows symlinks. What about circular symlinks? Or symlinks to non-existent targets? |
| M6 | **Non-ASCII paths** — Model directories with unicode characters (e.g., Chinese model names). Verify URL/string handling throughout the pipeline. |
| M7 | **Nested vs flat detection** — Scanner handles both `root/Model/` and `root/org/repo/`. What if both exist? (e.g., `~/MLXModels/qwen/Qwen3-8B/` AND `~/MLXModels/Qwen3-8B/`) |
| M8 | **Hot-reload model list** — When user downloads a new model or adds a models directory, does the picker update automatically? Or only on app restart? |

### VLM (Vision)

| # | Issue | Notes |
|---|-------|-------|
| M9 | **VLM content serialization** — `buildRequestBody()` serializes `contentParts` with `image_url` arrays. Supports base64 and URL images. Verify both paths work. |
| M10 | **Multiple images** — Can user send multiple images in one message? Does the engine handle this? |
| M11 | **video_url support** — No `video_url` in Swift `MessageContentPart`. Known limitation. |

### Remote Models

| # | Issue | Notes |
|---|-------|-------|
| M12 | **Remote model routing** — `ModelServiceRouter` checks remote services first. If a remote model name collides with a local model name, remote wins. Is this intentional? |
| M13 | **Foundation Model routing** — `FoundationModelService` handles `nil`, `""`, `"default"`, `"foundation"`. On macOS without Apple Intelligence, what happens? |
| M14 | **Bonjour discovery** — Ephemeral providers removed when agent goes offline. But what about network disconnects without clean shutdown? |

---

## Functions Reference

| Function | File | Purpose |
|----------|------|---------|
| `ModelManager.scanLocalModels()` | ModelManager.swift | Scan directories for models |
| `ModelManager.findInstalledModel(named:)` | ModelManager.swift | Find model by name/path |
| `ModelManager.installedModelNames()` | ModelManager.swift | List all model names |
| `ModelManager.discoverLocalModels()` | ModelManager.swift | Full discovery |
| `VMLXService.resolveModel()` | VMLXService.swift:407 | Map request → (name, path) |
| `VMLXGateway.register()` | VMLXGateway.swift:33 | Dual-key registration |
| `VMLXGateway.port(for:)` | VMLXGateway.swift:55 | Three-tier lookup |
| `ModelPickerItemCache` | ModelPickerItemCache.swift | Cached picker items |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Local model discovery | Working | Multi-dir, flat+nested, symlinks |
| JANG model support | Working | tokenizer_config.json accepted |
| Model picker | Working | Grouping, search, VLM badge |
| Gateway dual-key | Working | Name + path registration |
| Three-tier lookup | Working | Exact, case-insensitive, suffix |
| Eviction policy | Working | Strict single model default |
| Hub integration | Working | Size estimation, metadata |
| VLM image support | Working | contentParts with image_url |
| Video support | **Missing** | No video_url MessageContentPart |
| Loading progress | **Missing** | No progress during weight loading |
| Memory budget check | **Missing** | No pre-load RAM verification |
