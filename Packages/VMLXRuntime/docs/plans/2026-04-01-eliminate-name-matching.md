# Eliminate Model Name Matching — Refactor Plan

> All model capability detection must use `config.json` `model_type` field
> or user settings. NEVER substring match on model display names.

## Items to Fix (one by one)

### 1. `ModelConfigRegistry.detect()` — ModelConfig.swift:201
**Current:** `name.contains(config.family.lowercased())`
**Fix:** Add a `modelTypes: [String]` field to `ModelFamilyConfig` mapping config.json `model_type` values. Lookup by exact `model_type` match first, fall back to family name only for JANG models where model_type isn't standard.
**Status:** TODO

### 2. `VMLXService.handles()` — VMLXService.swift:75-98
**Current:** 30+ hardcoded family names in array with `.contains()`
**Fix:** Accept ALL local models. Let `ModelLoader.load()` determine if the architecture is supported. If unsupported, the ChatEngine fallback router retries with MLXService. Remove the family name array entirely.
**Status:** TODO

### 3. `StreamingMiddlewareResolver` — StreamingMiddleware.swift:147-152
**Current:** `vmlxFamilies` array with `.contains()` to detect VMLX vs MLX path
**Fix:** Pass a flag from the service layer indicating which runtime is handling the request. Or: don't use middleware for VMLX path at all (VMLXRuntimeActor handles thinking).
**Status:** TODO

### 4. `ModelDetector.detectFamily()` — ModelDetector.swift:494-541
**Current:** Two cascading if-chains with `.contains()` on model_type and source_model
**Fix:** Use the `model_type` field directly from config.json (already parsed). Map `model_type` → family via a dictionary lookup, not substring matching.
**Status:** TODO

### 5. `autoDetectReasoningParser()` — ReasoningParser.swift:30-55
**Current:** `.contains()` on model name
**Fix:** Already partially fixed — VMLXRuntimeActor uses `container.familyConfig.reasoningFormat`. But the standalone function still uses name matching. Remove it or make it use `model_type`.
**Status:** TODO

### 6. `GPTOSSReasoningProfile.matches()` — ModelOptions.swift:141
**Current:** `.contains("gpt-oss")` on model ID
**Fix:** Use `model_type` from the loaded model's config. The UI needs access to the model's `model_type` at profile selection time.
**Status:** TODO

### 7. `QwenThinkingProfile.matches()` — ModelOptions.swift:178
**Current:** `.contains("qwen3")`, `.contains("minimax")`, etc.
**Fix:** Same as #6 — use `model_type` from config.
**Status:** TODO

### 8. `VMLXServiceBridge._isMLXServiceOnlyModel()` — VMLXServiceBridge.swift:264
**Current:** Reads config.json `model_type` and checks against `mlxServiceOnlyTypes` set.
**Fix:** Already correct (uses model_type, not name). Keep as-is.
**Status:** DONE ✓

### 9. `ModelPickerItem` vision detection — ModelPickerItem.swift:379
**Current:** `.contains("vision")` || `.contains("pixtral")`
**Fix:** Use `supportsVision` from `ModelFamilyConfig` or `DetectedModel.hasVision`.
**Status:** TODO

### 10. `MLXService.getAllLocalModels()` / `getAvailableModels()` — MLXService.swift
**Current:** No name matching (discovery-based). OK.
**Fix:** None needed.
**Status:** DONE ✓

## Architecture After Refactor

```
config.json (model_type field)
    ↓
ModelLoader reads model_type
    ↓
ModelConfigRegistry.configForType(modelType:) → ModelFamilyConfig
    ↓
VMLXModelContainer.familyConfig (stored at load time)
    ↓
Used by:
  - VMLXRuntimeActor: toolCallFormat, reasoningFormat, thinkInTemplate
  - UI: thinking toggle, reasoning effort (via model_type → profile mapping)
  - Middleware: thinkInTemplate flag from loaded model config
```

No model name substring matching anywhere in the chain.
