<h1 align="center">Jangosaurus</h1>

<p align="center">
  <strong>VMLXRuntime: Native Swift Inference Engine for Osaurus</strong><br>
  <em>Development build -- this engine will replace mlx-swift-lm and all external model dependencies in the main <a href="https://github.com/osaurus-ai/osaurus">Osaurus</a> project.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Dev%20Build-yellow" alt="Status">
  <img src="https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-black?logo=apple" alt="Platform">
  <img src="https://img.shields.io/badge/Swift-6.2-orange?logo=swift" alt="Swift">
  <img src="https://img.shields.io/badge/MLX-Metal%20GPU-blue" alt="MLX">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

> **Development branch.** VMLXRuntime is being built as a standalone native inference engine to replace `mlx-swift-lm` and all external model library dependencies in [Osaurus](https://github.com/osaurus-ai/osaurus). Once validated end-to-end, this engine will be merged upstream as the production Osaurus inference backend -- giving Osaurus its own engine with zero third-party model library dependencies.

---

## Goal

Replace Osaurus's `mlx-swift-lm` dependency with a native Swift inference engine that:
- Has **zero external model library dependencies** (only mlx-swift for tensor ops + swift-transformers for tokenization)
- Supports **all model architectures natively** (standard transformers, hybrid SSM, MoE, MLA)
- Includes **production-grade caching** (5-layer stack with paged, prefix, memory, disk, SSM companion)
- Handles **JANG mixed-precision quantization** (per-layer 2/4/6/8-bit auto-detection)
- Provides **TurboQuant 3-bit KV compression**, continuous batching, tool/reasoning parsing, VL support, power management, and multi-model gateway

---

## Current Status

### Working (verified with real models)

| Model | Type | Status |
|-------|------|--------|
| **Llama 3.2 1B** (MLX 4-bit) | Standard transformer | Loads, generates coherent text |
| **Qwen 2.5 0.5B** (MLX 4-bit) | Standard transformer | Loads, generates coherent text |
| **Qwen 3.5 4B/9B/27B** (JANG 2-bit) | Hybrid SSM + GQA | Loads, generates coherent text, SSM cache splits correctly |
| **Qwen 3.5 35B/122B MoE** (JANG) | Hybrid SSM + MoE | Architecture loads (not tested on this machine -- too large) |
| **MiniMax M2.5** (JANG 2-bit) | MoE (256 experts) | Loads and runs forward pass, but generates garbage due to swift-transformers tokenizer incompatibility with MiniMax's unusual special tokens |

### Not Yet Working

| Model | Issue | What's Needed |
|-------|-------|---------------|
| **Mistral Small 4 (119B)** | FP8 quantization format (`_scale_inv`, fused `gate_up_proj`) | FP8 dequantization support -- completely different weight format from JANG/MLX |
| **NemotronH** | Not implemented | Dedicated hybrid SSM architecture |
| **MiniMax M2.5 text quality** | swift-transformers can't encode MiniMax's special tokens (`]~!b[`, `[e~[`) | Custom tokenizer path or swift-transformers fix |

### Infrastructure Done

- 5-layer cache stack (memory, prefix, disk, paged, SSM companion) -- verified with roundtrip tests
- Hybrid SSM cache splitting (24 SSM + 8 attention for Qwen3.5) -- verified
- Disk cache safetensors serialization -- verified roundtrip
- Per-layer mixed-precision auto-quantization (infers bits AND group_size from weight shapes)
- Model switching (unloads previous model before loading new one)
- Reasoning parser (always active for models with `<think>` in chat template)
- Settings UI: Local Inference panel with sampling, KV cache, cache stack (TurboQuant toggle, disk cache toggle, memory budget)
- Model cache inspector with unload button (wired to both old MLXService and VMLXRuntime)

### Still Needed Before Merge

- [ ] MiniMax tokenizer fix (custom encoding path for unusual special tokens)
- [ ] FP8 quantization support (Mistral 4, other FP8 models)
- [ ] NemotronH hybrid architecture
- [ ] TurboQuant Metal kernels (codebook encode/decode -- stubs exist)
- [ ] Vision encoder forward pass (preprocessing done, encoder integration pending)
- [ ] Full model unload/reload UI polish
- [ ] Multi-turn cache reuse in generation loop (CacheCoordinator fetch before prefill)

---

## Architecture

```
Osaurus App (SwiftUI)
  |
  v
VMLXServiceBridge (ToolCapableService)     -- drop-in for MLXService
  |
  v
VMLXRuntimeActor (singleton)               -- replaces ModelRuntime
  |
  +-- ModelLoader + VMLXModelRegistry      -- native model construction
  |     +-- StandardTransformerModel       -- Llama, Qwen2/3, Mistral, Gemma, Phi, MiniMax MoE
  |     +-- Qwen35TopLevelModel            -- Qwen3.5 hybrid (GatedDeltaNet + GQA + MoE)
  |
  +-- VMLXModelContainer (weights + tokenizer + metadata)
  +-- Scheduler (continuous batching)
  +-- CacheCoordinator (5-layer stack)
  +-- SSMReDeriver (async recovery)
  |
  v
mlx-swift (tensor ops) --> MLX C++ --> Metal GPU
```

### Weight Loading (zero external dependency)

1. Read `config.json` for `model_type` and quantization
2. `VMLXModelRegistry` creates the correct Module subclass
3. Load all `.safetensors` files
4. `sanitize(weights:)` -- model-specific key remapping (VL wrapper strip, conv1d transpose, norm shift)
5. Auto-quantize: infer per-layer bits AND group_size from `weight.dim * 32 / (scales.dim * group_size)`, swap `Linear` to `QuantizedLinear`
6. `model.update(parameters:)` loads weights

Handles JANG mixed-precision (different layers use different bits AND different group sizes).

---

## Package Structure

```
Packages/VMLXRuntime/           81 source files, 17,719 lines
  Sources/VMLXRuntime/
    Core/          8 files   -- Types, HybridCache, ModelLoader, ModelDetector, ModelConfig
    Cache/        13 files   -- 5-layer stack, TQ disk store, SSM companion, coordinator
    Quantization/  6 files   -- TurboQuant, JANG loader (7 profiles)
    Models/       14 files   -- StandardModel, Qwen3.5, MoE, MLA, Hybrid, GatedDelta kernel
    Generation/    5 files   -- Sampler, stop detector, stream accumulator, PLD
    Scheduler/     5 files   -- Continuous batching, request queue, batch builder
    Vision/        3 files   -- CoreImage processor, embedding cache, 7 VLM architectures
    Parsers/      19 files   -- 14 tool parsers + 3 reasoning parsers
    Integration/   3 files   -- VMLXRuntimeActor, VMLXService, ChatMessageMapper
    API/           4 files   -- Anthropic, Ollama, Completions, Embeddings adapters
  Tests/           44 files, 7,781 lines
```

---

## Model Support

### StandardTransformerModel (StandardModel.swift)

Handles all standard HuggingFace decoder-only transformers:
- **Architectures**: Llama 2/3/4, Qwen 2.5/3, Mistral (non-FP8), Gemma 2/3, Phi 3/4, StarCoder 2, InternLM 2, Granite, Cohere, MiniMax M2.5
- **Features**: GQA attention, SwiGLU MLP, MoE (via SwitchGLU), q/k norm, RoPE (default/linear/llama3/yarn/longrope/su-scaled), partial rotation, optional attention bias, tied embeddings, 2/4/6/8-bit quantization

### Qwen35TopLevelModel (Qwen35Model.swift)

Handles Qwen3.5 hybrid SSM models:
- **Architectures**: Qwen3.5-4B/9B/27B (dense), Qwen3.5-35B/122B (MoE)
- **Features**: GatedDeltaNet SSM (custom Metal kernel), full GQA attention, SwitchGLU MoE, shared expert, RMSNormGated, conv1d, language_model wrapper sanitization

### JANG Quantization

All 7 profiles from [JANGQ-AI](https://huggingface.co/JANGQ-AI): JANG_1L, JANG_2L, JANG_2S, JANG_3M, JANG_4K, JANG_4M, JANG_4S.

---

## Key Features

| Feature | Status | Notes |
|---------|--------|-------|
| 5-layer cache stack | Done, verified | Memory + prefix + disk + paged + SSM companion |
| TurboQuant 3-bit KV | Config + lifecycle done | Metal kernels pending |
| Continuous batching | Scheduler done | Single-request generation active |
| 14 tool call parsers | Done | Auto-detected from model name |
| 3 reasoning parsers | Done | Always active for thinking models |
| Power management | Done | softSleep/deepSleep/wake/JIT |
| Multi-model gateway | Done | Load by alias, route by name |
| Mid-prefill SSM checkpoint | Architecture done | Integration pending |
| Vision preprocessing | Done | CoreImage resize/normalize |
| Vision encoder inference | Pending | VLMModelProtocol defined, not implemented |

---

## Settings (Local Inference)

Settings -> Local Inference panel:

| Setting | Controls |
|---------|----------|
| Top P | Nucleus sampling threshold |
| Max Context Length | Maximum KV cache tokens |
| Cache Bits | KV cache quantization (2-8 bit) |
| Group Size | Quantization group size |
| Prefill Step | Tokens per prefill chunk |
| TurboQuant toggle | 3-bit KV compression |
| Disk Cache toggle | L2 SSD persistence |
| Memory Cache Budget | RAM fraction for cache (default 30%) |

All settings flow through: ConfigurationView -> ServerConfiguration -> RuntimeConfig -> VMLXServiceBridge -> VMLXRuntimeActor -> SchedulerConfig.

---

## Building

```bash
# Build VMLXRuntime standalone
cd Packages/VMLXRuntime && swift build

# Run tests (requires Xcode for Metal)
xcodebuild test -scheme VMLXRuntime -destination 'platform=macOS'

# Build full app (Xcode 16.4+, macOS 15.5+, Apple Silicon)
open osaurus.xcworkspace  # Cmd+R
```

---

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** -- Full architecture with code map
- **[FEATURE_COMPARISON.md](docs/FEATURE_COMPARISON.md)** -- Feature comparison with VMLX Python

---

## Credits

- **VMLXRuntime** -- Jinho Eric Jang
- **Osaurus** -- [osaurus-ai](https://github.com/osaurus-ai/osaurus) (Terence Pae / tpae)
- **MLX** -- Apple
- **JANG Quantization** -- [JANGQ-AI](https://huggingface.co/JANGQ-AI)
