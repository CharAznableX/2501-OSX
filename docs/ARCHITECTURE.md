# VMLXRuntime Architecture

Last updated: 2026-04-01

This document describes the architecture that is actually present on the current branch. Older versions of this file mixed live code, planned work, and aspirational status.

---

## 1. Topology

```text
Osaurus App
  -> VMLXServiceBridge
    -> VMLXService
      -> VMLXRuntimeActor
        -> VMLXModelContainer
        -> Scheduler
        -> CacheCoordinator
        -> SSMReDeriver
        -> native model implementations
          -> mlx-swift / Metal
```

Important runtime truth:

- `ChatEngine` tries `VMLXServiceBridge` before legacy `MLXService`.
- `VMLXRuntimeActor` is the hot-path generation engine.
- `Scheduler` and `BatchBuilder` exist, but the actor still runs one active generation task at a time.
- The branch is not yet doing true concurrent multi-request continuous batching in production flow.

---

## 2. Native Model Surface

Implemented native model families on this branch:

- Standard transformer path in `Models/StandardModel.swift`
- Qwen 3.5 hybrid SSM path in `Models/Qwen35Model.swift`
- NemotronH native path in `Models/NemotronHModel.swift`
- Mistral Small 4 native path in `Models/Mistral4Model.swift`
- GPT-OSS native path in `Models/GPTOSSModel.swift`

Current confidence level differs by family:

- Standard transformer and Qwen 3.5 are the most proven paths here.
- NemotronH and Mistral4 are no longer "not implemented"; they are implemented and under active correctness validation after a recent fix run.

---

## 3. Cache Design

### 3.1 Core Abstraction

The branch correctly treats hybrid caching as a first-class type problem:

- `LayerCacheEntry.attention(KVCacheLayer)`
- `LayerCacheEntry.ssm(SSMStateLayer)`
- `LayerCacheEntry.compressedAttention(...)`

`HybridCache` wraps the per-layer entries and enforces the important rule:

- attention KV is positional and can be truncated
- SSM state is path-dependent and cannot be truncated safely

### 3.2 Fetch Order

`CacheCoordinator.fetch()` currently tries:

1. paged cache
2. memory cache
3. prefix cache
4. disk cache
5. miss

For hybrid models, cache reuse is only complete when the attention KV side and the SSM side both exist for the same boundary.

### 3.3 Prefix vs Paged Reality

The branch does not use every cache layer equally:

- When paged cache is enabled, the simple trie prefix cache is disabled.
- The real fast-path prefix reuse is the paged block hash chain.
- Memory cache remains active as an LRU hot tier.
- Disk cache is exact-token-keyed L2 persistence.

### 3.4 `gen_prompt_len` Handling

The actor strips generation-prompt suffix tokens from cache keys before lookup and store. This is one of the most important practical fixes for multi-turn reuse because it prevents assistant header tokens from poisoning the cache key.

---

## 4. Hybrid SSM Rules

### 4.1 What Is Correctly Enforced

- SSM layers are non-truncatable.
- Hybrid cache entries carry attention and SSM state separately.
- SSM companion state is stored in `SSMStateCache`.
- Cache keys use stable prompt tokens, not generation suffix tokens.

### 4.2 What The Hot Path Still Does Conservatively

If a hybrid request gets attention KV but not matching SSM companion state:

- `CacheCoordinator` returns `.partialHit`
- `VMLXRuntimeActor.generateStream()` does a full prefill
- the request re-derives SSM state as a side effect of that full forward pass
- the new SSM companion state is stored for future turns

This means the branch currently prefers correctness over clever partial hybrid reuse.

### 4.3 `SSMReDeriver`

`SSMReDeriver` exists as an actor with:

- sync vs async decision logic
- task deduplication by token hash
- model-container hookup
- checkpoint storage back into `SSMStateCache`

But the main generation branch does not yet hand hybrid partial hits over to it as the primary recovery path. It is implemented infrastructure, not fully integrated hot-path behavior.

### 4.4 Current Snapshot Strategy

The original plan emphasized mid-prefill checkpointing. The current runtime instead uses:

- single-phase prefill for hybrid models
- post-prefill SSM snapshot capture

That change was made because the earlier more precise multi-phase path caused severe hangs in SSD/Mamba2 prefill.

---

## 5. TurboQuant

### 5.1 What Exists

The Swift side now contains real implementations for:

- `TurboQuantConfig`
- `TurboQuantEncoder`
- `TurboQuantKVCache`
- `EncodedKeys`
- `EncodedValues`
- `TQDiskStore`
- Hadamard, QJL, bit-packing, and codebook helpers

### 5.2 What The Runtime Uses Today

The hot path currently does:

1. prefill with float KV
2. snapshot original float KV for later cross-turn store
3. encode KV with TurboQuant
4. decode once back into float buffers
5. continue decode using those float buffers

This means:

- TurboQuant is already being used for memory reduction on the current request
- the branch intentionally avoids storing lossy TQ-decoded KV as the primary cross-turn cache artifact
- cache store remains conservative to avoid quality drift across turns

### 5.3 What Is Not Fully Closed Yet

- true compressed-attention-first cross-turn reuse is not the main persistence path
- the runtime still keeps the hot path conservative around quality and recovery

---

## 6. Scheduler Reality

The repo contains real scheduling primitives:

- `RequestQueue`
- `Scheduler`
- `BatchBuilder`
- `MLLMScheduler`
- auto-detected scheduler config

But the production actor still cancels any in-flight generation before starting another. So the branch has scheduler infrastructure and batch construction code, while full continuous batching remains incomplete at the runtime-actor level.

---

## 7. Osaurus App Integration

The app-side integration is live:

- `VMLXServiceBridge` adapts VMLXRuntime to Osaurus `ToolCapableService`
- `ChatEngine` puts `VMLXServiceBridge` ahead of legacy `MLXService`
- `ModelManager` merges VMLX-discovered models with MLX models
- `ConfigurationView` forwards Local Inference settings into `VMLXRuntimeActor`
- `ModelCacheInspectorView` can show and unload the VMLX-loaded model

This branch is not an isolated engine prototype anymore; it is wired into the app.

---

## 8. Recent Fixes That Matter

Recent commits changed the practical architecture story:

- `466dcc9a`: native NemotronH model landed
- `8a0a7d05`: TurboQuant decode-once lifecycle landed
- `b34f28ea`: SSD parallel scan replaced slow sequential Mamba2 prefill loop
- `b479140a`: SSD state projection fixed with efficient 4D matmul
- `44506ac0`: hybrid models moved to single-phase prefill with snapshot capture
- `5a7e1315`: NemotronH and Mistral4 inference corrections aligned with Python reference
- `582a6e8b`: follow-up audit fixes validated recent bug reports

---

## 9. Current Gaps

The most important remaining gaps are:

- true multi-request continuous batching in the hot path
- full `SSMReDeriver` integration for hybrid partial-hit recovery
- broader end-to-end validation for NemotronH and Mistral4 on real models
- full vision encoder inference
- MiniMax tokenizer compatibility

---

## 10. Source Of Truth

When docs and code disagree, use these files as the source of truth:

- `Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift`
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift`
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/SSMReDeriver.swift`
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/TurboQuantEncoder.swift`
- `Packages/OsaurusCore/Services/Inference/VMLXServiceBridge.swift`
- `Packages/OsaurusCore/Services/Chat/ChatEngine.swift`
