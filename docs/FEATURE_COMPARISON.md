# VMLXRuntime vs VMLX Python

Last updated: 2026-04-01

This file now tracks the current branch honestly. Older versions overstated some planned or partially integrated work as fully shipped.

## Legend

- `VERIFIED`: implemented and used with good confidence on this branch
- `IMPLEMENTED`: code is present and wired, but broader runtime validation is still in progress
- `PARTIAL`: important pieces exist, but the end-to-end production path is still conservative or incomplete
- `PENDING`: not done yet

---

## Snapshot

| Area | Python VMLX | Swift VMLXRuntime | Status | Notes |
|------|-------------|-------------------|--------|-------|
| Model loading and detection | Mature | Native loader + detector + JANG support | VERIFIED | Real Osaurus integration uses this path |
| Standard transformers | Mature | `StandardTransformerModel` | VERIFIED | Llama/Qwen paths are the most proven here |
| Qwen 3.5 hybrid SSM | Mature | `Qwen35Model` | VERIFIED | Hybrid cache split/restore path is active |
| NemotronH | Native in Python | `NemotronHModel` | IMPLEMENTED | Recent fixes corrected SSM scan, MoE routing, latent path, and projection dimensions |
| Mistral Small 4 | Native in Python | `Mistral4Model` | IMPLEMENTED | Recent fixes corrected config decoding and inference alignment |
| Cache coordinator | Mature | `CacheCoordinator` | VERIFIED | Paged + memory + prefix + disk + SSM companion |
| Hybrid cache safety | Mature | explicit `HybridCache` / `SSMStateLayer` rules | VERIFIED | SSM remains non-truncatable |
| `gen_prompt_len`-aware cache keys | Present | present | VERIFIED | Actor strips generation suffix before cache lookup/store |
| SSM re-derive actor | N/A / different handling | `SSMReDeriver` | PARTIAL | Implemented, but not the main recovery branch yet |
| TurboQuant encode/decode | Mature | `TurboQuantEncoder` + helpers | IMPLEMENTED | Swift encode/decode exists now |
| TurboQuant runtime usage | Mature | post-prefill encode/decode-once path | PARTIAL | Used for memory reduction, not yet the full cross-turn story |
| Continuous batching | Mature | scheduler primitives exist | PARTIAL | Actor still runs one active generation at a time |
| Vision preprocessing | Mature | `VisionProcessor` + cache | IMPLEMENTED | Encoder inference still pending |
| Osaurus app integration | N/A | bridge + routing + settings + discovery | VERIFIED | VMLX is wired into the app, not a side experiment |

---

## What Changed Since The Old Doc

The earlier version of this document was stale in a few important ways:

- It still described `TurboQuantEncoder` as a stub. It is no longer a stub.
- It described `SSMReDeriver` as fully wired into the main recovery path. It is not.
- It described continuous batching as if the runtime hot path were already using it. It is not.
- It did not reflect the native NemotronH and Mistral4 work that has since landed.

---

## Recent Branch Fix Log

These commits explain most of the current runtime shape:

| Commit | Meaning |
|--------|---------|
| `582a6e8b` | verified audit fixes after recent hybrid-model bug hunt |
| `5a7e1315` | corrected NemotronH and Mistral4 inference against Python reference |
| `44506ac0` | moved hybrid models to single-phase prefill plus snapshot capture |
| `b479140a` | fixed SSD state projection with efficient 4D matmul |
| `b34f28ea` | replaced sequential Mamba2 prefill with SSD parallel scan |
| `466dcc9a` | landed native NemotronH model |
| `8a0a7d05` | landed TurboQuant decode-once lifecycle |

---

## Current Gaps Relative To Python

| Gap | Status |
|-----|--------|
| Full multi-request continuous batching | PENDING |
| `SSMReDeriver` used as the main hybrid partial-hit recovery path | PENDING |
| Broader real-model validation for NemotronH | PENDING |
| Broader real-model validation for Mistral4 | PENDING |
| Full vision encoder inference | PENDING |
| MiniMax tokenizer compatibility | PENDING |

---

## Source Files To Trust

- `Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift`
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift`
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/SSMReDeriver.swift`
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/TurboQuantEncoder.swift`
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Models/NemotronHModel.swift`
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Models/Mistral4Model.swift`
- `Packages/OsaurusCore/Services/Inference/VMLXServiceBridge.swift`
