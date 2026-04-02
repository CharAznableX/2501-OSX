# Consolidated Audit — All Agents — 2026-04-02

## CRITICAL BUG FOUND & FIXED THIS PASS

### Stream.withNewDefaultStream Causes Cache Corruption
**Files:** VMLXRuntimeActor.swift lines 1065, 1145
**Root cause:** Other agent wrapped prefill and decode in SEPARATE `Stream.withNewDefaultStream` blocks. MLX streams don't share state — KV cache written during prefill (stream A) was invisible to decode (stream B). Result: garbage output on ALL prompts (not just multi-turn).
**Fix:** Removed both stream wraps. All generation ops now run on default stream.
**Impact:** Affected ALL models (Qwen3.5, Gemma, any model). Every request produced garbage.
**Status:** FIXED

---

## ALL CHANGES BY ALL AGENTS (consolidated)

### Speed Optimizations (7 fixes)
| # | Fix | File | Agent | Status |
|---|-----|------|-------|--------|
| G.1 | asyncEval all cache states | VMLXRuntimeActor.swift | Agent 2 | VERIFIED |
| G.2 | Compiled categorical sampler | Sampler.swift | Agent 2 | VERIFIED |
| G.3 | Index-gather repetition penalty | Sampler.swift | Agent 2 | VERIFIED |
| G.4 | Dedicated Metal stream | VMLXRuntimeActor.swift | Agent 2 | REVERTED (caused corruption) |
| G.5 | TQ unified buffer (zero-concat) | TurboQuantKVCache.swift | Agent 1+2 | VERIFIED |
| G.6 | clearCache 256->1024 | VMLXRuntimeActor.swift | Agent 2 | VERIFIED |
| G.7 | GatedDelta kernel fallback | GatedDelta.swift | Agent 2 | VERIFIED |

### Cache & TQ Fixes
| Fix | File | Agent | Status |
|-----|------|-------|--------|
| TQ compressed export (no double-lossy) | TurboQuantKVCache.swift | Agent 2 | VERIFIED |
| TQ compressed restore (direct install) | TurboQuantKVCache.swift | Agent 2 | VERIFIED |
| TQ pre-allocated unified buffer | TurboQuantKVCache.swift | Agent 1+2 | VERIFIED |
| DiskCache file_size on re-store | DiskCache.swift | Agent 1 | VERIFIED |
| Paged cache COW in finalize | CacheCoordinator.swift | Agent 1 | IN CODEBASE |
| Paged cache refCount leak | CacheCoordinator.swift | Agent 1 | IN CODEBASE |
| Reconstruct nil -> placeholder | CacheCoordinator.swift | Agent 1 | IN CODEBASE |

### GatedDelta Metal Kernel
| Fix | File | Agent | Status |
|-----|------|-------|--------|
| 4 kernel variants (scalar/vec x masked) | GatedDelta.swift | Agent 1 | VERIFIED |
| Compiled computeGatedDeltaG | GatedDelta.swift | Agent 1 | VERIFIED |
| Kernel enabled for Qwen3.5 | GatedDelta.swift | Agent 1 | VERIFIED |
| Ops fallback when kernel unavailable | GatedDelta.swift | Agent 2 | VERIFIED |

### Infrastructure
| Fix | File | Agent | Status |
|-----|------|-------|--------|
| Wired memory limit at model load | VMLXRuntimeActor.swift | Agent 1 | VERIFIED |
| Metal cache limit (25% max ws) | VMLXRuntimeActor.swift | Agent 1 | VERIFIED |
| Cmlx import for wired API | VMLXRuntimeActor.swift | Agent 1 | VERIFIED |
| SPM dep conflict (jjang vs ml-explore) | OsaurusCore/Package.swift | Agent 1 | WARNING ONLY |

### Model Detection
| Fix | File | Agent | Status |
|-----|------|-------|--------|
| SSM cross-detect from config.json | ModelDetector.swift | Agent 2 | VERIFIED |
| MoE cross-detect from config.json | ModelDetector.swift | Agent 2 | VERIFIED |
| 9 fields text_config fallback | ModelDetector.swift | Agent 2 | VERIFIED |
| sliding_attention type mapping | ModelContainer.swift | Agent 2 | VERIFIED |
| Hybrid pattern "-" mapping | HybridCache.swift | Agent 2 | VERIFIED |
| Gemma4 family config | ModelConfig.swift | Agent 1+2 | VERIFIED |

### Streaming & UI
| Fix | File | Agent | Status |
|-----|------|-------|--------|
| Partial tag false positives removed | StreamingDeltaProcessor.swift | Agent 2 | VERIFIED |
| Finalize incomplete tags | StreamingDeltaProcessor.swift | Agent 2 | VERIFIED |
| Sync throttle for large content | StreamingDeltaProcessor.swift | Agent 1+2 | VERIFIED |
| Live stats every 20 tokens | VMLXRuntimeActor.swift | Agent 2 | VERIFIED |
| Cache export before finish() | VMLXRuntimeActor.swift | Agent 1 | VERIFIED |
| Unload button state refresh | FloatingInputCard.swift | Agent 1 | VERIFIED |
| Thinking box tail-only render | ThinkingBlockView.swift | Agent 1 | IN CODEBASE |
| Markdown skip during streaming | MarkdownMessageView.swift | Agent 1 | IN CODEBASE |

### Engine Routing & Edge Cases
| Fix | File | Agent | Status |
|-----|------|-------|--------|
| VLM gate (vision models -> MLXService) | VMLXServiceBridge.swift | Agent 2 | VERIFIED |
| Engine attribution in stats | GenerationStats.swift | Agent 2 | VERIFIED |
| GPU OOM memory release | VMLXRuntimeActor.swift | Agent 2 | VERIFIED |
| Empty thinking events | VMLXService.swift | Agent 2 | VERIFIED |
| Build stubs (MissingViewStubs) | MissingViewStubs.swift | Agent 2 | VERIFIED |
| MemoryConfiguration property | MemoryConfiguration.swift | Agent 2 | VERIFIED |

---

## REMAINING WORK

### Gemma 4 (NOT STARTED)
- Dedicated model implementation (~600 lines)
- JANG conversion needs redo for VL + audio
- Plan at: docs/plans/2026-04-02-gemma4-implementation.md

### Speed Gap (55 -> 90+ tok/s)
- Stream.withNewDefaultStream REVERTED (was G.4, biggest estimated gain)
- Need alternative approach: single stream for entire generation, or synchronize between streams
- Remaining estimated gain without stream fix: 10-16 tok/s -> ~65-71 tok/s
- Plan at: docs/plans/2026-04-02-speed-optimization-55-to-90.md

### Settings
- Settings not applying on new chat without restart
- TQ on/off requires model reload

### Tests
- No tests for TQ unified buffer
- No tests for compressed export/restore round-trip
- No tests for GatedDelta kernel dispatch

---

## BUILD STATUS
- Release: BUILD SUCCEEDED (clean)
- Stream wraps: REMOVED (caused garbage output)
- SPM conflict: WARNING only (jjang vs ml-explore mlx-swift identity)

## FILE COUNT
- Total modified: ~40 files across both agents
- Lines changed: ~2000+ / -1200+
