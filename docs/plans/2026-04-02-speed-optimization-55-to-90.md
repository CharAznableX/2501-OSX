# Speed Optimization Plan: 55 to 90+ tok/s

## Current State
- Osaurus: ~55 tok/s on Qwen3.5-35B-A3B JANG 3-bit, ~38 tok/s on 4-bit MLX
- Ollama MLX: 113 tok/s on same 4-bit model
- Python VMLX: 73 tok/s on same model

## Root Cause Analysis (confirmed via profiling)

### Fix 1: Dedicated Metal Stream for Generation [~5-10 tok/s gain]
Python VMLX wraps all generation in `with mx.stream(generation_stream)` using a dedicated Metal command stream. This isolates generation GPU work from other Metal activity.

File: `VMLXRuntimeActor.swift` — wrap prefill + decode in `Stream.withNewDefaultStream`.
Problem: closure captures non-Sendable types. Need generation Task refactor.
Reference: `/Users/eric/mlx/vllm-mlx/vmlx_engine/mllm_batch_generator.py` line 1116

### Fix 2: Compile Samplers [~2-3 tok/s gain]
Python compiles `categorical_sampling`, `apply_top_p`, `apply_min_p` with `@mx.compile`.

File: `Sampler.swift` — wrap sampling functions in `compile()` with random state tracking.

### Fix 3: Fix Repetition Penalty — Index Gather [~1-2 tok/s gain]
Current: allocates full vocab-size (248K+) Float array on CPU every token.
Python: uses index-gather on just the unique tokens seen.

File: `Sampler.swift` — `applyRepetitionPenalty()` method.

### Fix 4: Submit All Cache States in asyncSubmit [~2-5 tok/s gain]
We only submit `[nextY!]`. Python submits `sampled, logprobs, *cache_states` — giving GPU more work to pipeline.

File: `VMLXRuntimeActor.swift` decode loop — include cache state arrays in async submission.

### Fix 5: TQ getKeys() Concatenation Elimination [~3-5 tok/s gain at long context]
Even with pre-allocated window buffer, `getKeys()` still concatenates decoded prefix + window slice. Need single pre-allocated buffer.

File: `TurboQuantKVCache.swift` — merge decoded prefix into buffer at restore time.

### Fix 6: Remove Memory.clearCache() from Decode Loop [~1 tok/s gain]
Called every 256 tokens. Forces GPU sync. May not be needed with wired memory limit set.

File: `VMLXRuntimeActor.swift` decode loop.

### Fix 7: Compile GatedDelta Step Ops [~5-10 tok/s gain]
Python compiles `_gated_delta_step_ops`. Verify Swift `_compiledGatedDeltaStep` is actually used.

File: `GatedDelta.swift`

## Expected Cumulative Gain
| Fix | Estimated Gain | Difficulty |
|-----|---------------|------------|
| Dedicated stream | 5-10 tok/s | Medium (Sendable refactor) |
| Compile samplers | 2-3 tok/s | Easy |
| Repetition penalty | 1-2 tok/s | Easy |
| Submit all cache states | 2-5 tok/s | Easy |
| TQ getKeys() | 3-5 tok/s | Medium |
| Remove clearCache | 1 tok/s | Easy |
| Compile step ops | 5-10 tok/s | Hard |

Total estimated: 19-36 tok/s => 74-91 tok/s

## Priority Order (by effort/impact)
1. Submit all cache states (easy, high impact)
2. Compile samplers (easy)
3. Repetition penalty (easy)
4. Dedicated stream (medium, high impact)
5. TQ getKeys() (medium)
6. Remove clearCache (easy, verify first)
7. Compile step ops (hard, biggest potential)
