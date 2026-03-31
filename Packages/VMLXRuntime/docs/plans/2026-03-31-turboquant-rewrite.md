# TurboQuant Rewrite — Per-Coordinate Scalar Quantization

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite TurboQuantEncoder to match Python VMLX's per-coordinate scalar quantization algorithm (currently broken: per-vector codebook = 93% reconstruction error).

**Architecture:** Lloyd-Max codebook + Randomized Hadamard rotation + per-coordinate quantization + QJL 1-bit correction for keys. Decode-once lifecycle (compress → decode once → reuse decoded buffer).

**Tech Stack:** MLX (mlx-swift), Swift, Metal (for Hadamard if needed)

---

## Current State

- `TurboQuantEncoder.swift` — BROKEN. Per-vector codebook lookup (1 index per 128-dim vector). Produces garbage reconstruction.
- `EncodedKeys.swift` / `EncodedValues.swift` — Struct fields don't match correct algorithm.
- `TurboQuantKVCache.swift` — Lifecycle wrapper, needs update for decode-once pattern.
- Cache integration in `VMLXRuntimeActor.swift` — Store/restore paths disabled pending TQ fix.

## Algorithm (from Python VMLX at `/Users/eric/jang/jang-tools/jang_tools/turboquant/`)

### Encode Keys (b bits total = (b-1) MSE + 1 QJL)

1. **Normalize**: `norms = ||keys||₂` per vector, `keys_unit = keys / norms`
2. **Hadamard rotate**: `rotated = H(keys_unit * random_signs)` — spreads energy uniformly across coordinates
3. **Per-coordinate MSE quantize**: Each of `head_dim` coordinates → nearest Lloyd-Max centroid (b-1 bits, 2^(b-1) centroids)
4. **QJL residual correction**: `residual = rotated - mse_dequant`, project through random Gaussian S, store 1-bit signs + residual norm
5. **Pack**: indices (b-1 bits each), QJL signs (1 bit each), vector norms, residual norms

### Encode Values (b bits total, MSE only)

1. **Normalize**: same as keys
2. **Hadamard rotate**: same (use seed+1 for different rotation)
3. **Per-coordinate MSE quantize**: b bits per coordinate
4. **Pack**: indices (b bits each), vector norms

### Decode Keys

1. **Unpack** indices + signs
2. **Codebook lookup**: per-coordinate centroid values
3. **QJL correction**: `decoded += sqrt(π/2)/dim * residual_norms * (signs @ S)`
4. **Inverse Hadamard**: `H⁻¹(decoded, random_signs)`
5. **Scale**: `result * vector_norms`

### Decode Values

1. **Unpack** indices
2. **Codebook lookup**
3. **Inverse Hadamard**
4. **Scale by norms**

## Components to Implement

### Task 1: TQCodebook.swift (Lloyd-Max centroids)

Precompute optimal scalar quantization centroids for the Beta((d-1)/2, (d-1)/2) distribution.

- `static func centroids(dim: Int, bits: Int) -> [Float]` — returns 2^bits centroids
- `static func boundaries(dim: Int, bits: Int) -> [Float]` — returns 2^bits-1 decision boundaries
- Cache by (dim, bits) pair
- Port from `codebook.py` (Lloyd-Max iteration on Beta distribution)

### Task 2: TQHadamard.swift (Randomized Hadamard rotation)

Fast Walsh-Hadamard transform in O(d·log d) using MLX ops.

- `static func rotate(_ x: MLXArray, signs: MLXArray) -> MLXArray` — forward transform
- `static func inverseRotate(_ x: MLXArray, signs: MLXArray) -> MLXArray` — inverse
- `static func randomSigns(dim: Int, seed: Int) -> MLXArray` — deterministic ±1 signs
- Handle non-power-of-2 dims via block decomposition (e.g., 192 = 128 + 64)
- Port from `rotation.py`

### Task 3: TQBitPack.swift (bit packing/unpacking)

Pack small-bit indices into uint32 arrays efficiently.

- `static func pack(_ values: MLXArray, bits: Int) -> MLXArray` — pack to uint32
- `static func unpack(_ packed: MLXArray, bits: Int, count: Int) -> MLXArray` — unpack
- Handle alignment (32/bits values per uint32)
- Port from bit packing utilities in `pipeline.py`

### Task 4: TQQJL.swift (QJL projection)

Random Gaussian projection for 1-bit correction on keys.

- `static func encode(residual: MLXArray, seed: Int) -> (signs: MLXArray, norms: MLXArray)`
- `static func decode(signs: MLXArray, norms: MLXArray, seed: Int, dim: Int) -> MLXArray`
- S matrix: generated from seed, shape [dim, dim]
- Scale factor: `sqrt(π/2) / dim`
- Port from `qjl.py`

### Task 5: Rewrite TurboQuantEncoder.swift

Replace broken per-vector algorithm with correct per-coordinate pipeline.

- `encodeKeys(keys:, bits:, seed:, sinkTokens:) -> EncodedKeys`
- `encodeValues(values:, bits:, seed:, sinkTokens:) -> EncodedValues`
- `decodeKeys(_ encoded:, seed:) -> MLXArray`
- `decodeValues(_ encoded:, seed:) -> MLXArray`

### Task 6: Update EncodedKeys/EncodedValues structs

Match new storage format:
- `indicesPacked: MLXArray` — uint32, packed per-coordinate indices
- `qjlSignsPacked: MLXArray` — uint32, packed 1-bit signs (keys only)
- `vectorNorms: MLXArray` — float16, per-vector L2 norms
- `residualNorms: MLXArray` — float16, per-vector QJL residual norms (keys only)
- `shape: [Int]` — original compressed region shape
- `indexBits: Int` — bits per index
- `seed: Int` — for reproducible Hadamard + QJL matrices
- `sinkData: MLXArray?` — full-precision first N tokens

### Task 7: Re-enable TQ in VMLXRuntimeActor cache paths

- Store: compress attention layers via new TQ encoder
- Restore: decode TQ entries and load into KV cache
- SSM layers: skip (unchanged)
- Verify round-trip quality: cosine similarity > 0.95 for 3-bit

### Task 8: Decode-once lifecycle (TurboQuantKVCache update)

Match Python's three-phase lifecycle:
1. **Fill**: standard float KV buffer during prefill
2. **Compress**: encode once after prefill, decode once into persistent buffer
3. **Generate**: new tokens append to float window, decoded buffer is read-only

This avoids re-decoding on every attention call.

## Key Considerations

- **Speed**: Hadamard must be fast (MLX ops, not CPU loops). Codebook lookup via MLX gather.
- **Memory**: Decoded buffer is temporary (same size as original). Packed storage is 5x smaller.
- **Hybrid SSM**: Only attention layers get TQ. SSM layers skip. Two-phase prefill captures SSM snapshot at correct boundary.
- **Sink tokens**: First 4 tokens preserved at full precision (attention sinks).
- **Bits**: Support 2, 3, 4, 8. Default 3-bit for most layers, 4-bit for critical (first/last 3 layers).
- **Disk cache**: Serialized packed format (indices + norms). Decoded on load.

## File Locations

- New: `Quantization/TQCodebook.swift`, `TQHadamard.swift`, `TQBitPack.swift`, `TQQJL.swift`
- Rewrite: `Quantization/TurboQuantEncoder.swift`
- Update: `Quantization/EncodedKeys.swift`, `EncodedValues.swift`
- Update: `Quantization/TurboQuantKVCache.swift`
- Re-enable: `Integration/VMLXRuntimeActor.swift` (store/restore paths)

## Python Source Reference

- `/Users/eric/jang/jang-tools/jang_tools/turboquant/codebook.py` — Lloyd-Max
- `/Users/eric/jang/jang-tools/jang_tools/turboquant/rotation.py` — Hadamard
- `/Users/eric/jang/jang-tools/jang_tools/turboquant/pipeline.py` — Encode/decode
- `/Users/eric/jang/jang-tools/jang_tools/turboquant/qjl.py` — QJL projection
- `/Users/eric/jang/jang-tools/jang_tools/turboquant/config.py` — Config
- `/Users/eric/jang/jang-tools/jang_tools/turboquant/cache.py` — Cache lifecycle
