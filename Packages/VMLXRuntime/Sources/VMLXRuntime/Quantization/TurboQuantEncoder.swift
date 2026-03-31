import Foundation
import MLX
import MLXRandom

/// TurboQuant encoder/decoder — per-coordinate scalar quantization with QJL correction.
///
/// Algorithm (matching Python VMLX `jang_tools.turboquant.pipeline`):
///   Keys (b bits = (b-1) MSE + 1 QJL):
///     1. Normalize to unit sphere, store vector norms
///     2. Randomized Hadamard rotation (spreads energy across coordinates)
///     3. Per-coordinate scalar quantize via Lloyd-Max codebook (b-1 bits)
///     4. QJL 1-bit correction on residual (unbiased inner products)
///     5. Pack indices + signs + norms
///   Values (b bits, MSE only):
///     1. Normalize, store norms
///     2. Hadamard rotate
///     3. Per-coordinate scalar quantize (b bits)
///     4. Pack indices + norms
///
/// Reference: TurboQuant (arXiv:2504.19874)
public struct TurboQuantEncoder: Sendable {

    /// Default number of sink tokens preserved at full precision.
    public static let defaultSinkTokens = 4

    // MARK: - Precomputed State

    /// Precomputed encoder state for a given (dim, keyBits, valueBits, seed).
    /// Create once per model layer configuration, reuse across encode/decode calls.
    public struct EncoderState: @unchecked Sendable {
        public let dim: Int
        public let keyBits: Int
        public let valueBits: Int
        public let seed: Int

        /// Hadamard rotation signs (deterministic from seed)
        public let rotationSigns: MLXArray

        /// Lloyd-Max codebook for keys (b-1 bits, 2^(b-1) centroids)
        public let keyCodebook: [Float]
        public let keyIndexBits: Int

        /// Lloyd-Max codebook for values (b bits, 2^b centroids)
        public let valueCodebook: [Float]
        public let valueIndexBits: Int

        /// QJL projection matrix S (dim × dim, for key residual correction)
        public let qjlS: MLXArray

        public init(dim: Int, keyBits: Int = 3, valueBits: Int = 3, seed: Int = 42) {
            self.dim = dim
            self.keyBits = keyBits
            self.valueBits = valueBits
            self.seed = seed

            self.rotationSigns = TQHadamard.generateRandomSigns(dim: dim, seed: seed)

            let kMseBits = max(keyBits - 1, 1)
            self.keyCodebook = TQCodebook.computeCodebook(dim: dim, bits: kMseBits)
            self.keyIndexBits = kMseBits

            self.valueCodebook = TQCodebook.computeCodebook(dim: dim, bits: valueBits)
            self.valueIndexBits = valueBits

            self.qjlS = TQQJL.generateProjection(dim: dim, seed: seed + 1000)
        }
    }

    // MARK: - Encode Keys

    /// Compress float keys to TurboQuant format.
    ///
    /// - Parameters:
    ///   - keys: Float16/32 key tensor, shape [batch, heads, tokens, head_dim]
    ///   - state: Precomputed encoder state (codebooks, rotation signs, QJL matrix)
    ///   - sinkTokens: Number of leading tokens to preserve at full precision (default 4)
    /// - Returns: EncodedKeys with packed per-coordinate indices, QJL signs, and norms
    public static func encodeKeys(
        _ keys: MLXArray,
        state: EncoderState,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedKeys {
        let origShape = keys.shape  // [batch, heads, tokens, head_dim]
        let seqLen = origShape[2]
        let dim = origShape[origShape.count - 1]

        // Extract sink tokens at full precision
        let sinkData: MLXArray?
        let compressKeys: MLXArray
        if sinkTokens > 0 && seqLen > sinkTokens {
            sinkData = keys[.ellipsis, 0..<sinkTokens, 0...]
            compressKeys = keys[.ellipsis, sinkTokens..., 0...]
        } else {
            sinkData = nil
            compressKeys = keys
        }

        let compressShape = compressKeys.shape

        // Step 1: Normalize to unit sphere
        let vectorNorms = (compressKeys * compressKeys).sum(axis: -1, keepDims: true).sqrt()
        let keysUnit = compressKeys / (vectorNorms + 1e-8)

        // Step 2: Randomized Hadamard rotation
        let keysRotated = TQHadamard.hadamardRotate(keysUnit, signs: state.rotationSigns)

        // Step 3: Per-coordinate MSE quantization (b-1 bits)
        let flatRotated = keysRotated.asType(.float32).reshaped([-1, dim])
        let mseIndices = TQCodebook.quantizeScalar(flatRotated, codebook: state.keyCodebook)
        let mseDequant = TQCodebook.dequantizeScalar(mseIndices, codebook: state.keyCodebook)

        // Step 4: QJL 1-bit correction on residual
        let residual = flatRotated - mseDequant
        let projected = matmul(residual, state.qjlS.transposed())
        let qjlSigns = which(projected .>= 0, MLXArray(Float(1.0)), MLXArray(Float(-1.0)))
        let residualNorms = (residual * residual).sum(axis: -1, keepDims: true).sqrt()

        // Step 5: Pack
        let packedIndices = TQBitPack.packBits(mseIndices.reshaped(-1), bits: state.keyIndexBits)
        let packedQJL = TQBitPack.packSigns(qjlSigns.reshaped(-1))

        return EncodedKeys(
            indicesPacked: packedIndices,
            qjlPacked: packedQJL,
            residualNorms: residualNorms
                .reshaped(Array(compressShape.dropLast()) + [1]).asType(.float16),
            vectorNorms: vectorNorms.asType(.float16),
            shape: compressShape,
            indexBits: state.keyIndexBits,
            seed: state.seed,
            sinkData: sinkData
        )
    }

    // MARK: - Decode Keys

    /// Decompress keys from TurboQuant format.
    public static func decodeKeys(_ encoded: EncodedKeys, state: EncoderState) -> MLXArray {
        let origShape = encoded.shape
        let dim = origShape[origShape.count - 1]
        let nElements = origShape.reduce(1, *)

        // Step 1: Unpack
        let flatIndices = TQBitPack.unpackBits(
            encoded.indicesPacked, bits: encoded.indexBits, nElements: nElements
        ).reshaped([-1, dim])
        let flatQJL = TQBitPack.unpackSigns(
            encoded.qjlPacked, nElements: nElements
        ).reshaped([-1, dim])
        let flatResNorms = encoded.residualNorms.asType(.float32).reshaped([-1, 1])
        let flatVecNorms = encoded.vectorNorms.asType(.float32).reshaped([-1, 1])

        // Step 2: Codebook lookup (per-coordinate)
        let mseDequant = TQCodebook.dequantizeScalar(flatIndices, codebook: state.keyCodebook)

        // Step 3: QJL correction
        let qjlScale = Float(Foundation.sqrt(Double.pi / 2.0)) / Float(dim)
        let qjlDequant = MLXArray(qjlScale) * flatResNorms * matmul(flatQJL, state.qjlS)

        // Step 4: Combine MSE + QJL, inverse Hadamard
        let reconstructedRotated = (mseDequant + qjlDequant).reshaped(origShape)
        let reconstructedUnit = TQHadamard.hadamardInverse(
            reconstructedRotated, signs: state.rotationSigns)

        // Step 5: Scale by stored norms
        var decoded = (reconstructedUnit * flatVecNorms.reshaped(
            Array(origShape.dropLast()) + [1])).asType(.float16)

        // Prepend sink tokens if present
        if let sink = encoded.sinkData {
            decoded = concatenated([sink, decoded], axis: 2)
        }

        return decoded
    }

    // MARK: - Encode Values

    /// Compress float values to TurboQuant format (MSE only, no QJL).
    public static func encodeValues(
        _ values: MLXArray,
        state: EncoderState,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedValues {
        let origShape = values.shape
        let seqLen = origShape[2]
        let dim = origShape[origShape.count - 1]

        // Extract sink tokens
        let sinkData: MLXArray?
        let compressValues: MLXArray
        if sinkTokens > 0 && seqLen > sinkTokens {
            sinkData = values[.ellipsis, 0..<sinkTokens, 0...]
            compressValues = values[.ellipsis, sinkTokens..., 0...]
        } else {
            sinkData = nil
            compressValues = values
        }

        let compressShape = compressValues.shape

        // Step 1: Normalize
        let vectorNorms = (compressValues * compressValues).sum(axis: -1, keepDims: true).sqrt()
        let valuesUnit = compressValues / (vectorNorms + 1e-8)

        // Step 2: Hadamard rotate
        let valuesRotated = TQHadamard.hadamardRotate(valuesUnit, signs: state.rotationSigns)

        // Step 3: Per-coordinate MSE quantization (b bits)
        let flatRotated = valuesRotated.asType(.float32).reshaped([-1, dim])
        let mseIndices = TQCodebook.quantizeScalar(flatRotated, codebook: state.valueCodebook)

        // Step 4: Pack
        let packedIndices = TQBitPack.packBits(mseIndices.reshaped(-1), bits: state.valueIndexBits)

        return EncodedValues(
            indicesPacked: packedIndices,
            vectorNorms: vectorNorms.asType(.float16),
            shape: compressShape,
            indexBits: state.valueIndexBits,
            seed: state.seed,
            sinkData: sinkData
        )
    }

    // MARK: - Decode Values

    /// Decompress values from TurboQuant format.
    public static func decodeValues(_ encoded: EncodedValues, state: EncoderState) -> MLXArray {
        let origShape = encoded.shape
        let dim = origShape[origShape.count - 1]
        let nElements = origShape.reduce(1, *)

        // Step 1: Unpack
        let flatIndices = TQBitPack.unpackBits(
            encoded.indicesPacked, bits: encoded.indexBits, nElements: nElements
        ).reshaped([-1, dim])
        let flatVecNorms = encoded.vectorNorms.asType(.float32).reshaped([-1, 1])

        // Step 2: Codebook lookup
        let mseDequant = TQCodebook.dequantizeScalar(flatIndices, codebook: state.valueCodebook)

        // Step 3: Inverse Hadamard
        let reconstructedRotated = mseDequant.reshaped(origShape)
        let reconstructedUnit = TQHadamard.hadamardInverse(
            reconstructedRotated, signs: state.rotationSigns)

        // Step 4: Scale by norms
        var decoded = (reconstructedUnit * flatVecNorms.reshaped(
            Array(origShape.dropLast()) + [1])).asType(.float16)

        // Prepend sink tokens
        if let sink = encoded.sinkData {
            decoded = concatenated([sink, decoded], axis: 2)
        }

        return decoded
    }

    // MARK: - Legacy API (backwards compatible)

    /// Encode keys using seed-based state creation (convenience).
    public static func encodeKeys(
        keys: MLXArray,
        bits: Int = 3,
        seed: Int = 42,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedKeys {
        let dim = keys.dim(keys.ndim - 1)
        let state = EncoderState(dim: dim, keyBits: bits, seed: seed)
        return encodeKeys(keys, state: state, sinkTokens: sinkTokens)
    }

    /// Encode values using seed-based state creation (convenience).
    public static func encodeValues(
        values: MLXArray,
        bits: Int = 3,
        seed: Int = 42,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedValues {
        let dim = values.dim(values.ndim - 1)
        let state = EncoderState(dim: dim, valueBits: bits, seed: seed)
        return encodeValues(values, state: state, sinkTokens: sinkTokens)
    }

    /// Decode keys using seed-based state creation (convenience).
    public static func decodeKeys(_ encoded: EncodedKeys, seed: Int = 42) -> MLXArray {
        let dim = encoded.shape.last ?? 128
        let keyBits = encoded.indexBits + 1  // indexBits = keyBits - 1
        let state = EncoderState(dim: dim, keyBits: keyBits, seed: encoded.seed)
        return decodeKeys(encoded, state: state)
    }

    /// Decode values using seed-based state creation (convenience).
    public static func decodeValues(_ encoded: EncodedValues, seed: Int = 42) -> MLXArray {
        let dim = encoded.shape.last ?? 128
        let state = EncoderState(dim: dim, valueBits: encoded.indexBits, seed: encoded.seed)
        return decodeValues(encoded, state: state)
    }
}
