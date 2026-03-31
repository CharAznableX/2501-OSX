import Foundation
import MLX

/// Bit-level packing and unpacking for TurboQuant indices and signs.
///
/// Packs multiple low-bit values (e.g., 3-bit codebook indices) into uint32
/// words to minimize memory footprint. For 3-bit indices, each uint32 holds
/// 10 values (30 bits used, 2 wasted). For 4-bit, 8 values per uint32.
///
/// Sign bits are packed 32 per uint32 (one bit each), encoding the QJL
/// projection sign for residual correction during key decoding.
///
/// All operations use MLXArray for GPU-friendly execution. Packing and
/// unpacking are element-wise bitshift/mask operations with no data-dependent
/// branching, making them efficient on Metal.
public struct TQBitPack: Sendable {

    // MARK: - Index Packing

    /// Pack low-bit index values into uint32 words.
    ///
    /// Each uint32 stores `32 / bits` values, with value `i` occupying
    /// bits `[i*bits, (i+1)*bits)`. Values are packed in little-endian
    /// bit order (value 0 in the lowest bits).
    ///
    /// The input is flattened and zero-padded to a multiple of `valsPerU32`
    /// before packing.
    ///
    /// - Parameters:
    ///   - values: Index tensor (any shape). Values must fit in `bits` bits.
    ///   - bits: Bits per value (e.g., 3 for 8-level codebook). Must divide 32 evenly
    ///     or the last partial slot is wasted.
    /// - Returns: Packed uint32 tensor, shape `[ceil(N / valsPerU32)]`.
    public static func packBits(_ values: MLXArray, bits: Int) -> MLXArray {
        let valsPerU32 = 32 / bits
        var flat = values.reshaped([-1]).asType(.uint32)
        let count = flat.dim(0)

        // Pad to multiple of valsPerU32
        let pad = (valsPerU32 - (count % valsPerU32)) % valsPerU32
        if pad > 0 {
            let padding = MLXArray.zeros([pad]).asType(.uint32)
            flat = concatenated([flat, padding], axis: 0)
        }

        // Reshape to [numWords, valsPerU32]
        let numWords = flat.dim(0) / valsPerU32
        flat = flat.reshaped([numWords, valsPerU32])

        // Pack: OR each column shifted to its bit position
        var packed = MLXArray.zeros([numWords]).asType(.uint32)
        for i in 0..<valsPerU32 {
            let column = flat[0..., i]
            let shift = MLXArray(UInt32(i * bits))
            packed = packed | (column << shift)
        }
        return packed
    }

    // MARK: - Index Unpacking

    /// Unpack uint32 words back to individual low-bit index values.
    ///
    /// Reverses `packBits`: extracts each value by shifting right and masking.
    /// The result is truncated to `nElements` to remove any padding added
    /// during packing.
    ///
    /// - Parameters:
    ///   - packed: Packed uint32 tensor from `packBits`.
    ///   - bits: Bits per value (must match the value used in `packBits`).
    ///   - nElements: Original number of values before packing (for truncation).
    /// - Returns: Uint8 tensor of shape `[nElements]` with unpacked index values.
    public static func unpackBits(_ packed: MLXArray, bits: Int, nElements: Int) -> MLXArray {
        let valsPerU32 = 32 / bits
        let mask = MLXArray(UInt32((1 << bits) - 1))

        // Extract each slot into a separate array
        var columns = [MLXArray]()
        for i in 0..<valsPerU32 {
            let shift = MLXArray(UInt32(i * bits))
            let extracted = ((packed >> shift) & mask).asType(.uint8)
            columns.append(extracted)
        }

        // Stack columns and flatten: [numWords, valsPerU32] -> [numWords * valsPerU32]
        let stacked = MLX.stacked(columns, axis: -1).reshaped([-1])

        // Truncate to original element count
        return stacked[..<nElements]
    }

    // MARK: - Sign Packing

    /// Pack float sign values (+1/-1) into uint32 bitmasks.
    ///
    /// Each uint32 stores 32 sign bits. The input signs are mapped:
    /// `+1.0 -> bit 1`, `-1.0 -> bit 0` via `(sign + 1) / 2`.
    ///
    /// - Parameter signs: Float tensor with values in {-1.0, +1.0} (any shape).
    /// - Returns: Packed uint32 tensor, shape `[ceil(N / 32)]`.
    public static func packSigns(_ signs: MLXArray) -> MLXArray {
        // Map {-1, +1} -> {0, 1}: bits = (signs + 1) / 2
        var bits = ((signs.reshaped([-1]) + MLXArray(Float(1.0))) / MLXArray(Float(2.0)))
            .asType(.uint32)
        let count = bits.dim(0)

        // Pad to multiple of 32
        let pad = (32 - (count % 32)) % 32
        if pad > 0 {
            let padding = MLXArray.zeros([pad]).asType(.uint32)
            bits = concatenated([bits, padding], axis: 0)
        }

        // Reshape to [numWords, 32]
        let numWords = bits.dim(0) / 32
        bits = bits.reshaped([numWords, 32])

        // Pack: OR each bit shifted to its position
        var packed = MLXArray.zeros([numWords]).asType(.uint32)
        for i in 0..<32 {
            let column = bits[0..., i]
            let shift = MLXArray(UInt32(i))
            packed = packed | (column << shift)
        }
        return packed
    }

    // MARK: - Sign Unpacking

    /// Unpack uint32 bitmasks back to float sign values (+1/-1).
    ///
    /// Reverses `packSigns`: extracts each bit, maps `{0, 1} -> {-1.0, +1.0}`
    /// via `bit * 2 - 1`.
    ///
    /// - Parameters:
    ///   - packed: Packed uint32 tensor from `packSigns`.
    ///   - nElements: Original number of sign values (for truncation).
    /// - Returns: Float32 tensor of shape `[nElements]` with values in {-1.0, +1.0}.
    public static func unpackSigns(_ packed: MLXArray, nElements: Int) -> MLXArray {
        var columns = [MLXArray]()
        for i in 0..<32 {
            let shift = MLXArray(UInt32(i))
            let extracted = ((packed >> shift) & MLXArray(UInt32(1))).asType(.float32)
            columns.append(extracted)
        }

        // Stack and flatten
        let flat = MLX.stacked(columns, axis: -1).reshaped([-1])

        // Truncate and map {0, 1} -> {-1, +1}
        let truncated = flat[..<nElements]
        return truncated * MLXArray(Float(2.0)) - MLXArray(Float(1.0))
    }
}
