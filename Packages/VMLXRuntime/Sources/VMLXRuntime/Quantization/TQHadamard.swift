import Foundation
import MLX

/// Randomized Hadamard rotation for TurboQuant.
///
/// The Hadamard transform spreads information across all dimensions of a vector,
/// making each component's marginal distribution approximately Beta-shaped.
/// This is critical for scalar quantization: without rotation, individual
/// dimensions can have very different scales, making a single codebook suboptimal.
///
/// The transform supports non-power-of-2 dimensions by decomposing into
/// power-of-2 blocks (e.g., dim=96 -> blocks [64, 32]) and applying the
/// Hadamard butterfly independently to each block.
///
/// A random diagonal sign matrix (D) is applied before the transform:
///   `y = H * D * x`
/// This breaks any structure in the input that could align with Hadamard
/// basis vectors and cause poor quantization on specific components.
///
/// The inverse is: `x = D * H * y` (since H is symmetric and orthogonal,
/// and D is self-inverse).
public struct TQHadamard: Sendable {

    // MARK: - Dimension Decomposition

    /// Decompose a dimension into a sum of descending powers of 2.
    ///
    /// Example: 96 -> [64, 32], 128 -> [128], 160 -> [128, 32]
    ///
    /// This allows applying the Hadamard butterfly (which requires power-of-2 size)
    /// to each block independently, then concatenating the results.
    ///
    /// - Parameter dim: Dimension to decompose (must be > 0).
    /// - Returns: Array of power-of-2 block sizes summing to `dim`.
    public static func decomposePow2Blocks(_ dim: Int) -> [Int] {
        var blocks = [Int]()
        var remaining = dim
        while remaining > 0 {
            // Largest power of 2 <= remaining
            let bitLen = Int.bitWidth - remaining.leadingZeroBitCount
            let p = 1 << (bitLen - 1)
            blocks.append(p)
            remaining -= p
        }
        return blocks
    }

    // MARK: - Random Sign Vector

    /// Generate a deterministic random sign vector (+1 / -1) of the given dimension.
    ///
    /// Uses `srand48`/`drand48` for CPU-side determinism that does not depend on
    /// MLX's GPU random state. The same seed always produces the same signs,
    /// which is required for correct inverse transforms.
    ///
    /// - Parameters:
    ///   - dim: Length of the sign vector.
    ///   - seed: Random seed for reproducibility.
    /// - Returns: MLXArray of shape [dim] with float32 values in {-1.0, +1.0}.
    public static func generateRandomSigns(dim: Int, seed: Int = 0) -> MLXArray {
        srand48(seed)
        var signs = [Float](repeating: 0, count: dim)
        for i in 0..<dim {
            signs[i] = drand48() < 0.5 ? -1.0 : 1.0
        }
        return MLXArray(signs)
    }

    // MARK: - Hadamard Butterfly (Power-of-2)

    /// Apply the (unnormalized) Hadamard butterfly transform to the last dimension.
    ///
    /// The last dimension must be a power of 2. Uses the in-place butterfly
    /// factorization: at each stage h, pairs of sub-vectors of size h are
    /// combined as (a+b, a-b), doubling h until it equals the full dimension.
    ///
    /// The result is scaled by `1/sqrt(d)` to make the transform orthogonal.
    ///
    /// - Parameter x: Input tensor whose last dimension is a power of 2.
    /// - Returns: Hadamard-transformed tensor (same shape as input).
    static func hadamardTransform(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let d = shape[shape.count - 1]

        // Flatten all batch dims into one: [N, d]
        let n = shape.dropLast().reduce(1, *)
        var result = x.reshaped([n, d])

        var h = 1
        while h < d {
            let groups = d / (2 * h)
            // Reshape to [N, groups, 2, h]
            result = result.reshaped([n, groups, 2, h])

            // Butterfly: a = result[:, :, 0, :], b = result[:, :, 1, :]
            let a = result[0..., 0..., 0..<1, 0...].reshaped([n, groups, h])
            let b = result[0..., 0..., 1..<2, 0...].reshaped([n, groups, h])

            // [a+b, a-b] along last axis, reshape back to [N, d]
            result = concatenated([a + b, a - b], axis: -1).reshaped([n, d])
            h *= 2
        }

        // Normalize: H / sqrt(d)
        let scale = MLXArray(Float(1.0 / sqrt(Float(d))))
        return (result * scale).reshaped(shape)
    }

    // MARK: - Forward Rotation

    /// Apply randomized Hadamard rotation: `y = H(D * x)` where D = diag(signs).
    ///
    /// For non-power-of-2 dimensions, the last axis is split into power-of-2 blocks
    /// and each block is transformed independently.
    ///
    /// - Parameters:
    ///   - x: Input tensor. The last dimension must equal `signs.dim(0)`.
    ///   - signs: Random sign vector from `generateRandomSigns`, shape [dim].
    /// - Returns: Rotated tensor (same shape as input).
    public static func hadamardRotate(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = x.dim(x.shape.count - 1)
        let blocks = decomposePow2Blocks(dim)

        if blocks.count == 1 {
            // Single power-of-2 block: apply directly
            return hadamardTransform(x * signs)
        }

        // Multiple blocks: split, transform each, concatenate
        var parts = [MLXArray]()
        var offset = 0
        for bs in blocks {
            let xSlice = x[.ellipsis, offset..<(offset + bs)]
            let sSlice = signs[offset..<(offset + bs)]
            parts.append(hadamardTransform(xSlice * sSlice))
            offset += bs
        }
        return concatenated(parts, axis: -1)
    }

    // MARK: - Inverse Rotation

    /// Apply inverse randomized Hadamard rotation: `x = D * H(y)`.
    ///
    /// Since H is symmetric and orthogonal (H^T = H, H*H = I after scaling),
    /// and D is a diagonal sign matrix (D^{-1} = D), the inverse is simply
    /// applying H first, then multiplying by the same signs.
    ///
    /// - Parameters:
    ///   - y: Rotated tensor.
    ///   - signs: The same sign vector used in `hadamardRotate`.
    /// - Returns: Original (unrotated) tensor (same shape as y).
    public static func hadamardInverse(_ y: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = y.dim(y.shape.count - 1)
        let blocks = decomposePow2Blocks(dim)

        if blocks.count == 1 {
            return hadamardTransform(y) * signs
        }

        var parts = [MLXArray]()
        var offset = 0
        for bs in blocks {
            let ySlice = y[.ellipsis, offset..<(offset + bs)]
            let sSlice = signs[offset..<(offset + bs)]
            parts.append(hadamardTransform(ySlice) * sSlice)
            offset += bs
        }
        return concatenated(parts, axis: -1)
    }
}
