import Foundation
import MLX
import MLXRandom

/// Quantized Johnson-Lindenstrauss (QJL) projection for TurboQuant key compression.
///
/// QJL is the core of TurboQuant's key encoding. It projects each key vector
/// through a random Gaussian matrix, stores only the signs of the projections
/// (1 bit each) plus the vector norm, and reconstructs an approximation during
/// attention by projecting the signs back through the same matrix.
///
/// The Johnson-Lindenstrauss lemma guarantees that random projections preserve
/// pairwise distances (and therefore attention scores) with high probability,
/// even after sign quantization. The reconstruction quality depends on `dim`:
/// higher dimensions give better preservation.
///
/// Key properties:
/// - Encode: `signs = sign(x @ S^T)`, `norm = ||x||` — O(d^2) per vector
/// - Decode: `x_hat = (sqrt(pi/2) / d) * norm * (signs @ S)` — O(d^2) per vector
/// - The projection matrix `S` is generated deterministically from a seed,
///   so it never needs to be stored — just regenerated from the same seed.
///
/// For TurboQuant, QJL replaces the codebook-based key encoding used for values.
/// Keys need higher fidelity because attention scores are computed as dot products
/// with queries, and QJL preserves these dot products better than scalar quantization.
public struct TQQJL: Sendable {

    // MARK: - Projection Matrix Generation

    /// Generate a random Gaussian projection matrix for QJL.
    ///
    /// The matrix entries are i.i.d. standard normal, generated deterministically
    /// from the given seed. The same seed always produces the same matrix,
    /// so encode and decode in the same process are guaranteed to match.
    ///
    /// - Parameters:
    ///   - dim: Vector dimension. Produces a dim x dim square matrix.
    ///   - seed: Random seed for deterministic generation.
    /// - Returns: MLXArray of shape [dim, dim] with float32 Gaussian entries.
    public static func generateProjection(dim: Int, seed: Int = 0) -> MLXArray {
        let rngKey = MLXRandom.key(UInt64(seed))
        return MLXRandom.normal([dim, dim], key: rngKey)
    }

    // MARK: - Encode

    /// Encode vectors to QJL sign representation.
    ///
    /// Projects each input vector through the random matrix `S` and stores:
    /// 1. The sign of each projection component (+1 or -1)
    /// 2. The L2 norm of the original vector
    ///
    /// Together, these allow approximate reconstruction via `qjlDecode`.
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape `[..., dim]`. Typically `[batch, heads, tokens, head_dim]`.
    ///   - S: Projection matrix of shape `[dim, dim]` from `generateProjection`.
    /// - Returns: Tuple of:
    ///   - `signs`: Float32 tensor of shape `[..., dim]` with values in {-1.0, +1.0}.
    ///   - `norm`: Float32 tensor of shape `[..., 1]` with per-vector L2 norms.
    public static func qjlEncode(_ x: MLXArray, S: MLXArray) -> (signs: MLXArray, norm: MLXArray) {
        // Project: x @ S^T gives [..., dim]
        let projected = matmul(x.asType(.float32), S.transposed())

        // Signs: +1 where projected >= 0, -1 otherwise
        let signs = which(
            projected .>= MLXArray(Float(0.0)),
            MLXArray(Float(1.0)),
            MLXArray(Float(-1.0))
        )

        // Per-vector L2 norm: sqrt(sum(x^2, axis=-1, keepdims=True))
        let xf = x.asType(.float32)
        let norm = (xf * xf).sum(axis: -1, keepDims: true).sqrt()

        return (signs, norm)
    }

    // MARK: - Decode

    /// Decode QJL sign representation back to approximate vectors.
    ///
    /// Reconstruction formula: `x_hat = sqrt(pi/2) / dim * norm * (signs @ S)`
    ///
    /// The `sqrt(pi/2) / dim` scaling factor comes from the expected magnitude
    /// of a sign-quantized random projection: E[|sign(z) * z|] = sqrt(2/pi)
    /// for standard normal z, and we sum `dim` such terms.
    ///
    /// - Parameters:
    ///   - signs: Sign tensor from `qjlEncode`, shape `[..., dim]`.
    ///   - norm: Norm tensor from `qjlEncode`, shape `[..., 1]`.
    ///   - S: The same projection matrix used in `qjlEncode`.
    /// - Returns: Reconstructed float32 tensor of shape `[..., dim]`.
    public static func qjlDecode(signs: MLXArray, norm: MLXArray, S: MLXArray) -> MLXArray {
        let dim = signs.dim(signs.shape.count - 1)
        let scale = MLXArray(Float(sqrt(Float.pi / 2.0)) / Float(dim))

        // Reconstruct: signs @ S
        let reconstructed = matmul(signs.asType(.float32), S)

        return scale * norm.asType(.float32) * reconstructed
    }
}
