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
}
