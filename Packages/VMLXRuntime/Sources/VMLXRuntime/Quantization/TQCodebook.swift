import Foundation
import MLX

// MARK: - Codebook Cache Key

/// Cache key for looking up precomputed codebooks by (dimension, bits).
private struct CodebookKey: Hashable, Sendable {
    let dim: Int
    let bits: Int
}

// MARK: - Codebook Cache

/// Thread-safe cache for precomputed Lloyd-Max codebooks.
/// Codebooks are expensive to compute (200 iterations of Lloyd-Max)
/// but only depend on (dim, bits), so we cache aggressively.
private final class CodebookCache: @unchecked Sendable {
    private var cache: [CodebookKey: [Float]] = [:]
    private let lock = NSLock()

    func get(dim: Int, bits: Int) -> [Float]? {
        lock.lock()
        defer { lock.unlock() }
        return cache[CodebookKey(dim: dim, bits: bits)]
    }

    func set(dim: Int, bits: Int, codebook: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        cache[CodebookKey(dim: dim, bits: bits)] = codebook
    }
}

private let sharedCodebookCache = CodebookCache()

// MARK: - TQCodebook

/// Lloyd-Max optimal scalar quantizer for TurboQuant.
///
/// Computes codebooks that minimize MSE under the marginal distribution of
/// Hadamard-rotated vector components. After rotation, each component follows
/// a distribution proportional to `(1 - x^2)^((d-3)/2)` on [-1, 1] (the
/// projection of a uniform sphere onto one axis), which is well-approximated
/// by a Beta PDF parameterized by dimension `d`.
///
/// The Lloyd-Max algorithm iteratively:
/// 1. Computes decision boundaries as midpoints between centroids
/// 2. Recomputes each centroid as the conditional mean within its Voronoi cell
///
/// Codebooks are cached by (dim, bits) since they are deterministic and
/// reused across all layers/heads with the same configuration.
public struct TQCodebook: Sendable {

    // MARK: - Beta PDF (Marginal Distribution)

    /// Evaluate the Beta-type PDF for the marginal distribution of a single
    /// component of a unit vector uniformly distributed on S^{d-1}.
    ///
    /// `p(x) = C_d * (1 - x^2)^((d-3)/2)` where C_d = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2))
    ///
    /// - Parameters:
    ///   - x: Evaluation point in [-1, 1].
    ///   - d: Ambient dimension (must be >= 2).
    /// - Returns: Unnormalized density value.
    static func betaPDF(_ x: Float, d: Int) -> Float {
        let df = Double(d)
        let logConst = lgamma(df / 2.0) - 0.5 * Darwin.log(Double.pi) - lgamma((df - 1.0) / 2.0)
        let xd = Double(x)
        let safe = Swift.max(1.0 - xd * xd, 1e-30)
        return Float(Darwin.exp(logConst + (df - 3.0) / 2.0 * Darwin.log(safe)))
    }

    // MARK: - Trapezoidal Integration

    /// Trapezoidal rule integration over paired (x, y) arrays.
    /// Equivalent to `numpy.trapezoid(y, x)`.
    ///
    /// - Parameters:
    ///   - y: Function values at sample points.
    ///   - x: Sample points (must be same length as y).
    /// - Returns: Approximate integral.
    static func trapezoid(_ y: [Float], _ x: [Float]) -> Float {
        guard y.count == x.count, y.count >= 2 else { return 0 }
        var result: Float = 0
        for i in 0..<(y.count - 1) {
            result += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.0
        }
        return result
    }

    // MARK: - Lloyd-Max Codebook

    /// Compute a Lloyd-Max optimal scalar codebook for the given dimension and bit width.
    ///
    /// The codebook minimizes E[(X - Q(X))^2] where X follows the Beta marginal
    /// distribution induced by Hadamard rotation in `dim` dimensions, and Q maps
    /// to one of `2^bits` reconstruction levels.
    ///
    /// Results are cached: repeated calls with the same (dim, bits) return instantly.
    ///
    /// - Parameters:
    ///   - dim: Vector dimension (determines the shape of the marginal PDF).
    ///   - bits: Number of index bits. Codebook size = 2^bits (e.g., 3 bits -> 8 levels).
    ///   - iterations: Number of Lloyd-Max iterations (default 200).
    /// - Returns: Sorted array of `2^bits` centroid values.
    public static func computeCodebook(dim: Int, bits: Int, iterations: Int = 200) -> [Float] {
        // Check cache first
        if let cached = sharedCodebookCache.get(dim: dim, bits: bits) {
            return cached
        }

        let nCodes = 1 << bits

        // Build the PDF on a fine grid over [-1, 1]
        let nGrid = 10000
        var grid = [Float](repeating: 0, count: nGrid)
        for i in 0..<nGrid {
            grid[i] = -1.0 + 2.0 * Float(i) / Float(nGrid - 1)
        }

        var pdf = grid.map { betaPDF($0, d: dim) }

        // Normalize so integral = 1
        let totalMass = trapezoid(pdf, grid)
        if totalMass > 0 {
            for i in 0..<pdf.count {
                pdf[i] /= totalMass
            }
        }

        // Initialize centroids uniformly within the effective support
        // Support width ~ 3/sqrt(dim), capturing the bulk of the distribution
        let support = 3.0 / sqrt(max(Float(dim), 1.0))
        var centroids = [Float](repeating: 0, count: nCodes)
        for i in 0..<nCodes {
            centroids[i] = -support + 2.0 * support * Float(i) / Float(max(nCodes - 1, 1))
        }

        // Lloyd-Max iteration
        for _ in 0..<iterations {
            // Compute decision boundaries (midpoints between adjacent centroids)
            var boundaries = [Float](repeating: 0, count: nCodes + 1)
            boundaries[0] = -1.0
            boundaries[nCodes] = 1.0
            for i in 0..<(nCodes - 1) {
                boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0
            }

            // Recompute centroids as conditional means within each Voronoi cell
            for i in 0..<nCodes {
                let lo = boundaries[i]
                let hi = boundaries[i + 1]

                // Gather grid points within [lo, hi)
                var maskedX = [Float]()
                var maskedPDF = [Float]()
                for j in 0..<nGrid {
                    if grid[j] >= lo && grid[j] < hi {
                        maskedX.append(grid[j])
                        maskedPDF.append(pdf[j])
                    }
                }

                guard maskedX.count >= 2 else { continue }

                let mass = trapezoid(maskedPDF, maskedX)
                let moment = trapezoid(
                    zip(maskedX, maskedPDF).map { $0.0 * $0.1 },
                    maskedX
                )
                centroids[i] = moment / max(mass, 1e-10)
            }
        }

        let result = centroids.sorted()
        sharedCodebookCache.set(dim: dim, bits: bits, codebook: result)
        return result
    }

    // MARK: - Scalar Quantization (MLXArray)

    /// Quantize an MLXArray of floats to codebook indices using boundary comparisons.
    ///
    /// For each element in `x`, finds the codebook bin by counting how many
    /// decision boundaries (midpoints between adjacent centroids) the value exceeds.
    /// This is equivalent to `numpy.searchsorted(boundaries, x) - 1` but expressed
    /// as a sum of comparisons for GPU-friendly execution.
    ///
    /// - Parameters:
    ///   - x: Input tensor of float values (any shape).
    ///   - codebook: Sorted array of centroid values from `computeCodebook`.
    /// - Returns: Tensor of uint8 indices (same shape as x), each in [0, codebook.count).
    public static func quantizeScalar(_ x: MLXArray, codebook: [Float]) -> MLXArray {
        // Compute midpoints between adjacent centroids (decision boundaries)
        var boundaries = [Float]()
        for i in 0..<(codebook.count - 1) {
            boundaries.append((codebook[i] + codebook[i + 1]) / 2.0)
        }

        // Index = number of boundaries that x exceeds
        // Start with zeros
        var indices = MLXArray.zeros(like: x).asType(.uint8)
        for b in boundaries {
            let threshold = MLXArray(b)
            indices = indices + (x .> threshold).asType(.uint8)
        }
        return indices
    }

    /// Dequantize codebook indices back to float values.
    ///
    /// Simple table lookup: maps each uint8 index to the corresponding centroid value.
    ///
    /// - Parameters:
    ///   - indices: Tensor of uint8 codebook indices.
    ///   - codebook: Sorted array of centroid values from `computeCodebook`.
    /// - Returns: Tensor of float32 values (same shape as indices).
    public static func dequantizeScalar(_ indices: MLXArray, codebook: [Float]) -> MLXArray {
        let codebookArray = MLXArray(codebook)
        return take(codebookArray, indices.asType(.int32))
    }
}
