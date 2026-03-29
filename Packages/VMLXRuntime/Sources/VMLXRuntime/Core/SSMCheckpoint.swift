import Foundation
import MLX

/// SSM state snapshot at a stable boundary (before gen_prompt_len).
/// Safe to store/fetch because key matches truncated KV cache key.
/// Used for thinking model support where post-generation SSM state is contaminated.
public struct SSMCheckpoint: Sendable {
    /// SSM layer states at the checkpoint boundary — one per SSM layer in the model.
    public let ssmStates: [SSMStateLayer]

    /// Token position where this checkpoint was taken.
    public let boundary: Int

    /// SHA-256 hash of tokens[:boundary] for cache key matching.
    public let tokenHash: String

    /// When this checkpoint was created.
    public let timestamp: Date

    public init(ssmStates: [SSMStateLayer], boundary: Int, tokenHash: String) {
        self.ssmStates = ssmStates
        self.boundary = boundary
        self.tokenHash = tokenHash
        self.timestamp = Date()
    }

    /// Total estimated memory across all SSM state arrays.
    public var estimatedBytes: Int {
        ssmStates.reduce(0) { $0 + $1.estimatedBytes }
    }
}
