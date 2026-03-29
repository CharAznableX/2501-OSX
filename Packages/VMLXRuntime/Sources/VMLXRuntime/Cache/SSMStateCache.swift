import Foundation
import MLX
import os
import CryptoKit

/// LRU companion cache for SSM layer state.
/// Used alongside paged KV cache for hybrid models (Nemotron-H, Jamba, Qwen3.5-A3B).
///
/// Critical invariants:
/// - Empty array [] is a MISS, not just nil (bug fix from VMLX ba07392)
/// - Deep-copy on fetch: SSM state is mutable, sharing causes corruption
/// - Stores SSMCheckpoint objects keyed by token hash + boundary
public final class SSMStateCache: @unchecked Sendable {

    private let lock = OSAllocatedUnfairLock()
    private let maxEntries: Int

    // LRU ordered: oldest at index 0, newest at end
    private var entries: [(key: String, checkpoint: SSMCheckpoint)] = []

    // Stats
    public private(set) var hits: Int = 0
    public private(set) var misses: Int = 0
    public private(set) var stores: Int = 0

    public init(maxEntries: Int = 50) {
        self.maxEntries = maxEntries
    }

    /// Store SSM checkpoint. Keyed by tokenHash + boundary.
    public func store(checkpoint: SSMCheckpoint) {
        let key = _makeKey(tokenHash: checkpoint.tokenHash, boundary: checkpoint.boundary)

        lock.withLock {
            // Remove existing if present
            if let idx = entries.firstIndex(where: { $0.key == key }) {
                entries.remove(at: idx)
            }

            entries.append((key: key, checkpoint: checkpoint))
            stores += 1

            // Evict oldest if over limit
            while entries.count > maxEntries {
                entries.removeFirst()
            }
        }
    }

    /// Store SSM states directly (convenience for non-thinking models).
    public func store(ssmStates: [SSMStateLayer], tokens: [Int], boundary: Int) {
        let tokenHash = Self.hashTokens(tokens, count: boundary)
        let checkpoint = SSMCheckpoint(ssmStates: ssmStates, boundary: boundary, tokenHash: tokenHash)
        store(checkpoint: checkpoint)
    }

    /// Fetch SSM checkpoint. Returns deep-copied states.
    /// CRITICAL: empty ssmStates is a MISS, not a hit.
    public func fetch(tokenHash: String, boundary: Int) -> SSMCheckpoint? {
        let key = _makeKey(tokenHash: tokenHash, boundary: boundary)

        return lock.withLock {
            guard let idx = entries.firstIndex(where: { $0.key == key }) else {
                misses += 1
                return nil
            }

            let checkpoint = entries[idx].checkpoint

            // CRITICAL: empty states == MISS (bug fix from VMLX ba07392)
            guard !checkpoint.ssmStates.isEmpty else {
                misses += 1
                return nil
            }

            // Move to end (MRU)
            let entry = entries.remove(at: idx)
            entries.append(entry)

            // Deep copy: SSM state is mutable, sharing causes corruption
            let copied = _deepCopy(checkpoint)
            hits += 1
            return copied
        }
    }

    /// Fetch by tokens directly (convenience).
    public func fetch(tokens: [Int], boundary: Int) -> SSMCheckpoint? {
        let tokenHash = Self.hashTokens(tokens, count: boundary)
        return fetch(tokenHash: tokenHash, boundary: boundary)
    }

    /// Remove checkpoint for given key.
    public func invalidate(tokenHash: String, boundary: Int) {
        let key = _makeKey(tokenHash: tokenHash, boundary: boundary)
        lock.withLock {
            entries.removeAll { $0.key == key }
        }
    }

    public var count: Int {
        lock.withLock { entries.count }
    }

    public func clear() {
        lock.withLock { entries.removeAll() }
    }

    // MARK: - Token Hashing

    /// Hash first `count` tokens for cache key.
    public static func hashTokens(_ tokens: [Int], count: Int) -> String {
        let subset = Array(tokens.prefix(count))
        let json = "[" + subset.map(String.init).joined(separator: ",") + "]"
        let hash = SHA256.hash(data: Data(json.utf8))
        return hash.map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Private

    private func _makeKey(tokenHash: String, boundary: Int) -> String {
        "\(tokenHash):\(boundary)"
    }

    /// Deep copy checkpoint. Each SSM state array is independently copied
    /// to prevent in-place mutation corruption across requests.
    private func _deepCopy(_ checkpoint: SSMCheckpoint) -> SSMCheckpoint {
        let copiedStates = checkpoint.ssmStates.map { layer -> SSMStateLayer in
            // Multiply by 1 to force a copy of the MLXArray
            // (MLXArray is copy-on-write, this triggers the copy)
            let copiedState = layer.state.map { array -> MLXArray in
                array * 1
            }
            return SSMStateLayer(state: copiedState, isCumulative: layer.isCumulative)
        }
        return SSMCheckpoint(
            ssmStates: copiedStates,
            boundary: checkpoint.boundary,
            tokenHash: checkpoint.tokenHash
        )
    }
}
