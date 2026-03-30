import Foundation
import MLX
import CryptoKit
import os

/// Status of a re-derivation task.
public enum ReDeriverStatus: Sendable {
    case idle
    case inProgress(tokenHash: String)
    case completed(SSMCheckpoint)
    case failed(Error)
}

/// Actor that manages async re-derivation of SSM state.
///
/// When SSM checkpoint has been evicted but KV blocks exist:
/// 1. Run full forward pass on cached tokens (all layers — SSM can't run independently)
/// 2. Checkpoint SSM at stable boundary
/// 3. Store checkpoint for future use
/// 4. As side effect: refresh attention KV cache
///
/// Decision logic:
/// - Tokens < syncThreshold (default 512): sync re-derive (wait for result)
/// - Tokens >= syncThreshold: async re-derive (full prefill now, re-derive in background)
///
/// Deduplicates concurrent requests for the same token hash.
///
/// ## Model Forward Pass Integration
///
/// The re-deriver accepts an optional `ModelForwardPass` reference. When a model is
/// available, `requestReDerive()` runs prefill on tokens up to `stableBoundary` to:
/// - Refresh attention KV cache state (side effect of prefill)
/// - Record the boundary in the checkpoint for cache keying
///
/// For pure-attention models (no Mamba layers), the SSM states array remains empty
/// because there are no SSM layers to extract state from. The checkpoint still records
/// the boundary position, which is used for cache key matching.
///
/// When Mamba layer support is added to `TransformerModel`, the re-deriver will also
/// extract SSM hidden states at the stable boundary and populate `ssmStates`.
public actor SSMReDeriver {

    /// Threshold: below this token count, re-derive synchronously (worth waiting).
    public let syncThreshold: Int

    /// In-progress re-derivation tasks, keyed by token hash.
    private var activeTasks: [String: Task<SSMCheckpoint, Error>] = [:]

    /// Completed checkpoints waiting to be consumed.
    private var completedCheckpoints: [String: SSMCheckpoint] = [:]

    /// SSM state cache to store re-derived checkpoints.
    private let ssmCache: SSMStateCache

    /// Optional model forward pass for running actual prefill during re-derivation.
    /// When nil, the re-deriver creates boundary-only checkpoints (no SSM state, no KV refresh).
    private var model: (any ModelForwardPass)?

    /// Stats
    public private(set) var syncReDerives: Int = 0
    public private(set) var asyncReDerives: Int = 0
    public private(set) var deduplicatedRequests: Int = 0

    public init(ssmCache: SSMStateCache, syncThreshold: Int = 512, model: (any ModelForwardPass)? = nil) {
        self.ssmCache = ssmCache
        self.syncThreshold = syncThreshold
        self.model = model
    }

    /// Update the model forward pass reference (e.g., after model load/unload).
    public func setModel(_ model: (any ModelForwardPass)?) {
        self.model = model
    }

    // MARK: - Decision Logic

    /// Decide whether to re-derive sync or async based on token count.
    public func shouldSyncReDerive(tokenCount: Int) -> Bool {
        tokenCount < syncThreshold
    }

    // MARK: - Re-Derivation

    /// Request SSM state re-derivation for a token sequence.
    ///
    /// If a `ModelForwardPass` is available (either set via `init` or `setModel(_:)`),
    /// prefill is run on `tokens[0..<stableBoundary]` to:
    /// - Refresh attention KV cache (side effect of running the forward pass)
    /// - Record the stable boundary in the checkpoint for cache keying
    ///
    /// SSM state extraction requires Mamba layer support in `TransformerModel`.
    /// For pure-attention models, `ssmStates` remains empty — the checkpoint still
    /// records the boundary position for correct cache key matching.
    ///
    /// If sync: waits and returns the checkpoint.
    /// If async: starts background task and returns nil (checkpoint stored when done).
    ///
    /// - Parameters:
    ///   - tokens: Full token sequence for the conversation.
    ///   - stableBoundary: Token index up to which state should be checkpointed.
    ///   - forceSync: If true, always wait for the result regardless of token count.
    ///   - model: Optional override for the model forward pass. If nil, uses the
    ///     instance-level model set via `init` or `setModel(_:)`.
    public func requestReDerive(
        tokens: [Int],
        stableBoundary: Int,
        forceSync: Bool = false,
        model override: (any ModelForwardPass)? = nil
    ) async throws -> SSMCheckpoint? {
        let tokenHash = SSMStateCache.hashTokens(tokens, count: stableBoundary)

        // Check if already completed
        if let existing = completedCheckpoints[tokenHash] {
            return existing
        }

        // Check if already in progress (deduplicate)
        if let existingTask = activeTasks[tokenHash] {
            if forceSync || shouldSyncReDerive(tokenCount: tokens.count) {
                deduplicatedRequests += 1
                return try await existingTask.value
            }
            deduplicatedRequests += 1
            return nil  // Async — will complete later
        }

        // Resolve which model to use: explicit override > instance model > nil
        let activeModel = override ?? self.model

        // Start new re-derivation
        let prefillTokens = Array(tokens.prefix(stableBoundary))
        let task = Task<SSMCheckpoint, Error> {
            // If we have a model, run actual prefill to refresh KV cache
            if let fwdPass = activeModel, !prefillTokens.isEmpty {
                let inputIds = MLXArray(prefillTokens.map { Int32($0) })
                var cacheArrays: [MLXArray] = []

                // Build causal mask for the prefill sequence
                let mask: MLXArray?
                if prefillTokens.count > 1 {
                    mask = TransformerModel.createCausalMask(
                        seqLen: prefillTokens.count,
                        offset: 0,
                        dtype: .float16
                    )
                } else {
                    mask = nil
                }

                // Run prefill — this refreshes the model's internal KV cache
                // as a side effect. For hybrid models with Mamba layers, this
                // would also recompute SSM hidden states.
                _ = try await fwdPass.prefill(
                    inputIds: inputIds,
                    cache: &cacheArrays,
                    mask: mask
                )
            }

            // Create checkpoint with boundary info.
            // SSM states are empty for pure-attention models — when Mamba layer
            // support is added to TransformerModel, extract SSM hidden states
            // here via a new protocol method (e.g., `extractSSMStates(atLayer:)`).
            let checkpoint = SSMCheckpoint(
                ssmStates: [],  // Empty for attention-only models; populated when Mamba layers are supported
                boundary: stableBoundary,
                tokenHash: tokenHash
            )

            return checkpoint
        }

        activeTasks[tokenHash] = task

        if forceSync || shouldSyncReDerive(tokenCount: tokens.count) {
            syncReDerives += 1
            let checkpoint = try await task.value
            activeTasks.removeValue(forKey: tokenHash)
            completedCheckpoints[tokenHash] = checkpoint
            ssmCache.store(checkpoint: checkpoint)
            return checkpoint
        } else {
            asyncReDerives += 1
            // Fire and forget — store when complete
            Task {
                do {
                    let checkpoint = try await task.value
                    self.activeTasks.removeValue(forKey: tokenHash)
                    self.completedCheckpoints[tokenHash] = checkpoint
                    self.ssmCache.store(checkpoint: checkpoint)
                } catch {
                    self.activeTasks.removeValue(forKey: tokenHash)
                }
            }
            return nil  // Async — caller should proceed with full prefill
        }
    }

    // MARK: - Queries

    /// Check if a re-derivation is in progress for this token hash.
    public func isReDeriving(tokenHash: String) -> Bool {
        activeTasks[tokenHash] != nil
    }

    /// Check if a completed checkpoint exists.
    public func hasCheckpoint(tokenHash: String) -> Bool {
        completedCheckpoints[tokenHash] != nil
    }

    /// Get a completed checkpoint (and remove from pending).
    public func consumeCheckpoint(tokenHash: String) -> SSMCheckpoint? {
        completedCheckpoints.removeValue(forKey: tokenHash)
    }

    /// Number of active re-derivation tasks.
    public var activeTaskCount: Int {
        activeTasks.count
    }

    /// Clear all completed checkpoints.
    public func clearCompleted() {
        completedCheckpoints.removeAll()
    }

    /// Cancel all active re-derivation tasks.
    public func cancelAll() {
        for (_, task) in activeTasks {
            task.cancel()
        }
        activeTasks.removeAll()
    }
}
