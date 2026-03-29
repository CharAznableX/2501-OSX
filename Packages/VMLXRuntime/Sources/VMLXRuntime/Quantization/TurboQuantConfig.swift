import Foundation

/// Configuration for TurboQuant KV cache compression.
/// Controls per-layer bit widths with critical layer overrides.
/// Hybrid-aware: returns nil for SSM layers (no KV to compress).
public struct TurboQuantConfig: Sendable {

    /// Default bit width for key compression (standard layers).
    public var defaultKeyBits: Int

    /// Default bit width for value compression (standard layers).
    public var defaultValueBits: Int

    /// Layer indices that use higher precision. Negative indices count from end.
    /// Default: first 3 and last 3 layers [0, 1, 2, -3, -2, -1].
    public var criticalLayers: [Int]

    /// Bit width for key compression on critical layers.
    public var criticalKeyBits: Int

    /// Bit width for value compression on critical layers.
    public var criticalValueBits: Int

    /// Random seed for codebook generation (reproducibility).
    public var seed: Int

    /// Layer pattern for hybrid models. nil = all attention (pure transformer).
    /// When set, SSM layers are skipped entirely during TQ compression.
    public var layerPattern: [LayerType]?

    /// For MLA models (DeepSeek): custom key dimension = qk_nope_head_dim + qk_rope_head_dim.
    /// nil = use standard head_dim.
    public var mlaKeyDim: Int?

    /// For MLA models: custom value dimension = v_head_dim.
    /// nil = use standard head_dim.
    public var mlaValueDim: Int?

    public init(
        defaultKeyBits: Int = 3,
        defaultValueBits: Int = 3,
        criticalLayers: [Int] = [0, 1, 2, -3, -2, -1],
        criticalKeyBits: Int = 4,
        criticalValueBits: Int = 4,
        seed: Int = 42,
        layerPattern: [LayerType]? = nil,
        mlaKeyDim: Int? = nil,
        mlaValueDim: Int? = nil
    ) {
        self.defaultKeyBits = defaultKeyBits
        self.defaultValueBits = defaultValueBits
        self.criticalLayers = criticalLayers
        self.criticalKeyBits = criticalKeyBits
        self.criticalValueBits = criticalValueBits
        self.seed = seed
        self.layerPattern = layerPattern
        self.mlaKeyDim = mlaKeyDim
        self.mlaValueDim = mlaValueDim
    }

    /// Returns the key bit width for a given layer, or nil if this layer has no KV (SSM layer).
    /// Critical layers get higher precision. SSM layers are skipped entirely.
    public func keyBits(forLayer i: Int, totalLayers: Int) -> Int? {
        // Check if this is an SSM layer (no KV to compress)
        if let pattern = layerPattern, i < pattern.count {
            if pattern[i] == .ssm { return nil }
            if pattern[i] == .expert { return nil }  // MoE expert layers may not have standard KV
        }

        // Resolve negative indices
        let resolvedCritical = criticalLayers.map { idx -> Int in
            idx < 0 ? totalLayers + idx : idx
        }

        let resolvedI = i < 0 ? totalLayers + i : i

        if resolvedCritical.contains(resolvedI) {
            return criticalKeyBits
        }
        return defaultKeyBits
    }

    /// Returns the value bit width for a given layer, or nil for SSM layers.
    public func valueBits(forLayer i: Int, totalLayers: Int) -> Int? {
        // Same SSM check as keyBits
        if let pattern = layerPattern, i < pattern.count {
            if pattern[i] == .ssm { return nil }
            if pattern[i] == .expert { return nil }
        }

        let resolvedCritical = criticalLayers.map { idx -> Int in
            idx < 0 ? totalLayers + idx : idx
        }

        let resolvedI = i < 0 ? totalLayers + i : i

        if resolvedCritical.contains(resolvedI) {
            return criticalValueBits
        }
        return defaultValueBits
    }

    /// Returns the key dimension for this model.
    /// Uses MLA custom dim if set, otherwise returns nil (caller uses standard head_dim).
    public var keyDim: Int? { mlaKeyDim }

    /// Returns the value dimension for this model.
    public var valueDim: Int? { mlaValueDim }

    /// Whether this config has MLA-specific dimensions.
    public var isMLA: Bool { mlaKeyDim != nil || mlaValueDim != nil }

    /// Number of attention layers (non-SSM) in the pattern.
    /// Returns totalLayers if no pattern set (all attention).
    public func attentionLayerCount(totalLayers: Int) -> Int {
        guard let pattern = layerPattern else { return totalLayers }
        return pattern.filter { $0 == .attention }.count
    }

    /// Indices of attention layers in the pattern.
    public func attentionLayerIndices(totalLayers: Int) -> [Int] {
        guard let pattern = layerPattern else {
            return Array(0..<totalLayers)
        }
        return pattern.enumerated().compactMap { i, t in
            t == .attention ? i : nil
        }
    }
}
