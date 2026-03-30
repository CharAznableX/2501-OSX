import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - TransformerConfig

/// Configuration for a decoder-only transformer (Llama / Qwen / Mistral style).
public struct TransformerConfig: Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let numKVHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let headDim: Int

    public init(
        hiddenSize: Int,
        numLayers: Int,
        numAttentionHeads: Int,
        numKVHeads: Int,
        intermediateSize: Int,
        vocabSize: Int,
        rmsNormEps: Float,
        ropeTheta: Float,
        maxPositionEmbeddings: Int,
        headDim: Int
    ) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKVHeads = numKVHeads
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.headDim = headDim
    }

    /// Parse from a model's config.json dictionary.
    /// Checks top-level keys first, then falls back to `text_config` (for VL models).
    public static func from(config: [String: Any]) -> TransformerConfig {
        let tc = config["text_config"] as? [String: Any]

        func get<T>(_ key: String, default defaultVal: T) -> T {
            (config[key] as? T) ?? (tc?[key] as? T) ?? defaultVal
        }

        let hiddenSize: Int = get("hidden_size", default: 4096)
        let numHeads: Int = get("num_attention_heads", default: 32)

        return TransformerConfig(
            hiddenSize: hiddenSize,
            numLayers: get("num_hidden_layers", default: 32),
            numAttentionHeads: numHeads,
            numKVHeads: get("num_key_value_heads", default: numHeads),
            intermediateSize: get("intermediate_size", default: 11008),
            vocabSize: get("vocab_size", default: 32000),
            rmsNormEps: get("rms_norm_eps", default: 1e-6),
            ropeTheta: get("rope_theta", default: 10000.0),
            maxPositionEmbeddings: get("max_position_embeddings", default: 8192),
            headDim: get("head_dim", default: hiddenSize / numHeads)
        )
    }
}

// MARK: - KV Cache

/// Per-layer KV cache for autoregressive decoding.
/// Stores keys and values that grow along the sequence dimension.
public final class KVCache {
    public var keys: MLXArray?
    public var values: MLXArray?

    /// Current sequence length stored in this cache.
    public var sequenceLength: Int {
        keys?.dim(2) ?? 0
    }

    public init() {}

    /// Append new key/value tensors along the sequence dimension.
    /// - Parameters:
    ///   - newKeys: shape [B, numKVHeads, newSeqLen, headDim]
    ///   - newValues: shape [B, numKVHeads, newSeqLen, headDim]
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            let updatedKeys = concatenated([existingKeys, newKeys], axis: 2)
            let updatedValues = concatenated([existingValues, newValues], axis: 2)
            keys = updatedKeys
            values = updatedValues
            return (updatedKeys, updatedValues)
        } else {
            keys = newKeys
            values = newValues
            return (newKeys, newValues)
        }
    }
}

// MARK: - TransformerAttention

/// Multi-head attention with RoPE, GQA, and KV caching.
///
/// Weight key mapping:
/// - `self_attn.q_proj` -> Q projection
/// - `self_attn.k_proj` -> K projection
/// - `self_attn.v_proj` -> V projection
/// - `self_attn.o_proj` -> output projection
public class TransformerAttention: Module {

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public init(_ config: TransformerConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(config.headDim))

        self.rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta
        )

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
    }

    /// Forward pass through attention.
    ///
    /// - Parameters:
    ///   - x: hidden states, shape [B, seqLen, hiddenSize]
    ///   - mask: optional additive attention mask, broadcastable to [B, numHeads, seqLen, totalSeqLen]
    ///   - cache: per-layer KV cache for autoregressive generation
    ///   - offset: position offset for RoPE (number of previously cached tokens)
    /// - Returns: output hidden states, shape [B, seqLen, hiddenSize]
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil,
        offset: Int = 0
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let seqLen = x.dim(1)

        // Project Q, K, V
        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape to [B, seqLen, numHeads, headDim] then transpose to [B, numHeads, seqLen, headDim]
        queries = queries.reshaped(batchSize, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(batchSize, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, seqLen, numKVHeads, headDim).transposed(0, 2, 1, 3)

        // Apply RoPE to Q and K
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Update KV cache (concatenate with previously cached keys/values)
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Scaled dot-product attention (handles GQA automatically)
        // Q: [B, numHeads, seqLen, headDim]
        // K: [B, numKVHeads, totalSeqLen, headDim]
        // V: [B, numKVHeads, totalSeqLen, headDim]
        let output: MLXArray
        if let mask {
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask
            )
        } else {
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: nil
            )
        }

        // output: [B, numHeads, seqLen, headDim] -> [B, seqLen, numHeads * headDim]
        let reshaped = output.transposed(0, 2, 1, 3).reshaped(batchSize, seqLen, numHeads * headDim)

        return oProj(reshaped)
    }
}

// MARK: - TransformerFFN

/// SwiGLU feed-forward network (gate/up/down projections).
///
/// Weight key mapping:
/// - `mlp.gate_proj` -> gate projection
/// - `mlp.up_proj` -> up projection
/// - `mlp.down_proj` -> down projection
///
/// Computes: `down_proj(silu(gate_proj(x)) * up_proj(x))`
public class TransformerFFN: Module {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(_ config: TransformerConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - TransformerBlock

/// A single transformer decoder block: pre-norm attention + FFN with residuals.
///
/// Weight key mapping:
/// - `self_attn.*` -> attention sub-module
/// - `mlp.*` -> FFN sub-module
/// - `input_layernorm.weight` -> pre-attention norm
/// - `post_attention_layernorm.weight` -> pre-FFN norm
public class TransformerBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: TransformerAttention
    @ModuleInfo(key: "mlp") var ffn: TransformerFFN
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ config: TransformerConfig) {
        self._attention.wrappedValue = TransformerAttention(config)
        self._ffn.wrappedValue = TransformerFFN(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Forward pass through the block.
    ///
    /// - Parameters:
    ///   - x: hidden states [B, seqLen, hiddenSize]
    ///   - mask: optional attention mask
    ///   - cache: KV cache for this layer
    ///   - offset: position offset for RoPE
    /// - Returns: updated hidden states [B, seqLen, hiddenSize]
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil,
        offset: Int = 0
    ) -> MLXArray {
        // Pre-norm attention with residual
        var h = x + attention(inputLayerNorm(x), mask: mask, cache: cache, offset: offset)
        // Pre-norm FFN with residual
        h = h + ffn(postAttentionLayerNorm(h))
        return h
    }
}

// MARK: - TransformerModelInner

/// The inner `model` wrapper that maps to `model.embed_tokens`, `model.layers`, `model.norm`.
/// This intermediate class ensures the key path hierarchy matches HF weight naming:
/// `model.embed_tokens.weight`, `model.layers.0.self_attn.q_proj.weight`, etc.
public class TransformerModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [TransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(_ config: TransformerConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numLayers).map { _ in TransformerBlock(config) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Forward pass: embed -> N layers -> final norm.
    ///
    /// - Parameters:
    ///   - inputIds: token IDs [B, seqLen]
    ///   - mask: optional attention mask
    ///   - kvCaches: per-layer KV caches (one per layer)
    /// - Returns: hidden states [B, seqLen, hiddenSize]
    public func callAsFunction(
        _ inputIds: MLXArray,
        mask: MLXArray? = nil,
        kvCaches: [KVCache]? = nil
    ) -> MLXArray {
        // Position offset from existing cache
        let offset = kvCaches?.first?.sequenceLength ?? 0

        var h = embedTokens(inputIds)

        for (i, layer) in layers.enumerated() {
            let cache = kvCaches?[i]
            h = layer(h, mask: mask, cache: cache, offset: offset)
        }

        return norm(h)
    }
}

// MARK: - TransformerModel

/// Complete decoder-only transformer model with LM head.
///
/// Key path hierarchy matches HuggingFace weight names:
/// ```
/// model.embed_tokens.weight
/// model.layers.0.self_attn.{q,k,v,o}_proj.weight
/// model.layers.0.mlp.{gate,up,down}_proj.weight
/// model.layers.0.input_layernorm.weight
/// model.layers.0.post_attention_layernorm.weight
/// model.norm.weight
/// lm_head.weight
/// ```
public class TransformerModel: Module {

    @ModuleInfo(key: "model") var model: TransformerModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: TransformerConfig

    public init(_ config: TransformerConfig) {
        self.config = config
        self._model.wrappedValue = TransformerModelInner(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    /// Run the full model: embedding -> layers -> norm -> LM head.
    ///
    /// - Parameters:
    ///   - inputIds: token IDs [B, seqLen]
    ///   - mask: optional attention mask
    ///   - kvCaches: per-layer KV caches
    /// - Returns: logits [B, seqLen, vocabSize]
    public func callAsFunction(
        _ inputIds: MLXArray,
        mask: MLXArray? = nil,
        kvCaches: [KVCache]? = nil
    ) -> MLXArray {
        let hidden = model(inputIds, mask: mask, kvCaches: kvCaches)
        return lmHead(hidden)
    }

    /// Create a causal attention mask for the given sequence length.
    /// Positions that should NOT be attended to are set to a large negative value.
    public static func createCausalMask(seqLen: Int, offset: Int = 0, dtype: DType = .float16) -> MLXArray {
        let totalLen = seqLen + offset
        let queryIndices = MLXArray(offset..<totalLen)
        let keyIndices = MLXArray(0..<totalLen)
        // mask[i,j] = large_neg where query position i < key position j (can't attend to future)
        var mask = expandedDimensions(queryIndices, axis: 1) .< expandedDimensions(keyIndices, axis: 0)
        mask = mask.asType(dtype) * Float(-1e9)
        return mask
    }

    /// Create empty KV caches for all layers.
    public func createKVCaches() -> [KVCache] {
        (0..<config.numLayers).map { _ in KVCache() }
    }

    /// Load weights from a flat `[String: MLXArray]` dictionary (e.g., from safetensors).
    /// Converts the flat dictionary to the nested `ModuleParameters` structure
    /// and applies it to the model.
    public func loadWeights(_ weights: [String: MLXArray]) {
        let parameters = ModuleParameters.unflattened(weights)
        self.update(parameters: parameters)
    }

    /// Convenience factory: create model from a config dictionary and raw weights.
    ///
    /// Note: This is the custom/experimental model path. The primary path uses
    /// mlx-swift-lm's model factory via `ModelLoader.load(from:)` which handles
    /// 50+ architectures with proven weight loading (including QuantizedLinear).
    public static func from(config: [String: Any], weights: [String: MLXArray]) -> TransformerModel {
        let tfConfig = TransformerConfig.from(config: config)
        let model = TransformerModel(tfConfig)
        model.loadWeights(weights)
        return model
    }
}

// MARK: - ModelForwardPass Conformance

/// Wraps a `TransformerModel` to conform to `ModelForwardPass`.
///
/// The protocol requires `Sendable` but `Module` is not `Sendable`.
/// This wrapper uses `@unchecked Sendable` because the model is only
/// accessed from the inference actor in practice.
///
/// The protocol uses `[MLXArray]` for the cache, but our model needs
/// structured `KVCache` objects. This wrapper manages the `KVCache` array
/// internally and uses the protocol's `[MLXArray]` as a placeholder
/// (empty array passed through).
public final class TransformerModelForwardPass: @unchecked Sendable {
    public let model: TransformerModel
    private var kvCaches: [KVCache]

    public init(model: TransformerModel) {
        self.model = model
        self.kvCaches = model.createKVCaches()
    }

    /// Reset all KV caches (e.g., for a new conversation).
    public func resetCaches() {
        kvCaches = model.createKVCaches()
    }

    /// Load cached KV state from a `HybridCache` into this model's internal KV caches.
    /// Only attention layers are loaded; SSM layers are skipped (handled separately).
    /// Call this after a cache hit to restore previously computed KV state.
    public func loadCache(_ cache: HybridCache) {
        for (index, entry) in cache.layers.enumerated() {
            guard index < kvCaches.count else { break }
            if case .attention(let layer) = entry {
                let kvc = KVCache()
                // Populate directly — update() concatenates, but we want to set initial state.
                kvc.keys = layer.keys
                kvc.values = layer.values
                kvCaches[index] = kvc
            }
        }
    }

    /// Export the current KV cache state as a `HybridCache` for storage in the CacheCoordinator.
    /// Each layer's KV cache is converted to a `LayerCacheEntry.attention`.
    /// Returns a pure-attention HybridCache (no SSM layers for standard transformers).
    public func exportCache() -> HybridCache {
        let entries = kvCaches.map { kvc -> LayerCacheEntry in
            .attention(KVCacheLayer(
                keys: kvc.keys ?? MLXArray(),
                values: kvc.values ?? MLXArray(),
                offset: kvc.sequenceLength
            ))
        }
        return HybridCache(layers: entries)
    }
}

extension TransformerModelForwardPass: ModelForwardPass {

    public var vocabSize: Int { model.config.vocabSize }
    public var layerCount: Int { model.config.numLayers }

    public func prefill(
        inputIds: MLXArray,
        cache: inout [MLXArray],
        mask: MLXArray?
    ) async throws -> MLXArray {
        let seqLen = inputIds.dim(inputIds.ndim - 1)
        let offset = kvCaches.first?.sequenceLength ?? 0

        // Build causal mask for the full sequence (cached + new tokens)
        let causalMask: MLXArray?
        if seqLen > 1 {
            causalMask = TransformerModel.createCausalMask(
                seqLen: seqLen, offset: offset,
                dtype: .float16
            )
        } else {
            causalMask = nil
        }

        // Ensure input has batch dimension [B, seqLen]
        let batchedInput: MLXArray
        if inputIds.ndim == 1 {
            batchedInput = inputIds.reshaped(1, seqLen)
        } else {
            batchedInput = inputIds
        }

        let logits = model(batchedInput, mask: causalMask, kvCaches: kvCaches)
        MLX.eval(logits)

        // Return logits for the last token position: [B, vocabSize]
        let lastLogits = logits[0..., (seqLen - 1), 0...]
        return lastLogits
    }

    public func decode(
        tokenId: Int,
        cache: inout [MLXArray]
    ) async throws -> MLXArray {
        let inputIds = MLXArray([Int32(tokenId)]).reshaped(1, 1)

        // Single-token decode: no causal mask needed (attending to all cached + this token)
        let logits = model(inputIds, mask: nil, kvCaches: kvCaches)
        MLX.eval(logits)

        // Return logits: [B, vocabSize] (squeeze the seqLen=1 dimension)
        return logits.squeezed(axis: 1)
    }
}
