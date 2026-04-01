//
//  GPTOSSModel.swift
//  VMLXRuntime
//
//  Ported from mlx-swift-lm's MLXLLM/Models/GPTOSS.swift
//  GPT-OSS: MoE with softmax routing, sliding window attention,
//  custom SwiGLU with alpha clipping, and attention sinks.
//
//  Uses VMLXKVCache protocol, VMLXSwitchLinear, VMLXYarnRoPE,
//  and other VMLXRuntime utilities instead of MLXLMCommon.
//

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - Configuration

public struct GPTOSSConfiguration: Codable, Sendable {
    public var modelType: String = "gpt_oss"
    public var hiddenLayers: Int = 36
    public var localExperts: Int = 128
    public var expertsPerToken: Int = 4
    public var vocabularySize: Int = 201088
    public var rmsNormEps: Float = 1e-5
    public var hiddenSize: Int = 2880
    public var intermediateSize: Int = 2880
    public var headDim: Int = 64
    public var attentionHeads: Int = 64
    public var kvHeads: Int = 8
    public var slidingWindow: Int = 128
    public var ropeTheta: Float = 150000
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var layerTypes: [String]? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case localExperts = "num_local_experts"
        case expertsPerToken = "num_experts_per_tok"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case headDim = "head_dim"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case slidingWindow = "sliding_window"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.localExperts = try container.decode(Int.self, forKey: .localExperts)
        self.expertsPerToken = try container.decode(Int.self, forKey: .expertsPerToken)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 150000
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
    }
}

// MARK: - Top-K Helper

private func mlxTopK(_ a: MLXArray, k: Int, axis: Int = -1) -> (values: MLXArray, indices: MLXArray)
{
    let partitionedIndices = argPartition(a, kth: -k, axis: axis)
    let topKIndices = partitionedIndices[.ellipsis, (-k)...]
    let topKValues = takeAlong(a, topKIndices, axis: axis)
    return (topKValues, topKIndices)
}

// MARK: - Custom SwiGLU (alpha=1.702, limit=7.0)

private func gptossSwiGLU(
    _ xLinear: MLXArray, _ xGlu: MLXArray, alpha: Float = 1.702, limit: Float = 7.0
) -> MLXArray {
    var xLinear = xLinear
    var xGlu = xGlu
    xGlu = clip(xGlu, max: MLXArray(limit))
    xLinear = clip(xLinear, min: MLXArray(-limit), max: MLXArray(limit))

    let gluScaled = alpha * xGlu
    let sig = sigmoid(gluScaled)

    let outGlu = xGlu * sig
    return outGlu * (xLinear + 1)
}

private let compiledGPTOSSSwiglu: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { xLinear, xGlu in
    gptossSwiGLU(xLinear, xGlu)
}

// MARK: - GPT-OSS SwiGLU SwitchGLU (MoE expert layer)

/// MoE SwitchGLU using GPT-OSS custom SwiGLU instead of standard silu.
/// All projections have bias=true.
class GPTOSSSwiGLUSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: VMLXSwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: VMLXSwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: VMLXSwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int

    init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = true
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts

        _gateProj.wrappedValue = VMLXSwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _upProj.wrappedValue = VMLXSwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _downProj.wrappedValue = VMLXSwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = vmlxGatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            compiledGPTOSSSwiglu(xUp, xGate),
            idx,
            sortedIndices: doSort)

        if doSort {
            x = vmlxScatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return x.squeezed(axis: -2)
    }
}

// MARK: - Attention Mask Helper

/// Create a causal mask with optional sliding window for GPT-OSS per-layer masks.
private func gptossCreateMask(
    n: Int, cache: VMLXKVCache?, windowSize: Int?
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    if n > 1 {
        let offset = cache?.offset ?? 0
        if offset == 0 && windowSize == nil {
            return .causal
        }
        let mask = vmlxCreateCausalMask(n: n, offset: offset, windowSize: windowSize)
        return .array(mask)
    }
    return .none
}

// MARK: - Attention

class GPTOSSAttention: Module {
    let headDim: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numKeyValueGroups: Int
    let smScale: Float

    @ParameterInfo(key: "sinks") var sinks: MLXArray
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: VMLXRoPELayer
    private var cachedSinksActive: Bool?

    init(_ config: GPTOSSConfiguration) {
        self.headDim = config.headDim
        self.numAttentionHeads = config.attentionHeads
        self.numKeyValueHeads = config.kvHeads
        self.numKeyValueGroups = config.attentionHeads / config.kvHeads

        _sinks.wrappedValue = zeros([config.attentionHeads])
        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: true)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: true)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: true)
        _oProj.wrappedValue = Linear(
            config.headDim * config.attentionHeads, config.hiddenSize, bias: true)

        self.smScale = 1.0 / sqrt(Float(config.headDim))

        // Use vmlxInitializeRope which handles YARN scaling automatically
        self.rope = vmlxInitializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: nil
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VMLXKVCache? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let D = headDim

        var q = qProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        var k = kProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)
        var v = vProj(x).reshaped(B, L, -1, D).swappedAxes(1, 2)

        let sinksActive =
            cachedSinksActive
            ?? {
                let active = (sinks * sinks).max().item(Float.self) > 0
                cachedSinksActive = active
                return active
            }()

        // Apply RoPE with cache offset
        let offset = cache?.offset ?? 0
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        // Update KV cache
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let vHat = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: smScale,
            mask: mask,
            sinks: sinksActive ? sinks : nil)

        return oProj(vHat.swappedAxes(1, 2).reshaped(B, L, -1))
    }
}

// MARK: - MLP Block (MoE with softmax routing)

class GPTOSSMLPBlock: Module {
    let hiddenSize: Int
    let numLocalExperts: Int
    let numExpertsPerTok: Int

    @ModuleInfo(key: "experts") var experts: GPTOSSSwiGLUSwitchGLU
    @ModuleInfo(key: "router") var router: Linear

    init(_ config: GPTOSSConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.numLocalExperts = config.localExperts
        self.numExpertsPerTok = config.expertsPerToken

        _experts.wrappedValue = GPTOSSSwiGLUSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.localExperts,
            bias: true
        )
        _router.wrappedValue = Linear(config.hiddenSize, config.localExperts, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = router(x)
        let (experts, indices) = mlxTopK(g, k: numExpertsPerTok, axis: -1)
        let stopIndices = MLX.stopGradient(indices)
        let expertWeights = softmax(experts, axis: -1, precise: true)

        var y = self.experts(x, stopIndices)

        y = y * expandedDimensions(expertWeights, axis: -1)
        return y.sum(axis: -2)
    }
}

// MARK: - Decoder Layer

class GPTOSSDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: GPTOSSAttention
    @ModuleInfo(key: "mlp") var mlp: GPTOSSMLPBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GPTOSSConfiguration) {
        _selfAttn.wrappedValue = GPTOSSAttention(config)
        _mlp.wrappedValue = GPTOSSMLPBlock(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VMLXKVCache? = nil
    ) -> MLXArray {
        var residual = x
        var h = inputLayerNorm(x)
        h = selfAttn(h, mask: mask, cache: cache)
        h = residual + h

        residual = h
        h = postAttentionLayerNorm(h)
        h = mlp(h)
        h = residual + h
        return h
    }
}

// MARK: - Model Inner

class GPTOSSModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let layerTypes: [String]
    fileprivate let layers: [GPTOSSDecoderLayer]
    let windowSize: Int
    let slidingAttentionIndex: Int
    let fullAttentionIndex: Int

    init(_ config: GPTOSSConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layerTypes =
            config.layerTypes
            ?? Array(
                repeating: [
                    "sliding_attention",
                    "full_attention",
                ], count: config.hiddenLayers / 2
            ).flatMap { $0 }
        self.layers = (0 ..< config.hiddenLayers).map { _ in GPTOSSDecoderLayer(config) }
        self.windowSize = config.slidingWindow
        self.slidingAttentionIndex =
            self.layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.fullAttentionIndex =
            self.layerTypes.firstIndex(of: "full_attention") ?? 0
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [VMLXKVCache?]? = nil
    ) -> MLXArray {
        var x = embedTokens(inputs)

        let caches: [VMLXKVCache?] = cache ?? [VMLXKVCache?](repeating: nil, count: layers.count)

        let seqLen = x.dim(1)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

        for (i, layer) in layers.enumerated() {
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if layerTypes[i] == "full_attention" {
                if fullMask == nil {
                    fullMask = gptossCreateMask(
                        n: seqLen,
                        cache: caches[fullAttentionIndex],
                        windowSize: nil
                    )
                }
                maskMode = fullMask!
            } else {
                if slidingMask == nil {
                    slidingMask = gptossCreateMask(
                        n: seqLen,
                        cache: caches[slidingAttentionIndex],
                        windowSize: windowSize
                    )
                }
                maskMode = slidingMask!
            }

            x = layer(x, mask: maskMode, cache: caches[i])
        }

        x = norm(x)

        return x
    }
}

// MARK: - MXFP4 Packed Tensor Conversion

private func convertMoePackedTensors(blocks: MLXArray, scales: MLXArray) -> MLXArray {
    precondition(
        blocks.shape.dropLast() == scales.shape,
        "blocks.shape=\(blocks.shape) does not match scales.shape=\(scales.shape)"
    )

    var scales = scales.asType(.int32) - 127
    let lut = MLXArray([
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]).asType(.bfloat16)

    let (prefixShape, G, B) = (Array(blocks.shape.dropLast(2)), blocks.dim(-2), blocks.dim(-1))

    let blocks = blocks.reshaped(-1, B)
    scales = scales.reshaped(-1, 1)

    let idxLo = blocks & 0x0F
    let idxHi = blocks >> 4

    var out = stacked([lut[idxLo], lut[idxHi]], axis: -1).flattened(start: -2)
    out = (2.0 ** scales) * out
    out = out.reshaped(prefixShape + [G * B * 2])
    return out.asType(.bfloat16)
}

// MARK: - Top-Level Model

public class GPTOSSTransformerModel: Module {
    public let vocabularySize: Int
    let configuration: GPTOSSConfiguration

    @ModuleInfo(key: "model") var model: GPTOSSModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: GPTOSSConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        _model.wrappedValue = GPTOSSModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [VMLXKVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead(hidden)
    }

    public func newCache() -> [VMLXKVCache] {
        // VMLXRuntime doesn't have RotatingKVCache, so use VMLXKVCacheSimple for all layers.
        // This works correctly but uses more memory for sliding window layers
        // compared to a rotating cache that would evict old entries.
        (0 ..< configuration.hiddenLayers).map { _ in VMLXKVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        // If weights already have gate_proj.weight keys, they're already sanitized
        if weights.keys.contains(where: { $0.contains("gate_proj.weight") }) {
            return weights
        }

        // Handle MXFP4 packed tensor format (gate_up_proj_blocks/scales)
        if weights.keys.contains(where: { $0.contains("gate_up_proj_scales") }) {
            var newWeights: [String: MLXArray] = [:]
            for (k, v) in weights {
                if k.hasSuffix("_scales") {
                    continue
                } else if k.hasSuffix("_blocks") {
                    let scaleKey = k.replacingOccurrences(of: "_blocks", with: "_scales")
                    if let scales = weights[scaleKey] {
                        let newV = convertMoePackedTensors(blocks: v, scales: scales)
                        let newK = k.replacingOccurrences(of: "_blocks", with: "")
                        newWeights[newK] = newV
                    }
                } else {
                    newWeights[k] = v
                }
            }
            weights = newWeights
        }

        // Split interleaved gate_up_proj into separate gate_proj and up_proj
        var finalWeights: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.contains("gate_up_proj"), !k.contains("bias") {
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj", with: "gate_proj.weight")
                ] = contiguous(v[.ellipsis, .stride(by: 2), 0...])
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj", with: "up_proj.weight")
                ] = contiguous(v[.ellipsis, .stride(from: 1, by: 2), 0...])
            } else if k.contains("down_proj"), !k.contains("bias") {
                finalWeights[
                    k.replacingOccurrences(of: "down_proj", with: "down_proj.weight")
                ] = contiguous(v)
            } else if k.contains("gate_up_proj_bias") {
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj_bias", with: "gate_proj.bias")
                ] = contiguous(v[.ellipsis, .stride(by: 2)])
                finalWeights[
                    k.replacingOccurrences(of: "gate_up_proj_bias", with: "up_proj.bias")
                ] = contiguous(v[.ellipsis, .stride(from: 1, by: 2)])
            } else if k.contains("down_proj_bias") {
                finalWeights[
                    k.replacingOccurrences(of: "down_proj_bias", with: "down_proj.bias")
                ] = contiguous(v)
            } else {
                finalWeights[k] = v
            }
        }

        return finalWeights
    }
}

// MARK: - VMLXNativeModel + VMLXSanitizable

extension GPTOSSTransformerModel: VMLXNativeModel, VMLXSanitizable {}
