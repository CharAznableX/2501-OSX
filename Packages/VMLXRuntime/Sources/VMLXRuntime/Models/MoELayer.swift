import Foundation
import MLX
import MLXNN

// MARK: - MoEConfig

/// Configuration for a Mixture of Experts layer, parsed from model config.json.
///
/// Supports two naming conventions from real model checkpoints:
/// - **Qwen3.5 MoE**: `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`,
///   `shared_expert_intermediate_size` (inside `text_config`)
/// - **MiniMax M2.5**: `num_local_experts`, `num_experts_per_tok`, `intermediate_size`
///   (top level, no shared expert)
public struct MoEConfig: Sendable {
    /// Total number of experts (typically 256).
    public let numExperts: Int

    /// Number of experts activated per token (top-k, typically 8).
    public let numExpertsPerTok: Int

    /// Model hidden dimension.
    public let hiddenSize: Int

    /// Per-expert FFN intermediate dimension.
    public let moeIntermediateSize: Int

    /// Whether a shared expert exists (Qwen3.5 style).
    public let hasSharedExpert: Bool

    /// Shared expert FFN intermediate dimension (nil if no shared expert).
    public let sharedExpertIntermediateSize: Int?

    /// Whether a gating sigmoid is applied to the shared expert output.
    public let hasSharedExpertGate: Bool

    public init(
        numExperts: Int,
        numExpertsPerTok: Int,
        hiddenSize: Int,
        moeIntermediateSize: Int,
        hasSharedExpert: Bool,
        sharedExpertIntermediateSize: Int?,
        hasSharedExpertGate: Bool
    ) {
        self.numExperts = numExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.hiddenSize = hiddenSize
        self.moeIntermediateSize = moeIntermediateSize
        self.hasSharedExpert = hasSharedExpert
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.hasSharedExpertGate = hasSharedExpertGate
    }

    /// Parse from a model's config.json dictionary.
    /// Checks top-level keys first, then falls back to `text_config` (for VL models).
    public static func from(config: [String: Any]) -> MoEConfig {
        let tc = config["text_config"] as? [String: Any]

        func get<T>(_ key: String, default d: T) -> T {
            (config[key] as? T) ?? (tc?[key] as? T) ?? d
        }

        let hidden: Int = get("hidden_size", default: 4096)
        let numExperts: Int = get("num_experts", default: get("num_local_experts", default: 256))
        let numPerTok: Int = get("num_experts_per_tok", default: 8)
        let moeInter: Int = get("moe_intermediate_size", default: get("intermediate_size", default: 1024))
        let sharedInter: Int = get("shared_expert_intermediate_size", default: 0)

        return MoEConfig(
            numExperts: numExperts,
            numExpertsPerTok: numPerTok,
            hiddenSize: hidden,
            moeIntermediateSize: moeInter,
            hasSharedExpert: sharedInter > 0,
            sharedExpertIntermediateSize: sharedInter > 0 ? sharedInter : nil,
            hasSharedExpertGate: sharedInter > 0
        )
    }
}

// MARK: - MoEGate

/// Router/gate that selects top-k experts per token.
///
/// Weight key mapping:
/// - `gate.weight` -> [numExperts, hiddenSize] (loaded as Linear-style row-major)
///
/// The gate computes logits via `x @ weight^T`, selects the top-k experts,
/// and returns softmax-normalized routing weights.
public class MoEGate: Module {

    /// Gate weight: [numExperts, hiddenSize].
    /// Stored as a Linear-style weight (transposed from the matmul perspective).
    @ParameterInfo(key: "weight") var weight: MLXArray

    let numExperts: Int
    let numExpertsPerTok: Int

    public init(hiddenSize: Int, numExperts: Int, numExpertsPerTok: Int) {
        self.numExperts = numExperts
        self.numExpertsPerTok = numExpertsPerTok
        self._weight.wrappedValue = MLXArray.zeros([numExperts, hiddenSize])
    }

    /// Route tokens to experts.
    ///
    /// - Parameter x: input tensor [tokens, hidden]
    /// - Returns: (indices: [tokens, k], weights: [tokens, k])
    ///   where indices are expert IDs and weights are softmax-normalized routing scores.
    public func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        // x: [tokens, hidden], weight: [numExperts, hidden]
        // logits = x @ weight^T -> [tokens, numExperts]
        let logits = matmul(x, weight.transposed())

        // Top-k selection: sort descending, take first k indices
        // argSort of negated logits gives descending order
        let sortedIndices = argSort(-logits, axis: -1)
        let topKIndices = sortedIndices[0..., ..<numExpertsPerTok]  // [tokens, k]

        // Gather the corresponding logits for the selected experts
        let topKLogits = takeAlong(logits, topKIndices, axis: -1)  // [tokens, k]

        // Softmax over the selected experts
        let routingWeights = softmax(topKLogits, axis: -1)  // [tokens, k]

        return (topKIndices, routingWeights)
    }
}

// MARK: - BatchedExperts

/// Batched expert FFNs -- all expert weights stored in single tensors.
///
/// Expert weights are stored as 3D tensors [numExperts, inDim, outDim].
/// Each expert performs a SwiGLU FFN: `down(silu(gate(x)) * up(x))`.
///
/// Weight key mapping (under `switch_mlp`):
/// - `switch_mlp.gate_proj.weight` -> [numExperts, hiddenSize, intermediateSize]
/// - `switch_mlp.up_proj.weight`   -> [numExperts, hiddenSize, intermediateSize]
/// - `switch_mlp.down_proj.weight`  -> [numExperts, intermediateSize, hiddenSize]
public class BatchedExperts: Module {

    @ParameterInfo(key: "gate_proj") var gateProj: MLXArray  // [E, hidden, inter]
    @ParameterInfo(key: "up_proj") var upProj: MLXArray      // [E, hidden, inter]
    @ParameterInfo(key: "down_proj") var downProj: MLXArray  // [E, inter, hidden]

    public init(numExperts: Int, hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = MLXArray.zeros([numExperts, hiddenSize, intermediateSize])
        self._upProj.wrappedValue = MLXArray.zeros([numExperts, hiddenSize, intermediateSize])
        self._downProj.wrappedValue = MLXArray.zeros([numExperts, intermediateSize, hiddenSize])
    }

    /// Run expert FFN for a single expert.
    ///
    /// - Parameters:
    ///   - x: input tensor [tokens, hidden]
    ///   - expertIndex: which expert to use (0..<numExperts)
    /// - Returns: output tensor [tokens, hidden]
    public func forward(_ x: MLXArray, expertIndex: Int) -> MLXArray {
        let gate = silu(matmul(x, gateProj[expertIndex]))
        let up = matmul(x, upProj[expertIndex])
        return matmul(gate * up, downProj[expertIndex])
    }
}

// MARK: - SharedExpert

/// Shared expert FFN (standard SwiGLU, not batched).
///
/// Applied to all tokens unconditionally and added to the MoE output.
/// Qwen3.5 models use this with a gating sigmoid.
///
/// Weight key mapping:
/// - `shared_expert.gate_proj.weight`
/// - `shared_expert.up_proj.weight`
/// - `shared_expert.down_proj.weight`
public class SharedExpert: Module {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

// MARK: - MoELayer

/// Full Mixture of Experts layer: router -> top-k expert dispatch -> optional shared expert.
///
/// Routing algorithm:
/// 1. Gate computes logits and selects top-k experts per token
/// 2. Each selected expert runs a SwiGLU FFN on the token
/// 3. Expert outputs are weighted by softmax routing weights and summed
/// 4. If a shared expert exists, its output (optionally gated by sigmoid) is added
///
/// Weight key mapping:
/// - `gate.weight`                          -> router weights
/// - `switch_mlp.{gate,up,down}_proj.weight` -> batched expert FFN weights
/// - `shared_expert.{gate,up,down}_proj.weight` -> shared expert (Qwen3.5 only)
/// - `shared_expert_gate.weight`            -> shared expert gate (Qwen3.5 only)
///
/// Supports:
/// - **Qwen3.5 MoE** (256 experts, top-8, shared expert with gate)
/// - **MiniMax M2.5** (256 experts, top-8, no shared expert)
public class MoELayer: Module {

    @ModuleInfo(key: "gate") var gate: MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: BatchedExperts
    @ModuleInfo(key: "shared_expert") var sharedExpert: SharedExpert?
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear?

    public let config: MoEConfig

    public init(config: MoEConfig) {
        self.config = config

        self._gate.wrappedValue = MoEGate(
            hiddenSize: config.hiddenSize,
            numExperts: config.numExperts,
            numExpertsPerTok: config.numExpertsPerTok
        )
        self._switchMLP.wrappedValue = BatchedExperts(
            numExperts: config.numExperts,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize
        )

        if config.hasSharedExpert, let sharedInter = config.sharedExpertIntermediateSize {
            self._sharedExpert.wrappedValue = SharedExpert(
                hiddenSize: config.hiddenSize,
                intermediateSize: sharedInter
            )
        }

        if config.hasSharedExpertGate {
            self._sharedExpertGate.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
        }
    }

    /// Forward pass through the MoE layer.
    ///
    /// - Parameter x: hidden states [batch, seq_len, hidden]
    /// - Returns: output hidden states [batch, seq_len, hidden]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape  // [batch, seq_len, hidden]
        let batchSeq = shape[0] * shape[1]
        let hidden = shape[2]

        // Flatten to [tokens, hidden] for routing and expert dispatch
        let flat = x.reshaped([batchSeq, hidden])

        // Route: get top-k expert indices and weights per token
        let (expertIndices, routingWeights) = gate(flat)
        // expertIndices: [tokens, k], routingWeights: [tokens, k]

        // Dispatch to experts and accumulate weighted outputs.
        // Strategy: iterate over expert slots (k), then over unique experts in each slot.
        // Each expert runs its FFN on all tokens, but output is masked to only the
        // tokens routed to that expert.
        var output = MLXArray.zeros([batchSeq, hidden])

        for slot in 0..<config.numExpertsPerTok {
            let slotIndices = expertIndices[0..., slot]   // [tokens] -- expert index per token
            let slotWeights = routingWeights[0..., slot]  // [tokens] -- routing weight per token

            // Force materialization so we can iterate unique expert IDs
            MLX.eval(slotIndices)
            let numTokens = slotIndices.dim(0)
            var expertSet = Set<Int>()
            for t in 0..<numTokens {
                expertSet.insert(slotIndices[t].item(Int.self))
            }

            for expertId in expertSet {
                // Boolean mask: which tokens are routed to this expert in this slot
                let mask = slotIndices .== MLXArray(Int32(expertId))
                let maskFloat = mask.asType(flat.dtype)

                // Run expert FFN on all tokens (cheap when masked by zero weight)
                let expertOutput = switchMLP.forward(flat, expertIndex: expertId)

                // Weight by routing weight and mask, then accumulate
                let weighted = expertOutput * expandedDimensions(slotWeights * maskFloat, axis: -1)
                output = output + weighted
            }
        }

        // Shared expert (Qwen3.5 style): applied to all tokens unconditionally
        if let shared = sharedExpert {
            var sharedOutput = shared(flat)
            if let sharedGateLayer = sharedExpertGate {
                let gateValue = sigmoid(sharedGateLayer(flat))  // [tokens, 1]
                sharedOutput = sharedOutput * gateValue
            }
            output = output + sharedOutput
        }

        return output.reshaped(shape)
    }
}
