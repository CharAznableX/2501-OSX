//
//  ModelRegistry.swift
//  VMLXRuntime
//
//  Registry mapping model_type strings to model constructors.
//  Creates the correct Module subclass for a given model architecture
//  and loads weights into it.
//

import Foundation
import MLX
import MLXNN

/// Protocol that all VMLXRuntime-native models must implement.
/// Provides the forward pass, cache creation, and weight sanitization.
public protocol VMLXNativeModel: AnyObject {
    /// Run the forward pass: tokens in, logits out.
    func callAsFunction(_ inputs: MLXArray, cache: [VMLXKVCache]?) -> MLXArray

    /// Create fresh KV/SSM caches for all layers.
    func newCache() -> [VMLXKVCache]

    /// Number of vocabulary tokens (for sampling).
    var vocabularySize: Int { get }
}

// Make our model types conform
extension Qwen35TopLevelModel: VMLXNativeModel, VMLXSanitizable {}
extension Qwen35TextModel: VMLXNativeModel, VMLXSanitizable {}
extension NemotronHModel: VMLXNativeModel {}
// GPTOSSTransformerModel conforms via extension in GPTOSSModel.swift

/// Registry of supported model architectures.
/// Maps `model_type` from config.json to model construction + weight loading.
public struct VMLXModelRegistry {

    /// Standard transformer model types that use the generic StandardTransformerModel.
    /// These all share the same weight key layout: model.layers.N.self_attn/mlp.
    private static let standardTransformerTypes: Set<String> = [
        "llama",            // Llama 2/3/3.1/3.2/3.3/4
        "qwen2",            // Qwen 2/2.5
        "qwen3",            // Qwen 3
        "mistral",          // Mistral v0.1/v0.2/v0.3 (standard quantization only)
        "minimax_m2",       // MiniMax M2.5 (MoE + q/k norm)
        "gemma2",           // Gemma 2
        "gemma3",           // Gemma 3
        "gemma3_text",      // Gemma 3 text
        "phi3",             // Phi 3
        "phi4mm",           // Phi 4
        "starcoder2",       // StarCoder 2
        "internlm2",        // InternLM 2
        "granite",          // IBM Granite
        "cohere",           // Command-R
        "cohere2",          // Command-R2
        "exaone",           // LG ExaOne
        "olmo",             // OLMo
        "olmo2",            // OLMo 2
        "stablelm",         // StableLM
    ]

    /// Model types that need dedicated implementations (NOT StandardTransformerModel).
    /// These are handled by mlx-swift-lm (MLXService) which has correct model classes.
    /// VMLXRuntime will add native support for these in future phases.
    public static let mlxServiceOnlyTypes: Set<String> = [
    ]

    /// Model types that use FP8 quantization or other unsupported weight formats.
    /// These cannot be loaded by VMLXRuntime and get a clear error message.
    private static let unsupportedTypes: Set<String> = [
    ]

    /// Load a native model from a directory.
    ///
    /// 1. Reads config.json to determine model_type and quantization
    /// 2. Creates the correct Module subclass
    /// 3. Loads and applies weights (with sanitization and quantization)
    /// 4. Returns the model as a VMLXNativeModel
    public static func loadModel(from directory: URL) throws -> (model: VMLXNativeModel & Module, modelType: String) {
        // Read config.json
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)

        // Parse base config for model_type and quantization
        let baseConfig = try JSONDecoder().decode(VMLXBaseConfiguration.self, from: configData)

        // Create the model based on model_type
        let model: VMLXNativeModel & Module
        let modelType = baseConfig.modelType

        // Reject unsupported models with a clear error
        if unsupportedTypes.contains(modelType) {
            throw ModelLoaderError.unsupportedArchitecture(
                "\(modelType): architecture not yet supported. "
                + "Supported: Qwen3.5 (hybrid SSM), Nemotron-H (Mamba2+MoE), standard transformers, Mistral4 (MLA+MoE)."
            )
        }

        switch modelType {
        // Nemotron-H hybrid (Mamba2 SSM + GQA attention + MoE-MLP)
        case "nemotron_h":
            let config = try JSONDecoder().decode(NemotronHConfiguration.self, from: configData)
            model = NemotronHModel(config)
        // Qwen3.5 hybrid (GatedDeltaNet + GQA + optional MoE)
        case "qwen3_5":
            let config = try JSONDecoder().decode(Qwen35Configuration.self, from: configData)
            model = Qwen35TopLevelModel(config)

        case "qwen3_5_text":
            let config = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: configData)
            model = Qwen35TextModel(config)

        case "qwen3_5_moe", "qwen3_5_moe_text":
            let config = try JSONDecoder().decode(Qwen35Configuration.self, from: configData)
            model = Qwen35TopLevelModel(config)

        // GPT-OSS: MoE with softmax routing, sliding window, custom SwiGLU, attention sinks
        case "gpt_oss":
            let config = try JSONDecoder().decode(GPTOSSConfiguration.self, from: configData)
            model = GPTOSSTransformerModel(config)

        // Mistral Small 4: MLA attention + MoE with FP8 weights
        // Top-level model_type is "mistral3", text config model_type is "mistral4"
        case "mistral3", "mistral4":
            let config = try JSONDecoder().decode(Mistral4Configuration.self, from: configData)
            model = Mistral4TransformerModel(config.textConfig)

        default:
            // Standard transformer models (Llama, Qwen2, Qwen3, Mistral, Gemma, etc.)
            if standardTransformerTypes.contains(modelType) {
                let config = try JSONDecoder().decode(
                    StandardModelConfiguration.self, from: configData)
                model = StandardTransformerModel(config)
            } else {
                // Last resort: try loading as standard transformer anyway.
                // Many custom/fine-tuned models use the standard architecture
                // but have non-standard model_type strings.
                print("[VMLXModelRegistry] WARNING: Unknown model_type '\(modelType)' — loading as StandardTransformerModel. Output may be incorrect if architecture differs.")
                let config = try JSONDecoder().decode(
                    StandardModelConfiguration.self, from: configData)
                model = StandardTransformerModel(config)
            }
        }

        // Load weights
        try vmlxLoadWeights(
            modelDirectory: directory,
            model: model,
            quantization: baseConfig.quantization
        )

        // For large MoE models (≥256 experts), convert non-quantized weights to bfloat16.
        // This prevents float16 overflow in gate routing math and avoids implicit float32
        // promotion that kills performance (e.g., MiniMax M2.5 with 512 experts: 23→75+ tok/s).
        // Matches Python mlx-lm behavior.
        let numExperts = _getNumExperts(configData: configData)
        if numExperts >= 256 {
            NSLog("[ModelRegistry] Large MoE (\(numExperts) experts): converting to bfloat16")
            _convertToBFloat16(model: model)
        }

        return (model, modelType)
    }

    /// Check if a model_type is supported natively.
    public static func isSupported(modelType: String) -> Bool {
        switch modelType {
        case "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text":
            return true
        default:
            // Standard transformers + unknown types (fallback to standard)
            return true
        }
    }

    // MARK: - Private Helpers

    /// Extract num_local_experts from config data.
    private static func _getNumExperts(configData: Data) -> Int {
        guard let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            return 0
        }
        if let n = json["num_local_experts"] as? Int { return n }
        if let tc = json["text_config"] as? [String: Any],
           let n = tc["num_local_experts"] as? Int { return n }
        return 0
    }

    /// Convert all non-quantized floating-point parameters to bfloat16.
    /// Quantized weights (packed int) are left as-is.
    private static func _convertToBFloat16(model: Module) {
        let params = model.parameters()
        let converted = params.mapValues { (arr: MLXArray) -> MLXArray in
            if arr.dtype == .float16 || arr.dtype == .float32 {
                return arr.asType(.bfloat16)
            }
            return arr
        }
        _ = try? model.update(parameters: converted, verify: [])
    }
}
