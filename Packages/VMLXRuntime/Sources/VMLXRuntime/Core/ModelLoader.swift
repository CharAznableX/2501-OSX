import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import Tokenizers
import Hub

/// A loaded model ready for inference.
///
/// Holds both the mlx-swift-lm `ModelContainer` (proven model implementations for 50+
/// architectures including quantized weights) and VMLXRuntime metadata (JANG detection,
/// hybrid layer info, family config).
public final class LoadedModel: @unchecked Sendable {

    /// The mlx-swift-lm model container.
    /// Handles the actual forward pass, tokenizer, KV cache, and weight loading
    /// for all supported architectures (Qwen3.5, Nemotron-H, DeepSeek, Llama, etc.).
    public let mlxContainer: MLXLMCommon.ModelContainer

    /// Model configuration from config.json (raw dictionary for VMLXRuntime introspection).
    public let config: [String: Any]

    /// Detected model properties (name, family, hybrid, JANG, etc.).
    public let detected: DetectedModel

    /// Model directory path.
    public let modelPath: URL

    /// Vocabulary size.
    public var vocabSize: Int {
        if let vs = config["vocab_size"] as? Int { return vs }
        if let tc = config["text_config"] as? [String: Any],
           let vs = tc["vocab_size"] as? Int { return vs }
        return 32000
    }

    /// Number of layers.
    public var numLayers: Int {
        if let nl = config["num_hidden_layers"] as? Int { return nl }
        if let tc = config["text_config"] as? [String: Any],
           let nl = tc["num_hidden_layers"] as? Int { return nl }
        return 32
    }

    /// Hidden dimension.
    public var hiddenSize: Int {
        if let hs = config["hidden_size"] as? Int { return hs }
        if let tc = config["text_config"] as? [String: Any],
           let hs = tc["hidden_size"] as? Int { return hs }
        return 4096
    }

    /// Number of attention heads.
    public var numAttentionHeads: Int {
        if let nh = config["num_attention_heads"] as? Int { return nh }
        if let tc = config["text_config"] as? [String: Any],
           let nh = tc["num_attention_heads"] as? Int { return nh }
        return 32
    }

    /// Number of KV heads (for GQA).
    public var numKVHeads: Int {
        if let nk = config["num_key_value_heads"] as? Int { return nk }
        if let tc = config["text_config"] as? [String: Any],
           let nk = tc["num_key_value_heads"] as? Int { return nk }
        return numAttentionHeads
    }

    /// EOS token IDs.
    public var eosTokenIds: Set<Int> {
        var ids = Set<Int>()
        // Check config for stop tokens
        if let eosIds = config["eos_token_id"] as? [Int] {
            ids.formUnion(eosIds)
        } else if let eosId = config["eos_token_id"] as? Int {
            ids.insert(eosId)
        }
        return ids
    }

    /// Tokenizer (accessed via mlx-swift-lm container).
    public var tokenizer: any Tokenizer {
        get async {
            await mlxContainer.tokenizer
        }
    }

    public init(mlxContainer: MLXLMCommon.ModelContainer,
                config: [String: Any], detected: DetectedModel,
                modelPath: URL) {
        self.mlxContainer = mlxContainer
        self.config = config
        self.detected = detected
        self.modelPath = modelPath
    }
}

/// Loads models from disk using mlx-swift-lm's proven model factory.
///
/// Instead of custom weight loading and model construction, delegates to
/// `MLXLMCommon.loadModelContainer(directory:)` which handles:
/// - Config parsing and model architecture detection (50+ families)
/// - Quantized weight loading (QuantizedLinear with scales/biases)
/// - Correct weight key path mapping for each architecture
/// - Tokenizer setup
public struct ModelLoader: Sendable {

    /// Load a model from a directory path using mlx-swift-lm's model factory.
    ///
    /// This is the primary loading path:
    /// 1. Uses `MLXLMCommon.loadModelContainer(directory:)` for proven weight loading
    /// 2. Runs `ModelDetector.detect(at:)` for JANG/hybrid/family metadata
    /// 3. Parses config.json for VMLXRuntime introspection
    public static func load(from path: URL) async throws -> LoadedModel {
        // 1. Detect model properties (JANG, hybrid, family, etc.)
        let detected = try ModelDetector.detect(at: path)

        // 2. Load config.json for VMLXRuntime metadata
        let config = try _loadConfig(at: path)

        // 3. Use mlx-swift-lm's model factory to load the model.
        //    This handles ALL architectures: Qwen3.5, Nemotron-H, DeepSeek-V3,
        //    Llama, Mistral, Gemma, Phi, MiniMax, GLM4, Falcon-H1, etc.
        //    It correctly handles:
        //    - QuantizedLinear (scales, biases, weight packing)
        //    - Architecture-specific weight key paths
        //    - Tokenizer loading and chat template setup
        let modelConfig = MLXLMCommon.ModelConfiguration(directory: path)
        let mlxContainer: MLXLMCommon.ModelContainer
        do {
            mlxContainer = try await MLXLMCommon.loadModelContainer(
                configuration: modelConfig
            )
        } catch {
            throw ModelLoaderError.unsupportedArchitecture(
                "mlx-swift-lm failed to load model at \(path.path): \(error.localizedDescription)"
            )
        }

        return LoadedModel(
            mlxContainer: mlxContainer,
            config: config,
            detected: detected,
            modelPath: path
        )
    }

    /// Load a model from a HuggingFace model name (downloads if needed).
    public static func loadFromHub(modelName: String) async throws -> LoadedModel {
        let hub = HubApi()
        let repo = Hub.Repo(id: modelName)
        let modelDir = try await hub.snapshot(from: repo)
        return try await load(from: modelDir)
    }

    // MARK: - Private

    private static func _loadConfig(at path: URL) throws -> [String: Any] {
        let configURL = path.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw ModelLoaderError.configNotFound(path.path)
        }
        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelLoaderError.invalidConfig("Failed to parse config.json")
        }
        return json
    }
}

// MARK: - Errors

public enum ModelLoaderError: Error, LocalizedError, Sendable {
    case configNotFound(String)
    case invalidConfig(String)
    case tokenizerNotFound(String)
    case weightsNotFound(String)
    case invalidWeightIndex
    case shardNotFound(String)
    case unsupportedArchitecture(String)

    public var errorDescription: String? {
        switch self {
        case .configNotFound(let p): return "config.json not found at: \(p)"
        case .invalidConfig(let m): return "Invalid config: \(m)"
        case .tokenizerNotFound(let p): return "Tokenizer not found at: \(p)"
        case .weightsNotFound(let p): return "No safetensors weights at: \(p)"
        case .invalidWeightIndex: return "Invalid model.safetensors.index.json"
        case .shardNotFound(let f): return "Weight shard not found: \(f)"
        case .unsupportedArchitecture(let a): return "Unsupported architecture: \(a)"
        }
    }
}
