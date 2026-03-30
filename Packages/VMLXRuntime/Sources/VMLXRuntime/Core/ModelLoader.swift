import Foundation
import MLX
import MLXNN
import Tokenizers
import Hub

/// A loaded model ready for inference.
public final class LoadedModel: @unchecked Sendable {
    /// Model weights as flat dictionary.
    public let weights: [String: MLXArray]

    /// Tokenizer for encoding/decoding text.
    public let tokenizer: any Tokenizer

    /// Model configuration from config.json.
    public let config: [String: Any]

    /// Detected model properties.
    public let detected: DetectedModel

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
        if let eos = tokenizer.eosTokenId { ids.insert(eos) }
        // Check config for additional stop tokens
        if let eosIds = config["eos_token_id"] as? [Int] {
            ids.formUnion(eosIds)
        } else if let eosId = config["eos_token_id"] as? Int {
            ids.insert(eosId)
        }
        return ids
    }

    public init(weights: [String: MLXArray], tokenizer: any Tokenizer,
                config: [String: Any], detected: DetectedModel) {
        self.weights = weights
        self.tokenizer = tokenizer
        self.config = config
        self.detected = detected
    }
}

/// Loads models from disk (safetensors weights + tokenizer).
public struct ModelLoader: Sendable {

    /// Load a model from a directory path.
    /// Reads safetensors weight files, tokenizer, and config.
    public static func load(from path: URL) async throws -> LoadedModel {
        // 1. Detect model
        let detected = try ModelDetector.detect(at: path)

        // 2. Load config.json
        let config = try _loadConfig(at: path)

        // 3. Load tokenizer
        let tokenizer = try await _loadTokenizer(at: path)

        // 4. Load weights from safetensors
        let weights = try _loadWeights(at: path)

        return LoadedModel(
            weights: weights,
            tokenizer: tokenizer,
            config: config,
            detected: detected
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

    private static func _loadTokenizer(at path: URL) async throws -> any Tokenizer {
        // AutoTokenizer.from(modelFolder:) handles tokenizer_config.json + tokenizer.json
        let tokenizer = try await AutoTokenizer.from(modelFolder: path)
        return tokenizer
    }

    private static func _loadWeights(at path: URL) throws -> [String: MLXArray] {
        // Check for sharded model (model.safetensors.index.json)
        let indexURL = path.appendingPathComponent("model.safetensors.index.json")

        if FileManager.default.fileExists(atPath: indexURL.path) {
            return try _loadShardedWeights(indexURL: indexURL, basePath: path)
        }

        // Single file model
        let singleURL = path.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: singleURL.path) {
            return try loadArrays(url: singleURL)
        }

        // Try weights.safetensors (some models use this name)
        let altURL = path.appendingPathComponent("weights.safetensors")
        if FileManager.default.fileExists(atPath: altURL.path) {
            return try loadArrays(url: altURL)
        }

        throw ModelLoaderError.weightsNotFound(path.path)
    }

    private static func _loadShardedWeights(indexURL: URL, basePath: URL) throws -> [String: MLXArray] {
        let data = try Data(contentsOf: indexURL)
        guard let index = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String] else {
            throw ModelLoaderError.invalidWeightIndex
        }

        // Get unique shard filenames
        let shardFiles = Set(weightMap.values).sorted()

        // Load each shard
        var allWeights: [String: MLXArray] = [:]
        for shardFile in shardFiles {
            let shardURL = basePath.appendingPathComponent(shardFile)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw ModelLoaderError.shardNotFound(shardFile)
            }
            let shardWeights = try loadArrays(url: shardURL)
            allWeights.merge(shardWeights) { _, new in new }
        }

        return allWeights
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
