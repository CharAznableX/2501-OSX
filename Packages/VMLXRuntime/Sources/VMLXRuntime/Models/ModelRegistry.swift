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

/// Registry of supported model architectures.
/// Maps `model_type` from config.json to model construction + weight loading.
public struct VMLXModelRegistry {

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

        switch modelType {
        case "qwen3_5":
            // Has language_model wrapper (VL-compatible or top-level)
            let config = try JSONDecoder().decode(Qwen35Configuration.self, from: configData)
            model = Qwen35TopLevelModel(config)

        case "qwen3_5_text":
            // Direct text model (no language_model wrapper)
            let config = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: configData)
            model = Qwen35TextModel(config)

        case "qwen3_5_moe":
            // MoE variant uses same top-level wrapper
            let config = try JSONDecoder().decode(Qwen35Configuration.self, from: configData)
            model = Qwen35TopLevelModel(config)

        default:
            throw ModelLoaderError.unsupportedArchitecture(
                "Native model type '\(modelType)' is not yet supported. " +
                "Supported: qwen3_5, qwen3_5_text, qwen3_5_moe"
            )
        }

        // Load weights
        try vmlxLoadWeights(
            modelDirectory: directory,
            model: model,
            quantization: baseConfig.quantization
        )

        return (model, modelType)
    }

    /// Check if a model_type is supported natively.
    public static func isSupported(modelType: String) -> Bool {
        switch modelType {
        case "qwen3_5", "qwen3_5_text", "qwen3_5_moe":
            return true
        default:
            return false
        }
    }
}
