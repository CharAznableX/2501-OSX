//
//  WeightLoader.swift
//  VMLXRuntime
//
//  Weight loading for native VMLXRuntime models.
//  Ported from mlx-swift-lm's Load.swift.
//
//  Handles:
//  - Loading safetensors files from a model directory
//  - Calling model.sanitize() for weight key remapping
//  - Auto-quantizing Linear -> QuantizedLinear when weights have .scales
//  - Calling model.update(parameters:) to load weights into the model
//

import Foundation
import MLX
import MLXNN

// MARK: - Base Configuration

/// Parsed from config.json to extract quantization info and model_type.
public struct VMLXBaseConfiguration: Codable, Sendable {
    public let modelType: String

    public struct Quantization: Codable, Sendable {
        public let groupSize: Int
        public let bits: Int
        private var _mode: QuantizationMode? = nil
        public var mode: QuantizationMode { _mode ?? .affine }

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
            case _mode = "mode"
        }
    }

    public let quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
    }
}

// MARK: - Weight Loading

/// Protocol for models that can sanitize their weight keys.
public protocol VMLXSanitizable {
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

/// Load safetensors weights from a model directory, apply sanitization and quantization,
/// and update the model's parameters.
///
/// Note: The `eval(model)` call at the end is MLX's lazy evaluation trigger (not code eval).
/// It forces all pending MLX computations to materialize, which is required after weight loading.
public func vmlxLoadWeights(
    modelDirectory: URL,
    model: Module,
    quantization: VMLXBaseConfiguration.Quantization? = nil
) throws {
    // 1. Load all safetensors files
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }

    // 2. Model-specific key sanitization
    if let sanitizable = model as? VMLXSanitizable {
        weights = sanitizable.sanitize(weights: weights)
    }

    // 3. Auto-quantize: if weights contain .scales keys, convert Linear -> QuantizedLinear
    //
    // For JANG mixed-precision models: different layers use different bit widths
    // (e.g., SSM layers at 4-bit, attention at 6-bit, embedding at 4-bit).
    // We infer the actual bits per-layer from weight/scales shapes:
    //   bits = weight.dim(1) * 32 / (scales.dim(1) * group_size)
    let hasScales = weights.keys.contains { $0.hasSuffix(".scales") }
    if hasScales {
        let defaultGroupSize = quantization?.groupSize ?? 64
        let defaultBits = quantization?.bits ?? 4
        let mode = quantization?.mode ?? .affine

        // Check if the config-level bits is unsupported by MLX (e.g. JANG_4K uses 3-bit)
        let mlxValidBits = [2, 4, 6, 8]
        if !mlxValidBits.contains(defaultBits) {
            throw ModelLoaderError.unsupportedArchitecture(
                "Model uses \(defaultBits)-bit quantization which MLX does not support. "
                + "Supported: 2, 4, 6, 8-bit. JANG_4K (3-bit) requires a custom dequantization kernel."
            )
        }

        quantize(model: model) { path, module in
            guard let scales = weights["\(path).scales"],
                  let weight = weights["\(path).weight"] else {
                return nil
            }

            // Infer actual bits AND group_size from weight/scales shapes.
            // JANG mixed-precision: layers use different bit widths AND group sizes.
            //
            // For Linear/QuantizedLinear (2D weights [out, packed_in]):
            //   in_features is known from the module's input dimension
            //   bits = weight_cols * 32 / in_features
            //   group_size = in_features / scales_cols
            //
            // For SwitchLinear (3D weights [experts, out, packed_in]):
            //   Same logic on last two dimensions
            let weightCols = weight.dim(weight.ndim - 1)
            let scalesCols = scales.dim(scales.ndim - 1)

            // Get the module's declared input features to compute bits precisely.
            // Linear: weight shape [out, in]
            // SwitchLinear: weight shape [experts, out, in]
            // Embedding: weight shape [vocab, dim]
            let inFeatures: Int
            if let linear = module as? Linear {
                inFeatures = linear.weight.dim(1)
            } else if let sw = module as? VMLXSwitchLinear {
                inFeatures = sw.inputDims
            } else if let emb = module as? Embedding {
                inFeatures = emb.weight.dim(1)
            } else {
                // Last resort: use config defaults
                return (defaultGroupSize, defaultBits, mode)
            }

            // Compute group_size from scales: group_size = in_features / scales_cols
            let inferredGroupSize = scalesCols > 0 ? inFeatures / scalesCols : defaultGroupSize
            // Compute bits from weight packing: bits = weight_cols * 32 / in_features
            let inferredBits = inFeatures > 0 ? (weightCols * 32) / inFeatures : defaultBits

            // MLX supports 2, 4, 6, 8 bit quantization only.
            // JANG _4K profile uses 3-bit which MLX cannot handle — skip these
            // weights. They'll fail at model.update() with a clear shape mismatch
            // error instead of crashing in quantized_matmul.
            let validBits = [2, 4, 6, 8]
            guard validBits.contains(inferredBits) else {
                return nil  // Skip — unsupported bit width
            }
            let safeBits = inferredBits
            let safeGroupSize = inferredGroupSize > 0 ? inferredGroupSize : defaultGroupSize

            return (safeGroupSize, safeBits, mode)
        }
    }

    // 4. Load weights into model
    // Use .noUnusedKeys to catch weight naming errors, but allow missing keys
    // (e.g., bias parameters that exist in the model but not in the weights —
    //  they stay at their initialized zero values, which is correct behavior
    //  for models like Qwen2 where Q/K/V have bias but O does not).
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // 5. Weights are loaded lazily — they materialize on GPU when first accessed
    // during inference. No need to force-eval here, which avoids OOM on large
    // models where materializing all weights at once exceeds Metal memory.
}
