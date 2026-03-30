import Foundation
import MLX
import MLXNN
import Tokenizers

/// Container wrapping a loaded model with runtime configuration.
/// This is what gets passed around during inference.
public final class VMLXModelContainer: @unchecked Sendable {

    /// The loaded model (weights + tokenizer + config).
    public let model: LoadedModel

    /// The native model for direct forward pass access.
    public let nativeModel: any VMLXNativeModel & Module

    /// Tokenizer reference.
    public let tokenizer: any Tokenizer

    /// Model name/path.
    public let name: String

    /// Whether this is a JANG model.
    public var isJang: Bool { model.detected.isJang }

    /// Whether this is a hybrid model (SSM + attention).
    public var isHybrid: Bool { model.detected.isHybrid }

    /// Whether this model supports vision.
    public var hasVision: Bool { model.detected.hasVision }

    /// TurboQuant config (nil if not a JANG model or TQ disabled).
    public let turboQuantConfig: TurboQuantConfig?

    /// Layer pattern for hybrid models.
    public let layerPattern: [LayerType]?

    /// EOS token IDs for stop detection.
    public let eosTokenIds: Set<Int>

    /// Model family config (tool format, reasoning format, etc.).
    public let familyConfig: ModelFamilyConfig

    private init(model: LoadedModel,
                 turboQuantConfig: TurboQuantConfig?, layerPattern: [LayerType]?) {
        self.model = model
        self.nativeModel = model.nativeModel
        self.tokenizer = model.tokenizer
        self.name = model.detected.name
        self.eosTokenIds = model.eosTokenIds
        self.familyConfig = ModelConfigRegistry.configFor(modelName: model.detected.name)
        self.turboQuantConfig = turboQuantConfig
        self.layerPattern = layerPattern
    }

    /// Factory method. Builds TQ/hybrid configuration from detected model properties.
    public static func create(model: LoadedModel) -> VMLXModelContainer {
        // Build TQ config and layer pattern
        let turboQuantConfig: TurboQuantConfig?
        let layerPattern: [LayerType]?

        if model.detected.isJang,
           JangLoader.isJangModel(at: model.detected.modelPath),
           let jangConfig = try? JangLoader.loadConfig(at: model.detected.modelPath) {

            let detectedLayerPattern: [LayerType]?
            if let patternStr = model.detected.hybridOverridePattern {
                detectedLayerPattern = parseHybridPattern(patternStr)
            } else if let layerTypeStrs = model.detected.layerTypes {
                detectedLayerPattern = layerTypeStrs.map { str -> LayerType in
                    switch str.lowercased() {
                    case "attention", "attn": return .attention
                    case "ssm", "mamba", "recurrent": return .ssm
                    default: return .attention
                    }
                }
            } else {
                detectedLayerPattern = nil
            }

            turboQuantConfig = JangLoader.buildTQConfig(
                from: jangConfig,
                layerPattern: detectedLayerPattern,
                kvLoraRank: model.detected.kvLoraRank,
                qkNopeHeadDim: model.detected.qkNopeHeadDim,
                qkRopeHeadDim: model.detected.qkRopeHeadDim
            )
            layerPattern = detectedLayerPattern
        } else {
            turboQuantConfig = nil

            if let patternStr = model.detected.hybridOverridePattern {
                layerPattern = parseHybridPattern(patternStr)
            } else if let layerTypeStrs = model.detected.layerTypes {
                layerPattern = layerTypeStrs.map { str -> LayerType in
                    switch str.lowercased() {
                    case "attention", "attn": return .attention
                    case "ssm", "mamba", "recurrent": return .ssm
                    default: return .attention
                    }
                }
            } else {
                layerPattern = nil
            }
        }

        return VMLXModelContainer(
            model: model,
            turboQuantConfig: turboQuantConfig,
            layerPattern: layerPattern
        )
    }

    // MARK: - Tokenization

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Decode token IDs to text.
    public func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    /// Apply chat template to messages and encode.
    public func applyChatTemplate(
        messages: [VMLXChatMessage],
        addGenerationPrompt: Bool = true
    ) throws -> [Int] {
        let chatMessages: [Message] = messages.map { msg in
            ["role": msg.role, "content": msg.textContent]
        }

        if tokenizer.hasChatTemplate {
            return try tokenizer.applyChatTemplate(
                messages: chatMessages,
                chatTemplate: nil,
                addGenerationPrompt: addGenerationPrompt,
                truncation: false,
                maxLength: nil,
                tools: nil
            )
        }

        let fullText = messages.map { msg in
            "\(msg.role): \(msg.textContent)"
        }.joined(separator: "\n")

        return encode(fullText)
    }

    /// Compute gen_prompt_len: difference between encoding with and without generation prompt.
    public func computeGenPromptLen(messages: [VMLXChatMessage]) -> Int {
        guard tokenizer.hasChatTemplate else { return 0 }
        do {
            let withGen = try applyChatTemplate(messages: messages, addGenerationPrompt: true)
            let withoutGen = try applyChatTemplate(messages: messages, addGenerationPrompt: false)
            return max(withGen.count - withoutGen.count, 0)
        } catch {
            return 0
        }
    }

    // MARK: - Inference

    /// Run the model forward pass (tokens in, logits out).
    public func forward(_ tokens: MLXArray, cache: [VMLXKVCache]?) -> MLXArray {
        nativeModel(tokens, cache: cache)
    }

    /// Create fresh caches for inference.
    public func newCache() -> [VMLXKVCache] {
        nativeModel.newCache()
    }
}
