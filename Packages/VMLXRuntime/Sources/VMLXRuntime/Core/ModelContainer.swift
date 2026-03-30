import Foundation
import MLX
import Tokenizers

/// Container wrapping a loaded model with runtime configuration.
/// This is what gets passed around during inference.
public final class ModelContainer: @unchecked Sendable {

    /// The loaded model (weights + tokenizer + config).
    public let model: LoadedModel

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

    public init(model: LoadedModel) {
        self.model = model
        self.name = model.detected.name
        self.eosTokenIds = model.eosTokenIds
        self.familyConfig = ModelConfigRegistry.configFor(modelName: model.detected.name)

        // Build TQ config from JANG settings if applicable
        if model.detected.isJang,
           JangLoader.isJangModel(at: model.detected.modelPath),
           let jangConfig = try? JangLoader.loadConfig(at: model.detected.modelPath) {

            // Build layer pattern from detected hybrid info
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

            self.turboQuantConfig = JangLoader.buildTQConfig(
                from: jangConfig,
                layerPattern: detectedLayerPattern,
                kvLoraRank: model.detected.kvLoraRank,
                qkNopeHeadDim: model.detected.qkNopeHeadDim,
                qkRopeHeadDim: model.detected.qkRopeHeadDim
            )
            self.layerPattern = detectedLayerPattern
        } else {
            self.turboQuantConfig = nil

            // Still parse layer pattern for non-JANG hybrid models
            if let patternStr = model.detected.hybridOverridePattern {
                self.layerPattern = parseHybridPattern(patternStr)
            } else if let layerTypeStrs = model.detected.layerTypes {
                self.layerPattern = layerTypeStrs.map { str -> LayerType in
                    switch str.lowercased() {
                    case "attention", "attn": return .attention
                    case "ssm", "mamba", "recurrent": return .ssm
                    default: return .attention
                    }
                }
            } else {
                self.layerPattern = nil
            }
        }
    }

    // MARK: - Tokenization

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        model.tokenizer.encode(text: text)
    }

    /// Decode token IDs to text.
    public func decode(_ tokens: [Int]) -> String {
        model.tokenizer.decode(tokens: tokens)
    }

    /// Apply chat template to messages and encode.
    ///
    /// Converts `VMLXChatMessage` to swift-transformers `Message` format
    /// (`[String: any Sendable]` dictionaries) and delegates to the tokenizer.
    public func applyChatTemplate(
        messages: [VMLXChatMessage],
        addGenerationPrompt: Bool = true
    ) throws -> [Int] {
        // Convert VMLXChatMessage to swift-transformers Message format
        let chatMessages: [Message] = messages.map { msg in
            ["role": msg.role, "content": msg.textContent]
        }

        // Try using the tokenizer's chat template
        if model.tokenizer.hasChatTemplate {
            return try model.tokenizer.applyChatTemplate(
                messages: chatMessages,
                chatTemplate: nil,
                addGenerationPrompt: addGenerationPrompt,
                truncation: false,
                maxLength: nil,
                tools: nil
            )
        }

        // Fallback: simple concatenation
        let fullText = messages.map { msg in
            "\(msg.role): \(msg.textContent)"
        }.joined(separator: "\n")

        return encode(fullText)
    }

    /// Compute gen_prompt_len: difference between encoding with and without generation prompt.
    ///
    /// Used for SSM checkpointing in hybrid thinking models: the generation prompt
    /// (e.g., `<think>`) tokens contaminate SSM state, so we need to know where
    /// the stable boundary is for checkpointing.
    public func computeGenPromptLen(messages: [VMLXChatMessage]) -> Int {
        guard model.tokenizer.hasChatTemplate else { return 0 }

        do {
            let withGen = try applyChatTemplate(messages: messages, addGenerationPrompt: true)
            let withoutGen = try applyChatTemplate(messages: messages, addGenerationPrompt: false)
            return max(withGen.count - withoutGen.count, 0)
        } catch {
            return 0
        }
    }
}
