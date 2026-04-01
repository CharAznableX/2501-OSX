import Foundation

/// Tool call format used by a model family.
public enum ToolCallFormat: String, Sendable {
    case qwen = "qwen"
    case llama = "llama"
    case mistral = "mistral"
    case deepseek = "deepseek"
    case hermes = "hermes"
    case functionary = "functionary"
    case granite = "granite"
    case glm = "glm"
    case minimax = "minimax"
    case nemotron = "nemotron"
    case xlam = "xlam"
    case moonshot = "moonshot"
    case stepfun = "stepfun"
    case generic = "generic"
    case none = "none"
}

/// Reasoning/thinking format used by a model.
public enum ReasoningFormat: String, Sendable {
    case qwen3 = "qwen3"          // <think>...</think>
    case deepseekR1 = "deepseek_r1"
    case gptoss = "gptoss"
    case mistral = "mistral"
    case none = "none"
}

/// Configuration for a specific model family.
public struct ModelFamilyConfig: Sendable {
    /// Model family name (e.g., "qwen3", "llama4", "nemotron").
    public let family: String

    /// config.json `model_type` values that map to this family.
    /// Primary lookup key — no model name substring matching needed.
    public let modelTypes: [String]

    /// Tool call format for this model family.
    public let toolCallFormat: ToolCallFormat

    /// Reasoning format (if model supports thinking).
    public let reasoningFormat: ReasoningFormat

    /// Whether the model supports vision/multimodal input.
    public let supportsVision: Bool

    /// Whether this is a hybrid SSM model.
    public let isHybrid: Bool

    /// Hybrid layer pattern (e.g., "MMM*MMM*" for Nemotron-H). nil = all attention.
    public let hybridPattern: String?

    /// Default context window size.
    public let defaultContextWindow: Int

    /// Whether the chat template natively injects <think> when thinking is enabled.
    public let thinkInTemplate: Bool

    /// Whether TurboQuant is recommended for this model.
    public let recommendTQ: Bool

    /// Stop tokens specific to this model family.
    public let defaultStopTokens: [String]

    public init(
        family: String,
        modelTypes: [String] = [],
        toolCallFormat: ToolCallFormat = .none,
        reasoningFormat: ReasoningFormat = .none,
        supportsVision: Bool = false,
        isHybrid: Bool = false,
        hybridPattern: String? = nil,
        thinkInTemplate: Bool = false,
        defaultContextWindow: Int = 8192,
        recommendTQ: Bool = true,
        defaultStopTokens: [String] = []
    ) {
        self.family = family
        self.modelTypes = modelTypes
        self.toolCallFormat = toolCallFormat
        self.reasoningFormat = reasoningFormat
        self.supportsVision = supportsVision
        self.isHybrid = isHybrid
        self.hybridPattern = hybridPattern
        self.thinkInTemplate = thinkInTemplate
        self.defaultContextWindow = defaultContextWindow
        self.recommendTQ = recommendTQ
        self.defaultStopTokens = defaultStopTokens
    }
}

/// Registry of model family configurations.
/// Auto-detects model family from name and returns appropriate config.
public struct ModelConfigRegistry: Sendable {

    /// All registered model families. Keyed by config.json `model_type`.
    public static let configs: [ModelFamilyConfig] = [
        // Qwen family
        ModelFamilyConfig(family: "qwen3", modelTypes: ["qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"],
                         toolCallFormat: .qwen, reasoningFormat: .qwen3, thinkInTemplate: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>", "<|im_end|>"]),
        ModelFamilyConfig(family: "qwen2.5-vl", modelTypes: ["qwen2_vl", "qwen2_5_vl"],
                         toolCallFormat: .qwen, supportsVision: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>", "<|im_end|>"]),
        ModelFamilyConfig(family: "qwen2.5", modelTypes: ["qwen2", "qwen2_moe"],
                         toolCallFormat: .qwen,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>", "<|im_end|>"]),
        ModelFamilyConfig(family: "qwq", modelTypes: ["qwq"],
                         toolCallFormat: .qwen, reasoningFormat: .qwen3, thinkInTemplate: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>", "<|im_end|>"]),

        // Llama family
        ModelFamilyConfig(family: "llama", modelTypes: ["llama", "llama4", "llama4_text"],
                         toolCallFormat: .llama,
                         defaultContextWindow: 131072, defaultStopTokens: ["<|eot_id|>"]),

        // Mistral/Mixtral
        ModelFamilyConfig(family: "mistral", modelTypes: ["mistral", "mixtral"],
                         toolCallFormat: .mistral, reasoningFormat: .mistral,
                         defaultContextWindow: 32768, defaultStopTokens: ["</s>"]),
        ModelFamilyConfig(family: "mistral4", modelTypes: ["mistral3", "mistral4"],
                         toolCallFormat: .mistral, reasoningFormat: .mistral,
                         defaultContextWindow: 131072, defaultStopTokens: ["</s>"]),
        ModelFamilyConfig(family: "codestral", modelTypes: ["codestral"],
                         toolCallFormat: .mistral,
                         defaultContextWindow: 32768, defaultStopTokens: ["</s>"]),
        ModelFamilyConfig(family: "pixtral", modelTypes: ["pixtral"],
                         toolCallFormat: .mistral, supportsVision: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["</s>"]),

        // DeepSeek
        ModelFamilyConfig(family: "deepseek", modelTypes: ["deepseek_v3", "deepseek_v2"],
                         toolCallFormat: .deepseek,
                         defaultContextWindow: 65536, defaultStopTokens: ["<|end\u{2581}of\u{2581}sentence|>"]),

        // Nemotron
        ModelFamilyConfig(family: "nemotron", modelTypes: ["nemotron", "nemotron_h"],
                         toolCallFormat: .nemotron,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>"]),

        // Gemma
        ModelFamilyConfig(family: "gemma", modelTypes: ["gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n"],
                         toolCallFormat: .generic,
                         defaultContextWindow: 32768, defaultStopTokens: ["<end_of_turn>"]),

        // InternVL
        ModelFamilyConfig(family: "internvl", modelTypes: ["internvl", "internlm2"],
                         toolCallFormat: .generic, supportsVision: true,
                         defaultContextWindow: 32768),

        // Phi
        ModelFamilyConfig(family: "phi", modelTypes: ["phi3", "phi3v", "phi4", "phi4mm", "phi4_reasoning"],
                         toolCallFormat: .generic,
                         defaultContextWindow: 16384, defaultStopTokens: ["<|endoftext|>"]),

        // GPT-OSS
        ModelFamilyConfig(family: "gpt-oss", modelTypes: ["gpt_oss"],
                         toolCallFormat: .generic, reasoningFormat: .qwen3,
                         defaultContextWindow: 131072, defaultStopTokens: ["<|return|>"]),

        // GLM
        ModelFamilyConfig(family: "glm", modelTypes: ["chatglm", "glm4", "glm4_moe", "glm4_moe_lite"],
                         toolCallFormat: .glm,
                         defaultContextWindow: 32768),

        // MiniMax
        ModelFamilyConfig(family: "minimax", modelTypes: ["minimax_m2", "minimax_text_01"],
                         toolCallFormat: .minimax, reasoningFormat: .qwen3, thinkInTemplate: true,
                         defaultContextWindow: 32768),

        // Jamba (hybrid SSM)
        ModelFamilyConfig(family: "jamba", modelTypes: ["jamba"],
                         toolCallFormat: .generic, isHybrid: true,
                         defaultContextWindow: 262144),

        // Hermes
        ModelFamilyConfig(family: "hermes", modelTypes: ["hermes"],
                         toolCallFormat: .hermes,
                         defaultContextWindow: 8192, defaultStopTokens: ["<|im_end|>"]),

        // Granite
        ModelFamilyConfig(family: "granite", modelTypes: ["granite"],
                         toolCallFormat: .granite,
                         defaultContextWindow: 8192),

        // Cohere
        ModelFamilyConfig(family: "cohere", modelTypes: ["cohere", "cohere2"],
                         toolCallFormat: .generic,
                         defaultContextWindow: 131072),

        // Others
        ModelFamilyConfig(family: "starcoder", modelTypes: ["starcoder2"],
                         toolCallFormat: .generic, defaultContextWindow: 16384),
        ModelFamilyConfig(family: "olmo", modelTypes: ["olmo", "olmo2"],
                         toolCallFormat: .generic, defaultContextWindow: 8192),
        ModelFamilyConfig(family: "exaone", modelTypes: ["exaone"],
                         toolCallFormat: .generic, defaultContextWindow: 32768),
        ModelFamilyConfig(family: "stablelm", modelTypes: ["stablelm"],
                         toolCallFormat: .generic, defaultContextWindow: 8192),
    ]

    /// Lookup table built from configs: model_type → config.
    /// Built once at app start, used for O(1) lookups.
    private static let modelTypeMap: [String: ModelFamilyConfig] = {
        var map = [String: ModelFamilyConfig]()
        for config in configs {
            for mt in config.modelTypes {
                map[mt] = config
            }
        }
        return map
    }()

    /// Detect model family from config.json `model_type` field.
    /// This is the PRIMARY lookup — no model name matching.
    public static func configForModelType(_ modelType: String) -> ModelFamilyConfig? {
        modelTypeMap[modelType]
    }

    /// Get config for a model name (LEGACY — prefer configForModelType).
    /// Used only when model_type is not available (e.g., display name contexts).
    public static func configFor(modelName: String) -> ModelFamilyConfig {
        // First try exact model_type lookup
        let normalized = modelName.lowercased()
            .replacingOccurrences(of: "_", with: "-")
            .replacingOccurrences(of: " ", with: "-")
        if let config = modelTypeMap[modelName] ?? modelTypeMap[normalized] {
            return config
        }
        // Fallback: generic config
        return ModelFamilyConfig(family: "generic", toolCallFormat: .generic, defaultContextWindow: 8192)
    }
}
