import Foundation
import MLX

// MARK: - Chat Message Types (OpenAI-compatible)

/// A chat message in OpenAI format.
public struct VMLXChatMessage: Sendable, Codable {
    public let role: String           // "system", "user", "assistant", "tool"
    public let content: String?       // Text content
    public let contentParts: [VMLXContentPart]?  // Multimodal content
    public let toolCalls: [VMLXToolCall]?         // Assistant's tool invocations
    public let toolCallId: String?    // For role=="tool" responses

    public init(role: String, content: String) {
        self.role = role
        self.content = content
        self.contentParts = nil
        self.toolCalls = nil
        self.toolCallId = nil
    }

    public init(role: String, contentParts: [VMLXContentPart]) {
        self.role = role
        self.content = nil
        self.contentParts = contentParts
        self.toolCalls = nil
        self.toolCallId = nil
    }

    public init(role: String, content: String? = nil, contentParts: [VMLXContentPart]? = nil,
                toolCalls: [VMLXToolCall]? = nil, toolCallId: String? = nil) {
        self.role = role
        self.content = content
        self.contentParts = contentParts
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
    }

    /// Extract text from content or contentParts.
    public var textContent: String {
        if let content = content { return content }
        if let parts = contentParts {
            return parts.compactMap { part -> String? in
                if case .text(let text) = part { return text }
                return nil
            }.joined()
        }
        return ""
    }

    /// Extract image URLs/data from contentParts.
    public var imageURLs: [String] {
        contentParts?.compactMap { part -> String? in
            if case .imageURL(let url, _) = part { return url }
            return nil
        } ?? []
    }

    /// Whether this message contains images.
    public var hasImages: Bool {
        !imageURLs.isEmpty
    }
}

/// Content part in a multimodal message.
public enum VMLXContentPart: Sendable, Codable {
    case text(String)
    case imageURL(url: String, detail: String?)

    // Codable conformance
    enum CodingKeys: String, CodingKey {
        case type, text, image_url
    }

    struct ImageURLData: Codable {
        let url: String
        let detail: String?
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(text)
        case "image_url":
            let imgData = try container.decode(ImageURLData.self, forKey: .image_url)
            self = .imageURL(url: imgData.url, detail: imgData.detail)
        default:
            self = .text("")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .imageURL(let url, let detail):
            try container.encode("image_url", forKey: .type)
            try container.encode(ImageURLData(url: url, detail: detail), forKey: .image_url)
        }
    }
}

/// Tool call from assistant.
public struct VMLXToolCall: Sendable, Codable {
    public let id: String
    public let type: String   // "function"
    public let function: VMLXFunctionCall

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.type = "function"
        self.function = VMLXFunctionCall(name: name, arguments: arguments)
    }
}

/// Function call details.
public struct VMLXFunctionCall: Sendable, Codable {
    public let name: String
    public let arguments: String  // JSON string
}

/// Tool definition for function calling.
public struct VMLXToolDefinition: Sendable, Codable {
    public let type: String   // "function"
    public let function: VMLXFunctionSpec

    public init(name: String, description: String?, parameters: [String: Any]? = nil) {
        self.type = "function"
        self.function = VMLXFunctionSpec(name: name, description: description)
    }
}

/// Function specification.
public struct VMLXFunctionSpec: Sendable, Codable {
    public let name: String
    public let description: String?
}

// MARK: - Chat Completion Request

/// OpenAI-compatible chat completion request.
public struct VMLXChatCompletionRequest: Sendable {
    public let messages: [VMLXChatMessage]
    public let model: String?
    public let temperature: Float?
    public let maxTokens: Int?
    public let topP: Float?
    public let repetitionPenalty: Float?
    public let stop: [String]?
    public let stream: Bool
    public let tools: [VMLXToolDefinition]?
    public let toolChoice: String?   // "auto", "none", or function name
    public let enableThinking: Bool?
    public let reasoningEffort: String?
    public let sessionId: String?     // KV cache reuse across turns
    public let cacheHint: String?     // Explicit prefix cache key
    public let toolParserOverride: String?
    public let reasoningParserOverride: String?

    public init(
        messages: [VMLXChatMessage],
        model: String? = nil,
        temperature: Float? = nil,
        maxTokens: Int? = nil,
        topP: Float? = nil,
        repetitionPenalty: Float? = nil,
        stop: [String]? = nil,
        stream: Bool = true,
        tools: [VMLXToolDefinition]? = nil,
        toolChoice: String? = nil,
        enableThinking: Bool? = nil,
        reasoningEffort: String? = nil,
        sessionId: String? = nil,
        cacheHint: String? = nil,
        toolParserOverride: String? = nil,
        reasoningParserOverride: String? = nil
    ) {
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.stop = stop
        self.stream = stream
        self.tools = tools
        self.toolChoice = toolChoice
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.sessionId = sessionId
        self.cacheHint = cacheHint
        self.toolParserOverride = toolParserOverride
        self.reasoningParserOverride = reasoningParserOverride
    }

    /// Convert to SamplingParams.
    public func toSamplingParams() -> SamplingParams {
        SamplingParams(
            maxTokens: maxTokens ?? 2048,
            temperature: temperature ?? 0.7,
            topP: topP ?? 0.9,
            repetitionPenalty: repetitionPenalty ?? 1.0,
            stop: stop ?? []
        )
    }

    /// Whether this request includes multimodal content.
    public var isMultimodal: Bool {
        messages.contains { $0.hasImages }
    }
}

// MARK: - Chat Completion Response (SSE streaming)

/// Streaming chat completion chunk.
public struct VMLXChatCompletionChunk: Sendable, Codable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [VMLXStreamChoice]

    public init(id: String, model: String, delta: VMLXDelta, finishReason: String? = nil) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = [VMLXStreamChoice(
            index: 0,
            delta: delta,
            finishReason: finishReason
        )]
    }
}

public struct VMLXStreamChoice: Sendable, Codable {
    public let index: Int
    public let delta: VMLXDelta
    public let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index, delta
        case finishReason = "finish_reason"
    }
}

public struct VMLXDelta: Sendable, Codable {
    public let role: String?
    public let content: String?
    public let reasoningContent: String?
    public let toolCalls: [VMLXToolCall]?

    enum CodingKeys: String, CodingKey {
        case role, content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }

    public init(role: String? = nil, content: String? = nil,
                reasoningContent: String? = nil, toolCalls: [VMLXToolCall]? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }
}
