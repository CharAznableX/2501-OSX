import Foundation

// MARK: - Legacy Completions Request / Response

/// OpenAI legacy completions request (`/v1/completions`).
public struct CompletionsRequest: Sendable, Codable {
    public let model: String?
    public let prompt: CompletionsPrompt
    public let maxTokens: Int?
    public let temperature: Float?
    public let topP: Float?
    public let stop: [String]?
    public let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, stop, stream
        case maxTokens = "max_tokens"
        case topP = "top_p"
    }
}

public enum CompletionsPrompt: Sendable, Codable {
    case single(String)
    case batch([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .single(s)
        } else {
            self = .batch(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .single(let s): try container.encode(s)
        case .batch(let b): try container.encode(b)
        }
    }
}

/// OpenAI legacy completions response.
public struct CompletionsResponse: Sendable, Codable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [CompletionsChoice]
    public let usage: CompletionsUsage

    public init(id: String, model: String, text: String, finishReason: String, promptTokens: Int, completionTokens: Int) {
        self.id = id
        self.object = "text_completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = [CompletionsChoice(text: text, index: 0, finishReason: finishReason)]
        self.usage = CompletionsUsage(prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens)
    }
}

public struct CompletionsChoice: Sendable, Codable {
    public let text: String
    public let index: Int
    public let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case text, index
        case finishReason = "finish_reason"
    }
}

public struct CompletionsUsage: Sendable, Codable {
    public let prompt_tokens: Int
    public let completion_tokens: Int
    public let total_tokens: Int
}

// MARK: - Adapter

/// Translates between the legacy OpenAI `/v1/completions` format (text-in, text-out)
/// and the chat completions format used by VMLXRuntime.
public struct CompletionsAdapter {
    /// Convert a legacy completions request into a VMLXChatCompletionRequest.
    public static func toVMLXRequest(_ req: CompletionsRequest) -> VMLXChatCompletionRequest {
        let text: String
        switch req.prompt {
        case .single(let s): text = s
        case .batch(let b): text = b.joined(separator: "\n")
        }

        return VMLXChatCompletionRequest(
            messages: [VMLXChatMessage(role: "user", content: text)],
            model: req.model,
            temperature: req.temperature,
            maxTokens: req.maxTokens,
            topP: req.topP,
            stop: req.stop,
            stream: req.stream ?? false
        )
    }
}
