import Foundation
import MLX

// MARK: - Request / Response Types

/// OpenAI-compatible embeddings request.
public struct EmbeddingsRequest: Sendable, Codable {
    public let model: String?
    public let input: EmbeddingsInput
    public let dimensions: Int?          // Optional dimension truncation
    public let encodingFormat: String?   // "float" or "base64"

    enum CodingKeys: String, CodingKey {
        case model, input, dimensions
        case encodingFormat = "encoding_format"
    }
}

public enum EmbeddingsInput: Sendable, Codable {
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

    public var texts: [String] {
        switch self {
        case .single(let s): return [s]
        case .batch(let b): return b
        }
    }
}

/// OpenAI-compatible embeddings response.
public struct EmbeddingsResponse: Sendable, Codable {
    public let object: String
    public let data: [EmbeddingData]
    public let model: String
    public let usage: EmbeddingsUsage

    public init(embeddings: [[Float]], model: String, promptTokens: Int) {
        self.object = "list"
        self.data = embeddings.enumerated().map { i, emb in
            EmbeddingData(index: i, embedding: emb)
        }
        self.model = model
        self.usage = EmbeddingsUsage(prompt_tokens: promptTokens, total_tokens: promptTokens)
    }
}

public struct EmbeddingData: Sendable, Codable {
    public let object: String
    public let index: Int
    public let embedding: [Float]

    public init(index: Int, embedding: [Float]) {
        self.object = "embedding"
        self.index = index
        self.embedding = embedding
    }
}

public struct EmbeddingsUsage: Sendable, Codable {
    public let prompt_tokens: Int
    public let total_tokens: Int
}
