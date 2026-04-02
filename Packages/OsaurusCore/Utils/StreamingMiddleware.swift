//
//  StreamingMiddleware.swift
//  osaurus
//
//  Transforms raw streaming deltas before they reach StreamingDeltaProcessor's
//  tag parser. Model-specific streaming behavior lives here, keeping the
//  processor itself model-agnostic.
//

/// Transforms raw streaming deltas before they reach the tag parser.
/// Stateful — create a new instance per streaming session.
@MainActor
protocol StreamingMiddleware: AnyObject {
    func process(_ delta: String) -> String
}

// MARK: - Middleware Implementations

/// Prepends `<think>` on the first non-empty delta for models that output
/// thinking content without the opening tag (thinkInTemplate models).
/// Streams everything through immediately — no buffering delay.
/// If the model isn't thinking, `<think>` goes through but StreamingDeltaProcessor
/// will see it as content and the thinking box opens (acceptable trade-off
/// vs buffering 20+ tokens and delaying all output).
@MainActor
final class PrependThinkTagMiddleware: StreamingMiddleware {
    private var hasFired = false

    func process(_ delta: String) -> String {
        guard !hasFired else { return delta }
        guard !delta.isEmpty else { return delta }
        hasFired = true
        // Don't double-inject if VMLX already added <think>
        if delta.contains("<think>") { return delta }
        return "<think>" + delta
    }
}

// MARK: - Channel Tag Middleware (GPT-OSS)

/// Transforms GPT-OSS channel tags into standard `<think>`/`</think>` tags.
/// Handles ALL channel types: analysis (thinking), reply/final/assistant (content).
/// Also strips leaked special tokens like `<|end|>`, `<|start|>`.
@MainActor
final class ChannelTagMiddleware: StreamingMiddleware {
    private var buffer = ""
    private var insideAnalysis = false

    // Tags that START thinking
    private static let thinkStartTags = [
        "<|channel|>analysis<|message|>",
    ]

    // Tags that END thinking and start content
    private static let thinkEndTags = [
        "<|channel|>reply<|message|>",
        "<|channel|>final<|message|>",
        "<|channel|>assistant<|message|>",
    ]

    // Special tokens to strip from output
    private static let stripTokens = [
        "<|end|>", "<|start|>", "<|endoftext|>",
    ]

    func process(_ delta: String) -> String {
        buffer += delta

        // Strip special tokens
        for token in Self.stripTokens {
            if buffer.contains(token) {
                buffer = buffer.replacingOccurrences(of: token, with: "")
            }
        }

        var output = ""
        while !buffer.isEmpty {
            // Try to match start-of-thinking tags
            var matched = false
            for tag in Self.thinkStartTags {
                if let range = buffer.range(of: tag) {
                    output += buffer[..<range.lowerBound]
                    output += "<think>"
                    buffer = String(buffer[range.upperBound...])
                    insideAnalysis = true
                    matched = true
                    break
                }
            }
            if matched { continue }

            // Try to match end-of-thinking tags
            for tag in Self.thinkEndTags {
                if let range = buffer.range(of: tag) {
                    output += buffer[..<range.lowerBound]
                    output += "</think>"
                    buffer = String(buffer[range.upperBound...])
                    insideAnalysis = false
                    matched = true
                    break
                }
            }
            if matched { continue }

            // Check for partial channel tag at end of buffer
            if buffer.contains("<|") {
                // Find the last `<|` that could be the start of a channel tag
                if let lastPipe = buffer.range(of: "<|", options: .backwards) {
                    let suffix = String(buffer[lastPipe.lowerBound...])
                    // Check if any tag starts with this suffix
                    let allTags = Self.thinkStartTags + Self.thinkEndTags + Self.stripTokens
                    if allTags.contains(where: { $0.hasPrefix(suffix) }) && suffix.count < 40 {
                        output += buffer[..<lastPipe.lowerBound]
                        buffer = suffix
                        break
                    }
                }
            }

            // No tag or partial — emit everything
            output += buffer
            buffer = ""
        }

        return output
    }
}

/// Transforms Mistral-style `[THINK]...[/THINK]` markers into standard
/// `<think>...</think>` tags consumed by StreamingDeltaProcessor.
@MainActor
final class MistralTagMiddleware: StreamingMiddleware {
    private var buffer = ""

    private static let openTag = "[THINK]"
    private static let closeTag = "[/THINK]"

    func process(_ delta: String) -> String {
        buffer += delta

        var output = ""
        while !buffer.isEmpty {
            if let range = buffer.range(of: Self.openTag) {
                output += buffer[..<range.lowerBound]
                output += "<think>"
                buffer = String(buffer[range.upperBound...])
                continue
            }

            if let range = buffer.range(of: Self.closeTag) {
                output += buffer[..<range.lowerBound]
                output += "</think>"
                buffer = String(buffer[range.upperBound...])
                continue
            }

            let partials = [Self.closeTag, Self.openTag]
            if let partial = partials.first(where: { tag in
                (1..<tag.count).reversed().contains { len in
                    buffer.hasSuffix(String(tag.prefix(len)))
                }
            }) {
                for len in (1..<partial.count).reversed() {
                    let prefix = String(partial.prefix(len))
                    if buffer.hasSuffix(prefix) {
                        output += buffer.dropLast(len)
                        buffer = prefix
                        return output
                    }
                }
            }

            output += buffer
            buffer = ""
        }

        return output
    }
}

// MARK: - Resolver

enum StreamingMiddlewareResolver {
    @MainActor
    static func resolve(
        for modelId: String,
        modelOptions: [String: ModelOptionValue] = [:],
        globalReasoningParserOverride: String? = nil
    ) -> StreamingMiddleware? {
        let thinkingDisabled = modelOptions["disableThinking"]?.boolValue == true
        let id = modelId.lowercased()
        let effectiveReasoningParser = LocalParserOptions.resolveReasoningOverride(
            perModel: modelOptions["reasoningParser"]?.stringValue,
            global: globalReasoningParserOverride
        )

        switch effectiveReasoningParser {
        case "none":
            return nil
        case "gptoss":
            return ChannelTagMiddleware()
        case "mistral":
            return MistralTagMiddleware()
        case "think":
            return thinkingDisabled ? nil : PrependThinkTagMiddleware()
        default:
            break
        }

        if id.contains("gptoss") || id.contains("harmony") {
            return ChannelTagMiddleware()
        }

        if id.contains("mistral") && (id.contains("4") || id.contains("large")) {
            return MistralTagMiddleware()
        }

        let usesThinkTags =
            id.contains("qwen3")
            || id.contains("qwen2.5")
            || id.contains("deepseek")
            || id.contains("qwq")
            || id.contains("glm")
            || id.contains("minimax")
            || id.contains("phi-4")

        if !thinkingDisabled && usesThinkTags {
            return PrependThinkTagMiddleware()
        }

        return nil
    }

    /// Matches parameter-count tokens like "4b" while ignoring
    /// quantization suffixes like "4bit" that share a prefix.
    private static func hasParamSize(_ id: String, anyOf sizes: String...) -> Bool {
        sizes.contains { id.range(of: "\($0)(?!it)", options: .regularExpression) != nil }
    }
}
