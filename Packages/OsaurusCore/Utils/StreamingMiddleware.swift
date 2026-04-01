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

// MARK: - Resolver

enum StreamingMiddlewareResolver {
    @MainActor
    static func resolve(
        for modelId: String,
        modelOptions: [String: ModelOptionValue] = [:]
    ) -> StreamingMiddleware? {
        let thinkingDisabled = modelOptions["disableThinking"]?.boolValue == true
        let id = modelId.lowercased()

        // PrependThinkTagMiddleware: for MLXService fallback models that output
        // </think> without <think>. VMLX models handle thinking at the engine level
        // via VMLXRuntimeActor's text buffer. Since all local models now try VMLX
        // first, this middleware only activates for the few that fall through to MLX.
        // No model name matching — the middleware is safe for any model (it only
        // prepends <think> on the first non-empty delta if thinking is enabled).
        if !thinkingDisabled {
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
