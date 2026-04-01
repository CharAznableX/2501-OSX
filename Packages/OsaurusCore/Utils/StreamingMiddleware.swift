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

/// Buffers early deltas for models that emit `</think>` without `<think>`.
/// Only prepends `<think>` if a `</think>` is detected in the first N tokens,
/// confirming the model is actually reasoning. Otherwise, flushes buffered
/// content as-is (no false thinking box).
@MainActor
final class PrependThinkTagMiddleware: StreamingMiddleware {
    private var state: State = .buffering
    private var buffer: String = ""
    private var deltaCount = 0
    private static let maxBufferDeltas = 20  // Check first ~20 tokens

    private enum State {
        case buffering   // Accumulating early deltas to check for </think>
        case confirmed   // </think> found — already prepended <think>
        case passthrough // No </think> detected — no thinking, pass through
    }

    func process(_ delta: String) -> String {
        switch state {
        case .confirmed, .passthrough:
            return delta

        case .buffering:
            deltaCount += 1
            buffer += delta

            // Check if </think> appeared — confirms model is reasoning
            if buffer.contains("</think>") {
                state = .confirmed
                let result = "<think>" + buffer
                buffer = ""
                return result
            }

            // If we've buffered enough without seeing </think>, give up
            if deltaCount >= Self.maxBufferDeltas {
                state = .passthrough
                let result = buffer
                buffer = ""
                return result
            }

            // Still buffering — suppress output for now
            return ""
        }
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

        // PrependThinkTagMiddleware: for models that output </think> but NOT <think>.
        // These models have thinkInTemplate=true — the chat template injects <think>
        // in the generation prompt, so the model's output starts INSIDE thinking.
        // The MLXService path doesn't inject <think>, so the middleware does it.
        // (VMLX path handles this via VMLXRuntimeActor's thinkInTemplate flag.)
        let needsPrependThink =
            !thinkingDisabled
            && ((id.contains("glm") && id.contains("flash"))
                || id.contains("gpt-oss") || id.contains("gpt_oss") || id.contains("gptoss"))

        return needsPrependThink ? PrependThinkTagMiddleware() : nil
    }

    /// Matches parameter-count tokens like "4b" while ignoring
    /// quantization suffixes like "4bit" that share a prefix.
    private static func hasParamSize(_ id: String, anyOf sizes: String...) -> Bool {
        sizes.contains { id.range(of: "\($0)(?!it)", options: .regularExpression) != nil }
    }
}
