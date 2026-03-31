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

/// Transforms GPT-OSS `<|channel|>analysis<|message|>` / `<|channel|>reply<|message|>`
/// into standard `<think>` / `</think>` tags that StreamingDeltaProcessor handles.
/// Accumulates tokens until complete tags can be detected and replaced.
@MainActor
final class ChannelTagMiddleware: StreamingMiddleware {
    private var buffer = ""

    private static let analysisTag = "<|channel|>analysis<|message|>"
    private static let replyTag = "<|channel|>reply<|message|>"

    func process(_ delta: String) -> String {
        buffer += delta

        var output = ""
        while !buffer.isEmpty {
            // Try to match complete tags
            if let range = buffer.range(of: Self.analysisTag) {
                output += buffer[..<range.lowerBound]
                output += "<think>"
                buffer = String(buffer[range.upperBound...])
                continue
            }
            if let range = buffer.range(of: Self.replyTag) {
                output += buffer[..<range.lowerBound]
                output += "</think>"
                buffer = String(buffer[range.upperBound...])
                continue
            }

            // Check if the buffer ENDS with a partial that could become either tag.
            // Test progressively shorter suffixes of the buffer against tag prefixes.
            var partialLen = 0
            let maxCheck = min(buffer.count, max(Self.analysisTag.count, Self.replyTag.count) - 1)
            for len in stride(from: maxCheck, through: 1, by: -1) {
                let suffix = String(buffer.suffix(len))
                if Self.analysisTag.hasPrefix(suffix) || Self.replyTag.hasPrefix(suffix) {
                    partialLen = len
                    break
                }
            }

            if partialLen > 0 {
                // Emit everything before the partial, keep partial buffered
                let emitEnd = buffer.index(buffer.endIndex, offsetBy: -partialLen)
                output += buffer[..<emitEnd]
                buffer = String(buffer[emitEnd...])
                break
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

        // GPT-OSS: transform <|channel|>analysis/reply tags to <think>/</ think>
        if !thinkingDisabled && (id.contains("gpt-oss") || id.contains("gpt_oss")) {
            return ChannelTagMiddleware()
        }

        // PrependThinkTagMiddleware: for models that output </think> but NOT <think>
        let needsPrependThink =
            !thinkingDisabled
            && (id.contains("glm") && id.contains("flash"))

        return needsPrependThink ? PrependThinkTagMiddleware() : nil
    }

    /// Matches parameter-count tokens like "4b" while ignoring
    /// quantization suffixes like "4bit" that share a prefix.
    private static func hasParamSize(_ id: String, anyOf sizes: String...) -> Bool {
        sizes.contains { id.range(of: "\($0)(?!it)", options: .regularExpression) != nil }
    }
}
