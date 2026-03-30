import Foundation

/// Tool call parser for DeepSeek V3 and R1 models.
///
/// Supports DeepSeek's tool call format with special unicode delimiter tokens:
/// ```
/// <|tool_calls_begin|>
/// <|tool_call_begin|>function<|tool_sep|>get_weather
/// ```json
/// {"city": "Paris"}
/// ```<|tool_call_end|>
/// <|tool_calls_end|>
/// ```
///
/// The actual DeepSeek tokens use fullwidth unicode characters but we normalize
/// both ASCII-bracket forms and the unicode forms for robustness.
public struct DeepSeekToolParser: ToolCallParser {

    public static var supportedModels: [String] { ["deepseek"] }

    private var buffer: String = ""
    private var state: State = .text

    private enum State {
        case text
        case potentialTag       // Might be start of tool calls
        case inToolCalls        // Between tool_calls_begin and tool_calls_end
    }

    // Support both unicode fullwidth and ASCII bracket forms
    // Unicode form: <｜tool▁calls▁begin｜>
    // ASCII form:   <|tool_calls_begin|>
    private static let toolCallsBeginTokens = [
        "<\u{FF5C}tool\u{2581}calls\u{2581}begin\u{FF5C}>",  // Unicode fullwidth
        "<|tool_calls_begin|>",                                 // ASCII normalized
    ]
    private static let toolCallsEndTokens = [
        "<\u{FF5C}tool\u{2581}calls\u{2581}end\u{FF5C}>",
        "<|tool_calls_end|>",
    ]
    private static let toolCallBeginTokens = [
        "<\u{FF5C}tool\u{2581}call\u{2581}begin\u{FF5C}>",
        "<|tool_call_begin|>",
    ]
    private static let toolCallEndTokens = [
        "<\u{FF5C}tool\u{2581}call\u{2581}end\u{FF5C}>",
        "<|tool_call_end|>",
    ]
    private static let toolSepTokens = [
        "<\u{FF5C}tool\u{2581}sep\u{FF5C}>",
        "<|tool_sep|>",
    ]

    public init() {}

    public mutating func processChunk(_ text: String) -> [ToolParserResult] {
        var results: [ToolParserResult] = []
        buffer += text

        while !buffer.isEmpty {
            switch state {
            case .text:
                // Look for tool_calls_begin marker
                if let (range, _) = _findToken(in: buffer, tokens: Self.toolCallsBeginTokens) {
                    let prefix = String(buffer[buffer.startIndex..<range.lowerBound])
                    if !prefix.isEmpty {
                        results.append(.text(prefix))
                    }
                    buffer = String(buffer[range.upperBound...])
                    state = .inToolCalls
                    continue
                }

                // Check for potential partial marker at the end
                if _hasPotentialTokenStart(buffer) {
                    state = .potentialTag
                    results.append(.buffered)
                    return results
                }

                results.append(.text(buffer))
                buffer = ""

            case .potentialTag:
                if _findToken(in: buffer, tokens: Self.toolCallsBeginTokens) != nil {
                    state = .text
                    continue
                } else if _isPotentialTokenPrefix(buffer) {
                    results.append(.buffered)
                    return results
                } else {
                    state = .text
                    continue
                }

            case .inToolCalls:
                // Look for tool_calls_end marker
                if let (endRange, _) = _findToken(in: buffer, tokens: Self.toolCallsEndTokens) {
                    let toolContent = String(buffer[buffer.startIndex..<endRange.lowerBound])
                    let calls = _parseToolCallsBlock(toolContent)
                    for call in calls {
                        results.append(.toolCall(call))
                    }
                    buffer = String(buffer[endRange.upperBound...])
                    state = .text
                } else {
                    // Check if we have complete individual tool calls even without the outer end
                    if let (endRange, _) = _findToken(in: buffer, tokens: Self.toolCallEndTokens) {
                        // We have at least one complete tool call, but might be more coming
                        // Keep buffering until we see tool_calls_end
                        results.append(.buffered)
                        return results
                    } else {
                        results.append(.buffered)
                        return results
                    }
                }
            }
        }

        return results
    }

    public mutating func finalize() -> [ParsedToolCall] {
        guard !buffer.isEmpty else { return [] }

        let text = buffer
        buffer = ""
        state = .text

        return _parseToolCallsBlock(text)
    }

    public mutating func reset() {
        buffer = ""
        state = .text
    }

    // MARK: - Private

    private func _findToken(in text: String, tokens: [String]) -> (Range<String.Index>, String)? {
        for token in tokens {
            if let range = text.range(of: token) {
                return (range, token)
            }
        }
        return nil
    }

    private func _hasPotentialTokenStart(_ text: String) -> Bool {
        // Check if buffer ends with a prefix of any begin token
        let allTokens = Self.toolCallsBeginTokens
        for token in allTokens {
            for i in 1..<token.count {
                let prefix = String(token.prefix(i))
                if text.hasSuffix(prefix) { return true }
            }
        }
        return false
    }

    private func _isPotentialTokenPrefix(_ text: String) -> Bool {
        let allTokens = Self.toolCallsBeginTokens
        for token in allTokens {
            if token.hasPrefix(text) { return true }
        }
        return false
    }

    /// Parse the content between tool_calls_begin and tool_calls_end.
    /// Contains one or more individual tool calls in DeepSeek format.
    private func _parseToolCallsBlock(_ text: String) -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []
        var remaining = text

        // Find each tool_call_begin ... tool_call_end pair
        while let (beginRange, _) = _findToken(in: remaining, tokens: Self.toolCallBeginTokens) {
            guard let (endRange, _) = _findToken(
                in: String(remaining[beginRange.upperBound...]),
                tokens: Self.toolCallEndTokens
            ) else {
                break
            }

            let afterBegin = String(remaining[beginRange.upperBound...])
            let callContent = String(afterBegin[afterBegin.startIndex..<endRange.lowerBound])

            if let call = _parseSingleToolCall(callContent) {
                calls.append(call)
            }

            // Move past this tool call end
            remaining = String(afterBegin[endRange.upperBound...])
        }

        return calls
    }

    /// Parse a single tool call content:
    /// `function<|tool_sep|>get_weather\n```json\n{"city": "Paris"}\n````
    private func _parseSingleToolCall(_ content: String) -> ParsedToolCall? {
        // Find tool_sep to split type and name
        guard let (sepRange, _) = _findToken(in: content, tokens: Self.toolSepTokens) else {
            // No separator, try simpler format: just name\n```json\n{...}\n```
            return _parseSimpleToolCall(content)
        }

        let afterSep = String(content[sepRange.upperBound...])

        // Name is on the first line after the separator
        let lines = afterSep.components(separatedBy: "\n")
        let funcName = lines[0].trimmingCharacters(in: .whitespacesAndNewlines)

        guard !funcName.isEmpty else { return nil }

        // Extract JSON from ```json ... ``` block
        if let jsonStr = _extractJSONFromCodeBlock(afterSep) {
            return ParsedToolCall(name: funcName, argumentsJSON: jsonStr)
        }

        // Fallback: try to find raw JSON after the name
        let afterName = lines.dropFirst().joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = afterName.data(using: .utf8),
           let _ = try? JSONSerialization.jsonObject(with: data) {
            return ParsedToolCall(name: funcName, argumentsJSON: afterName)
        }

        return ParsedToolCall(name: funcName, argumentsJSON: "{}")
    }

    private func _parseSimpleToolCall(_ content: String) -> ParsedToolCall? {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        let lines = trimmed.components(separatedBy: "\n")
        guard !lines.isEmpty else { return nil }

        let funcName = lines[0].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !funcName.isEmpty else { return nil }

        let rest = lines.dropFirst().joined(separator: "\n")
        if let jsonStr = _extractJSONFromCodeBlock(rest) {
            return ParsedToolCall(name: funcName, argumentsJSON: jsonStr)
        }

        return nil
    }

    /// Extract JSON from a markdown code block: ```json\n{...}\n```
    private func _extractJSONFromCodeBlock(_ text: String) -> String? {
        // Find ```json marker
        guard let jsonStart = text.range(of: "```json") else { return nil }
        let afterMarker = String(text[jsonStart.upperBound...])

        // Find closing ```
        guard let jsonEnd = afterMarker.range(of: "```") else { return nil }
        let jsonStr = String(afterMarker[afterMarker.startIndex..<jsonEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)

        // Validate it's valid JSON
        guard let data = jsonStr.data(using: .utf8),
              let _ = try? JSONSerialization.jsonObject(with: data) else {
            return jsonStr  // Return raw even if invalid
        }

        return jsonStr
    }
}
