import Foundation

/// Tool call parser for Qwen models.
///
/// Supports two Qwen tool call formats:
/// - XML style: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
/// - Bracket style: `[Calling tool: func_name({"arg": "value"})]`
public struct QwenToolParser: ToolCallParser {

    public static var supportedModels: [String] { ["qwen", "qwq"] }

    private var buffer: String = ""
    private var state: State = .text

    private enum State {
        case text
        case potentialTag       // Accumulating what might be `<tool_call>`
        case inToolCall         // Between `<tool_call>` and `</tool_call>`
        case potentialEndTag    // Inside tool call, accumulating what might be `</tool_call>`
    }

    private static let openTag = "<tool_call>"
    private static let closeTag = "</tool_call>"

    public init() {}

    public mutating func processChunk(_ text: String) -> [ToolParserResult] {
        var results: [ToolParserResult] = []
        buffer += text

        while !buffer.isEmpty {
            switch state {
            case .text:
                // Look for start of `<tool_call>` tag
                if let tagRange = buffer.range(of: Self.openTag) {
                    // Emit any text before the tag
                    let prefix = String(buffer[buffer.startIndex..<tagRange.lowerBound])
                    if !prefix.isEmpty {
                        results.append(.text(prefix))
                    }
                    buffer = String(buffer[tagRange.upperBound...])
                    state = .inToolCall
                } else if buffer.hasSuffix("<") || _hasPotentialTagPrefix(buffer) {
                    // Might be start of `<tool_call>`, buffer it
                    state = .potentialTag
                    results.append(.buffered)
                    return results
                } else {
                    // Also check for bracket-style: [Calling tool: ...]
                    var temp = buffer
                    if let bracketResult = _tryExtractBracketStyle(&temp) {
                        buffer = temp
                        results.append(.toolCall(bracketResult))
                        continue
                    }
                    // No tag found, emit everything as text
                    results.append(.text(buffer))
                    buffer = ""
                }

            case .potentialTag:
                if buffer.hasPrefix(Self.openTag) || buffer.contains(Self.openTag) {
                    // Found the full tag, transition
                    state = .text
                    continue
                } else if Self.openTag.hasPrefix(buffer) {
                    // Still a valid prefix of `<tool_call>`, keep buffering
                    results.append(.buffered)
                    return results
                } else {
                    // Not a tag, emit as text
                    state = .text
                    continue
                }

            case .inToolCall:
                if let closeRange = buffer.range(of: Self.closeTag) {
                    // Extract JSON between tags
                    let jsonStr = String(buffer[buffer.startIndex..<closeRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                    buffer = String(buffer[closeRange.upperBound...])
                    state = .text

                    if let toolCall = _parseToolCallJSON(jsonStr) {
                        results.append(.toolCall(toolCall))
                    } else {
                        // Failed to parse, emit as text
                        results.append(.text(Self.openTag + jsonStr + Self.closeTag))
                    }
                } else if buffer.contains("</") {
                    // Might be partial close tag
                    state = .potentialEndTag
                    continue
                } else {
                    // Still accumulating tool call content
                    results.append(.buffered)
                    return results
                }

            case .potentialEndTag:
                if buffer.contains(Self.closeTag) {
                    state = .inToolCall
                    continue
                } else if _hasPotentialCloseTagSuffix(buffer) {
                    results.append(.buffered)
                    return results
                } else {
                    // Not a close tag, keep accumulating
                    state = .inToolCall
                    results.append(.buffered)
                    return results
                }
            }
        }

        return results
    }

    public mutating func finalize() -> [ParsedToolCall] {
        guard !buffer.isEmpty else { return [] }

        var calls: [ParsedToolCall] = []

        // Try to extract any complete tool calls from remaining buffer
        let fullText = buffer
        buffer = ""
        state = .text

        // Try XML pattern
        let xmlCalls = _extractAllXMLToolCalls(fullText)
        calls.append(contentsOf: xmlCalls)

        // Try bracket pattern
        let bracketCalls = _extractAllBracketToolCalls(fullText)
        calls.append(contentsOf: bracketCalls)

        return calls
    }

    public mutating func reset() {
        buffer = ""
        state = .text
    }

    // MARK: - Private

    private func _hasPotentialTagPrefix(_ text: String) -> Bool {
        // Check if the text ends with a prefix of "<tool_call>"
        for i in 1..<Self.openTag.count {
            let prefix = String(Self.openTag.prefix(i))
            if text.hasSuffix(prefix) { return true }
        }
        return false
    }

    private func _hasPotentialCloseTagSuffix(_ text: String) -> Bool {
        for i in 1..<Self.closeTag.count {
            let suffix = String(Self.closeTag.prefix(i))
            if text.hasSuffix(suffix) { return true }
        }
        return false
    }

    private func _parseToolCallJSON(_ json: String) -> ParsedToolCall? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String else {
            return nil
        }

        let argsJSON: String
        if let argsDict = obj["arguments"] as? [String: Any],
           let argsData = try? JSONSerialization.data(withJSONObject: argsDict),
           let argsStr = String(data: argsData, encoding: .utf8) {
            argsJSON = argsStr
        } else if let argsStr = obj["arguments"] as? String {
            argsJSON = argsStr
        } else {
            argsJSON = "{}"
        }

        let id = obj["id"] as? String ?? ""
        return ParsedToolCall(name: name, argumentsJSON: argsJSON, id: id)
    }

    private func _extractAllXMLToolCalls(_ text: String) -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []
        var searchText = text

        while let openRange = searchText.range(of: Self.openTag),
              let closeRange = searchText.range(of: Self.closeTag, range: openRange.upperBound..<searchText.endIndex) {
            let jsonStr = String(searchText[openRange.upperBound..<closeRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            if let call = _parseToolCallJSON(jsonStr) {
                calls.append(call)
            }
            searchText = String(searchText[closeRange.upperBound...])
        }
        return calls
    }

    private func _tryExtractBracketStyle(_ text: inout String) -> ParsedToolCall? {
        // Pattern: [Calling tool: func_name({"arg": "value"})]
        guard let startRange = text.range(of: "[Calling tool: ") else { return nil }
        guard let endRange = text.range(of: ")]", range: startRange.upperBound..<text.endIndex) else { return nil }

        let inner = String(text[startRange.upperBound..<endRange.lowerBound])
        guard let parenIdx = inner.firstIndex(of: "(") else { return nil }

        let funcName = String(inner[inner.startIndex..<parenIdx]).trimmingCharacters(in: .whitespaces)
        let argsStr = String(inner[inner.index(after: parenIdx)...])

        guard let data = argsStr.data(using: .utf8),
              let _ = try? JSONSerialization.jsonObject(with: data) else {
            return nil
        }

        // Remove the matched bracket call from the buffer
        let afterEnd = text.index(endRange.upperBound, offsetBy: 0)
        let prefix = String(text[text.startIndex..<startRange.lowerBound])
        let suffix = String(text[afterEnd...])
        text = prefix + suffix

        return ParsedToolCall(name: funcName, argumentsJSON: argsStr)
    }

    private func _extractAllBracketToolCalls(_ text: String) -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []
        var searchText = text

        while let startRange = searchText.range(of: "[Calling tool: "),
              let endRange = searchText.range(of: ")]", range: startRange.upperBound..<searchText.endIndex) {
            let inner = String(searchText[startRange.upperBound..<endRange.lowerBound])
            if let parenIdx = inner.firstIndex(of: "(") {
                let funcName = String(inner[inner.startIndex..<parenIdx]).trimmingCharacters(in: .whitespaces)
                let argsStr = String(inner[inner.index(after: parenIdx)...])
                if let data = argsStr.data(using: .utf8),
                   let _ = try? JSONSerialization.jsonObject(with: data) {
                    calls.append(ParsedToolCall(name: funcName, argumentsJSON: argsStr))
                }
            }
            searchText = String(searchText[endRange.upperBound...])
        }
        return calls
    }
}
