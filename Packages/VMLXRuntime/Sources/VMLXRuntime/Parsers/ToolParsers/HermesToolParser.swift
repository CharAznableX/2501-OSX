import Foundation

/// Tool call parser for Hermes/NousResearch models.
///
/// Supports Hermes tool call formats:
/// - XML tags: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
/// - Raw JSON fallback: `{"name": "func", "arguments": {...}}` (when tags are omitted)
/// - Reasoning tags: `<tool_call_reasoning>...</tool_call_reasoning>` (stripped)
///
/// Very similar to Qwen but with additional fallback patterns and reasoning tag support.
public struct HermesToolParser: ToolCallParser {

    public static var supportedModels: [String] { ["hermes", "nous"] }

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
    private static let reasoningOpenTag = "<tool_call_reasoning>"
    private static let reasoningCloseTag = "</tool_call_reasoning>"

    public init() {}

    public mutating func processChunk(_ text: String) -> [ToolParserResult] {
        var results: [ToolParserResult] = []
        buffer += text

        while !buffer.isEmpty {
            switch state {
            case .text:
                // Strip reasoning tags inline (they're informational, not content)
                if let reasonRange = buffer.range(of: Self.reasoningOpenTag) {
                    if let reasonEnd = buffer.range(of: Self.reasoningCloseTag, range: reasonRange.upperBound..<buffer.endIndex) {
                        // Emit text before reasoning, strip the reasoning block
                        let prefix = String(buffer[buffer.startIndex..<reasonRange.lowerBound])
                        if !prefix.isEmpty {
                            results.append(.text(prefix))
                        }
                        buffer = String(buffer[reasonEnd.upperBound...])
                        continue
                    }
                }

                // Look for start of `<tool_call>` tag
                if let tagRange = buffer.range(of: Self.openTag) {
                    let prefix = String(buffer[buffer.startIndex..<tagRange.lowerBound])
                    if !prefix.isEmpty {
                        results.append(.text(prefix))
                    }
                    buffer = String(buffer[tagRange.upperBound...])
                    state = .inToolCall
                } else if _hasPotentialTagPrefix(buffer, tag: Self.openTag) {
                    state = .potentialTag
                    results.append(.buffered)
                    return results
                } else {
                    // Fallback: check for raw JSON tool call (no tags)
                    var temp = buffer
                    if let rawCall = _tryExtractRawJSON(&temp) {
                        buffer = temp
                        results.append(.toolCall(rawCall))
                        continue
                    }
                    results.append(.text(buffer))
                    buffer = ""
                }

            case .potentialTag:
                if buffer.hasPrefix(Self.openTag) || buffer.contains(Self.openTag) {
                    state = .text
                    continue
                } else if Self.openTag.hasPrefix(buffer) {
                    results.append(.buffered)
                    return results
                } else {
                    state = .text
                    continue
                }

            case .inToolCall:
                if let closeRange = buffer.range(of: Self.closeTag) {
                    let jsonStr = String(buffer[buffer.startIndex..<closeRange.lowerBound])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    buffer = String(buffer[closeRange.upperBound...])
                    state = .text

                    if let toolCall = _parseToolCallJSON(jsonStr) {
                        results.append(.toolCall(toolCall))
                    } else {
                        results.append(.text(Self.openTag + jsonStr + Self.closeTag))
                    }
                } else if buffer.contains("</") {
                    state = .potentialEndTag
                    continue
                } else {
                    results.append(.buffered)
                    return results
                }

            case .potentialEndTag:
                if buffer.contains(Self.closeTag) {
                    state = .inToolCall
                    continue
                } else if _hasPotentialTagPrefix(buffer, tag: Self.closeTag) {
                    results.append(.buffered)
                    return results
                } else {
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
        let fullText = buffer
        buffer = ""
        state = .text

        // Strip reasoning tags
        let cleaned = _stripReasoningTags(fullText)

        // Try XML-tagged tool calls
        calls.append(contentsOf: _extractAllXMLToolCalls(cleaned))

        // Try raw JSON fallback
        if calls.isEmpty {
            calls.append(contentsOf: _extractAllRawJSONToolCalls(cleaned))
        }

        return calls
    }

    public mutating func reset() {
        buffer = ""
        state = .text
    }

    // MARK: - Private

    private func _hasPotentialTagPrefix(_ text: String, tag: String) -> Bool {
        for i in 1..<tag.count {
            let prefix = String(tag.prefix(i))
            if text.hasSuffix(prefix) { return true }
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
            let jsonStr = String(searchText[openRange.upperBound..<closeRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let call = _parseToolCallJSON(jsonStr) {
                calls.append(call)
            }
            searchText = String(searchText[closeRange.upperBound...])
        }
        return calls
    }

    /// Try to extract a raw JSON tool call `{"name": "...", "arguments": {...}}` from text.
    private mutating func _tryExtractRawJSON(_ text: inout String) -> ParsedToolCall? {
        guard let nameRange = text.range(of: #"{"name":"#) ?? text.range(of: #"{"name" :"#) else {
            return nil
        }

        // Find the matching closing brace using depth tracking
        let startIdx = nameRange.lowerBound
        var depth = 0
        var endIdx: String.Index?
        var idx = startIdx

        while idx < text.endIndex {
            let c = text[idx]
            if c == "{" { depth += 1 }
            if c == "}" {
                depth -= 1
                if depth == 0 {
                    endIdx = text.index(after: idx)
                    break
                }
            }
            idx = text.index(after: idx)
        }

        guard let end = endIdx else { return nil }

        let jsonStr = String(text[startIdx..<end])
        guard let call = _parseToolCallJSON(jsonStr) else { return nil }

        // Remove matched JSON from buffer
        let prefix = String(text[text.startIndex..<startIdx])
        let suffix = String(text[end...])
        text = prefix + suffix

        return call
    }

    private func _extractAllRawJSONToolCalls(_ text: String) -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []
        var remaining = text

        while let nameRange = remaining.range(of: #"{"name":"#) ?? remaining.range(of: #"{"name" :"#) {
            let startIdx = nameRange.lowerBound
            var depth = 0
            var endIdx: String.Index?
            var idx = startIdx

            while idx < remaining.endIndex {
                let c = remaining[idx]
                if c == "{" { depth += 1 }
                if c == "}" {
                    depth -= 1
                    if depth == 0 {
                        endIdx = remaining.index(after: idx)
                        break
                    }
                }
                idx = remaining.index(after: idx)
            }

            guard let end = endIdx else { break }

            let jsonStr = String(remaining[startIdx..<end])
            if let call = _parseToolCallJSON(jsonStr) {
                calls.append(call)
            }
            remaining = String(remaining[end...])
        }

        return calls
    }

    private func _stripReasoningTags(_ text: String) -> String {
        var result = text
        while let openRange = result.range(of: Self.reasoningOpenTag),
              let closeRange = result.range(of: Self.reasoningCloseTag, range: openRange.upperBound..<result.endIndex) {
            result = String(result[result.startIndex..<openRange.lowerBound]) +
                     String(result[closeRange.upperBound...])
        }
        return result
    }
}
