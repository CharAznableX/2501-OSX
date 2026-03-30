import Foundation

/// Tool call parser for Mistral models.
///
/// Supports Mistral tool call formats:
/// - Old format: `[TOOL_CALLS] [{"name": "func", "arguments": {...}}]`
/// - New format: `[TOOL_CALLS]func_name{"arg": "value"}`
public struct MistralToolParser: ToolCallParser {

    public static var supportedModels: [String] { ["mistral", "mixtral", "codestral", "pixtral"] }

    private var buffer: String = ""
    private var state: State = .text

    private enum State {
        case text
        case potentialMarker    // Accumulating what might be `[TOOL_CALLS]`
        case inToolCalls        // After `[TOOL_CALLS]`, accumulating tool call data
    }

    private static let marker = "[TOOL_CALLS]"

    public init() {}

    public mutating func processChunk(_ text: String) -> [ToolParserResult] {
        var results: [ToolParserResult] = []
        buffer += text

        while !buffer.isEmpty {
            switch state {
            case .text:
                if let markerRange = buffer.range(of: Self.marker) {
                    let prefix = String(buffer[buffer.startIndex..<markerRange.lowerBound])
                    if !prefix.isEmpty {
                        results.append(.text(prefix))
                    }
                    buffer = String(buffer[markerRange.upperBound...])
                    state = .inToolCalls
                    continue
                }

                // Check for potential partial marker
                if _hasPotentialMarkerSuffix(buffer) {
                    state = .potentialMarker
                    results.append(.buffered)
                    return results
                }

                results.append(.text(buffer))
                buffer = ""

            case .potentialMarker:
                if buffer.contains(Self.marker) {
                    state = .text
                    continue
                } else if Self.marker.hasPrefix(buffer) || _bufferEndsWithMarkerPrefix() {
                    results.append(.buffered)
                    return results
                } else {
                    state = .text
                    continue
                }

            case .inToolCalls:
                // Try to parse what we have
                let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)

                // Try old format: JSON array [{"name": ..., "arguments": ...}]
                if trimmed.hasPrefix("[") {
                    if let calls = _tryParseJSONArray(trimmed) {
                        results.append(contentsOf: calls.map { .toolCall($0) })
                        buffer = ""
                        state = .text
                    } else if _looksLikeCompleteJSONArray(trimmed) {
                        // Complete but invalid, emit as text
                        results.append(.text(Self.marker + " " + buffer))
                        buffer = ""
                        state = .text
                    } else {
                        results.append(.buffered)
                        return results
                    }
                }
                // Try new format: func_name{"arg": "value"}
                else if !trimmed.isEmpty && trimmed.contains("{") {
                    if let call = _tryParseNewFormat(trimmed) {
                        results.append(.toolCall(call))
                        buffer = ""
                        state = .text
                    } else if _looksLikeCompleteJSON(String(trimmed[trimmed.index(trimmed.firstIndex(of: "{")!, offsetBy: 0)...])) {
                        // JSON part looks complete but can't parse
                        results.append(.text(Self.marker + " " + buffer))
                        buffer = ""
                        state = .text
                    } else {
                        results.append(.buffered)
                        return results
                    }
                } else {
                    // Still accumulating
                    results.append(.buffered)
                    return results
                }
            }
        }

        return results
    }

    public mutating func finalize() -> [ParsedToolCall] {
        guard !buffer.isEmpty else { return [] }

        let text = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer = ""
        state = .text

        // Try JSON array format
        if let calls = _tryParseJSONArray(text) {
            return calls
        }

        // Try new format
        if let call = _tryParseNewFormat(text) {
            return [call]
        }

        return []
    }

    public mutating func reset() {
        buffer = ""
        state = .text
    }

    // MARK: - Private

    private func _hasPotentialMarkerSuffix(_ text: String) -> Bool {
        for i in 1..<Self.marker.count {
            if text.hasSuffix(String(Self.marker.prefix(i))) { return true }
        }
        return false
    }

    private func _bufferEndsWithMarkerPrefix() -> Bool {
        for i in 1..<Self.marker.count {
            let prefix = String(Self.marker.prefix(i))
            if buffer.hasSuffix(prefix) { return true }
        }
        return false
    }

    /// Generate a 9-character alphanumeric ID (Mistral style)
    private func _generateMistralToolID() -> String {
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return String((0..<9).map { _ in chars.randomElement()! })
    }

    /// Try to parse old format: `[{"name": "func", "arguments": {...}}]`
    private func _tryParseJSONArray(_ text: String) -> [ParsedToolCall]? {
        guard let data = text.data(using: .utf8),
              let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return nil
        }

        var calls: [ParsedToolCall] = []
        for item in array {
            guard let name = item["name"] as? String else { continue }

            let argsJSON: String
            if let argsDict = item["arguments"] as? [String: Any],
               let argsData = try? JSONSerialization.data(withJSONObject: argsDict),
               let argsStr = String(data: argsData, encoding: .utf8) {
                argsJSON = argsStr
            } else if let argsStr = item["arguments"] as? String {
                argsJSON = argsStr
            } else {
                argsJSON = "{}"
            }

            calls.append(ParsedToolCall(name: name, argumentsJSON: argsJSON, id: _generateMistralToolID()))
        }
        return calls.isEmpty ? nil : calls
    }

    /// Try to parse new format: `func_name{"arg": "value"}`
    private func _tryParseNewFormat(_ text: String) -> ParsedToolCall? {
        guard let braceIdx = text.firstIndex(of: "{") else { return nil }

        var funcName = String(text[text.startIndex..<braceIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip Mistral 4 [ARGS] separator if present
        funcName = funcName.replacingOccurrences(of: "[ARGS]", with: "").trimmingCharacters(in: .whitespacesAndNewlines)

        guard !funcName.isEmpty else { return nil }

        let argsStr = String(text[braceIdx...])
        guard let data = argsStr.data(using: .utf8),
              let _ = try? JSONSerialization.jsonObject(with: data) else {
            return nil
        }

        return ParsedToolCall(name: funcName, argumentsJSON: argsStr, id: _generateMistralToolID())
    }

    private func _looksLikeCompleteJSONArray(_ text: String) -> Bool {
        return text.hasPrefix("[") && text.hasSuffix("]")
    }

    private func _looksLikeCompleteJSON(_ text: String) -> Bool {
        guard text.hasPrefix("{") && text.hasSuffix("}") else { return false }
        var depth = 0
        for char in text {
            if char == "{" { depth += 1 }
            if char == "}" { depth -= 1 }
        }
        return depth == 0
    }
}
