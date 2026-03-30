import Foundation

/// Tool call parser for Llama 3/4 models.
///
/// Supports two Llama tool call formats:
/// - Function tag style: `<function=name>{"arg": "value"}</function>`
/// - Python tag style: `<|python_tag|>{"name": "func", "parameters": {...}}`
public struct LlamaToolParser: ToolCallParser {

    public static var supportedModels: [String] { ["llama"] }

    private var buffer: String = ""
    private var state: State = .text

    private enum State {
        case text
        case potentialTag       // Accumulating what might be `<function=...>` or `<|python_tag|>`
        case inFunctionCall     // Between `<function=name>` and `</function>`
        case inPythonTag        // After `<|python_tag|>`, accumulating JSON
    }

    private static let functionPrefix = "<function="
    private static let functionClose = "</function>"
    private static let pythonTag = "<|python_tag|>"

    public init() {}

    public mutating func processChunk(_ text: String) -> [ToolParserResult] {
        var results: [ToolParserResult] = []
        buffer += text

        while !buffer.isEmpty {
            switch state {
            case .text:
                // Check for <function= tag
                if let funcRange = buffer.range(of: Self.functionPrefix) {
                    let prefix = String(buffer[buffer.startIndex..<funcRange.lowerBound])
                    if !prefix.isEmpty {
                        results.append(.text(prefix))
                    }
                    buffer = String(buffer[funcRange.lowerBound...])

                    // Find the closing `>` of the function tag to extract the name
                    if let closeAngle = buffer.range(of: ">", range: buffer.index(buffer.startIndex, offsetBy: Self.functionPrefix.count)..<buffer.endIndex) {
                        state = .inFunctionCall
                        // Keep the full tag in buffer for later extraction
                    } else {
                        // Partial tag, need more data
                        results.append(.buffered)
                        return results
                    }
                    continue
                }

                // Check for <|python_tag|>
                if let pyRange = buffer.range(of: Self.pythonTag) {
                    let prefix = String(buffer[buffer.startIndex..<pyRange.lowerBound])
                    if !prefix.isEmpty {
                        results.append(.text(prefix))
                    }
                    buffer = String(buffer[pyRange.upperBound...])
                    state = .inPythonTag
                    continue
                }

                // Check for potential partial tag at end
                if _hasPotentialTagStart(buffer) {
                    state = .potentialTag
                    results.append(.buffered)
                    return results
                }

                // No tag found
                results.append(.text(buffer))
                buffer = ""

            case .potentialTag:
                // Check if we now have a full tag
                if buffer.contains(Self.functionPrefix) || buffer.contains(Self.pythonTag) {
                    state = .text
                    continue
                } else if Self.functionPrefix.hasPrefix(buffer) || Self.pythonTag.hasPrefix(buffer) ||
                          _isPartialPrefixOf(buffer, target: Self.functionPrefix) ||
                          _isPartialPrefixOf(buffer, target: Self.pythonTag) {
                    results.append(.buffered)
                    return results
                } else {
                    state = .text
                    continue
                }

            case .inFunctionCall:
                // We have `<function=name>...` in buffer, look for `</function>`
                if let closeRange = buffer.range(of: Self.functionClose) {
                    // Extract name and arguments
                    if let call = _parseFunctionTag(String(buffer[buffer.startIndex..<closeRange.lowerBound])) {
                        results.append(.toolCall(call))
                    } else {
                        results.append(.text(String(buffer[buffer.startIndex..<closeRange.upperBound])))
                    }
                    buffer = String(buffer[closeRange.upperBound...])
                    state = .text
                } else {
                    results.append(.buffered)
                    return results
                }

            case .inPythonTag:
                // After <|python_tag|>, accumulate JSON. It ends at end-of-text or another special token.
                // Try to parse accumulated buffer as JSON
                let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
                if let call = _parsePythonTagJSON(trimmed) {
                    results.append(.toolCall(call))
                    buffer = ""
                    state = .text
                } else if _looksLikeCompleteJSON(trimmed) {
                    // JSON is complete but doesn't parse as tool call, emit as text
                    results.append(.text(buffer))
                    buffer = ""
                    state = .text
                } else {
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
        let text = buffer
        buffer = ""
        state = .text

        // Try function tag pattern
        calls.append(contentsOf: _extractAllFunctionTags(text))

        // Try python tag pattern
        if calls.isEmpty, let pyRange = text.range(of: Self.pythonTag) {
            let jsonPart = String(text[pyRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if let call = _parsePythonTagJSON(jsonPart) {
                calls.append(call)
            }
        }

        // If in python tag state, try the raw buffer
        if calls.isEmpty {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if let call = _parsePythonTagJSON(trimmed) {
                calls.append(call)
            }
        }

        return calls
    }

    public mutating func reset() {
        buffer = ""
        state = .text
    }

    // MARK: - Private

    private func _hasPotentialTagStart(_ text: String) -> Bool {
        // Check if text ends with a prefix of either tag
        for tag in [Self.functionPrefix, Self.pythonTag] {
            for i in 1..<tag.count {
                if text.hasSuffix(String(tag.prefix(i))) { return true }
            }
        }
        return false
    }

    private func _isPartialPrefixOf(_ text: String, target: String) -> Bool {
        return target.hasPrefix(text)
    }

    /// Parse `<function=name>{"arg": "val"}` format
    private func _parseFunctionTag(_ text: String) -> ParsedToolCall? {
        guard text.hasPrefix(Self.functionPrefix) else { return nil }

        let afterPrefix = String(text.dropFirst(Self.functionPrefix.count))
        guard let closeAngle = afterPrefix.firstIndex(of: ">") else { return nil }

        let funcName = String(afterPrefix[afterPrefix.startIndex..<closeAngle])
        let argsStr = String(afterPrefix[afterPrefix.index(after: closeAngle)...]).trimmingCharacters(in: .whitespacesAndNewlines)

        // Validate JSON
        if !argsStr.isEmpty,
           let data = argsStr.data(using: .utf8),
           let _ = try? JSONSerialization.jsonObject(with: data) {
            return ParsedToolCall(name: funcName, argumentsJSON: argsStr)
        }

        // Empty args or non-JSON args
        return ParsedToolCall(name: funcName, argumentsJSON: argsStr.isEmpty ? "{}" : argsStr)
    }

    /// Parse `{"name": "func", "parameters": {...}}` from python tag
    private func _parsePythonTagJSON(_ json: String) -> ParsedToolCall? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String else {
            return nil
        }

        // Llama uses "parameters" key (not "arguments")
        let argsJSON: String
        if let params = obj["parameters"] as? [String: Any],
           let paramsData = try? JSONSerialization.data(withJSONObject: params),
           let paramsStr = String(data: paramsData, encoding: .utf8) {
            argsJSON = paramsStr
        } else if let args = obj["arguments"] as? [String: Any],
                  let argsData = try? JSONSerialization.data(withJSONObject: args),
                  let argsStr = String(data: argsData, encoding: .utf8) {
            argsJSON = argsStr
        } else if let argsStr = obj["parameters"] as? String {
            argsJSON = argsStr
        } else if let argsStr = obj["arguments"] as? String {
            argsJSON = argsStr
        } else {
            argsJSON = "{}"
        }

        return ParsedToolCall(name: name, argumentsJSON: argsJSON)
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

    private func _extractAllFunctionTags(_ text: String) -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []
        var searchText = text

        while let openRange = searchText.range(of: Self.functionPrefix),
              let closeRange = searchText.range(of: Self.functionClose, range: openRange.upperBound..<searchText.endIndex) {
            let tagContent = String(searchText[openRange.lowerBound..<closeRange.lowerBound])
            if let call = _parseFunctionTag(tagContent) {
                calls.append(call)
            }
            searchText = String(searchText[closeRange.upperBound...])
        }
        return calls
    }
}
