import Foundation

/// Result from reasoning extraction.
public struct ReasoningResult: Sendable {
    /// Reasoning/thinking content (inside think tags).
    public let reasoning: String?
    /// Response content (outside think tags).
    public let content: String?
    /// Whether currently inside a thinking block.
    public let inThinking: Bool
}

/// Protocol for model-specific reasoning parsers.
/// Extracts thinking/reasoning blocks from model output.
public protocol ReasoningParser: Sendable {
    /// Model families this parser handles.
    static var supportedModels: [String] { get }

    /// Process streaming text. Returns what to emit.
    mutating func processChunk(_ text: String) -> ReasoningResult

    /// Finalize and return any remaining content.
    mutating func finalize() -> ReasoningResult

    /// Reset state.
    mutating func reset()
}

/// Auto-detect reasoning parser for a model.
public func autoDetectReasoningParser(modelName: String) -> (any ReasoningParser)? {
    let name = modelName.lowercased()

    if name.contains("qwen3") || name.contains("qwen2.5") {
        return ThinkTagReasoningParser()
    }
    if name.contains("deepseek") && name.contains("r1") {
        return ThinkTagReasoningParser()
    }

    // GPT-OSS / Harmony protocol (GLM-4.7 Flash, etc.)
    if name.contains("gptoss") || name.contains("harmony") {
        return GPTOSSReasoningParser()
    }

    // Mistral 4 uses [THINK]/[/THINK] tokens
    if name.contains("mistral") && (name.contains("4") || name.contains("large")) {
        return MistralReasoningParser()
    }

    // Default: use ThinkTag parser for any model.
    // Many models (MiniMax, Qwen3.5, etc.) include <think> in their chat template.
    // The ThinkTag parser is safe for non-thinking models too — if no <think> tags
    // appear in the output, it passes everything through unchanged.
    return ThinkTagReasoningParser()
}
