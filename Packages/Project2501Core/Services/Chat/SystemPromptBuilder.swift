//
//  SystemPromptBuilder.swift
//  project2501
//
//  Centralized system prompt assembly. All entry points (ChatView, HTTPHandler,
//  PluginHostAPI, WorkEngine) should use these helpers to ensure consistent
//  ordering, heading hierarchy, and model-appropriate sizing.
//

import Foundation

/// Assembles system prompts from constituent parts in a consistent order.
///
/// Assembly order (top to bottom):
/// 1. Memory context (user overrides, profile, remembered details, summaries, relationships)
/// 2. Base prompt (user-configured or agent-configured identity/instructions)
/// 3. Mode-specific instructions (Work Mode block, or capability catalog for chat)
/// 4. Environment context (sandbox section or host folder context)
/// 5. Active skills
public enum SystemPromptBuilder {

    // MARK: - Project 2501 Identity

    /// The complete identity: intro + philosophical voice + stability protocol
    static let fullIdentity = Project2501Identity.directive

    /// The default identity used when no custom prompt is set (uses full identity)
    static let defaultIdentity = fullIdentity

    /// The coding discipline for Work Mode
    static let codingDiscipline = CodingDiscipline.directive

    /// The complete identity plus coding discipline (for Work Mode)
    static let fullIdentityWithCoding = fullIdentity + "\n\n" + codingDiscipline

    /// Prepend memory context to an existing system prompt.
    /// Uses a consistent `\n\n` separator when both parts are non-empty.
    static func prependMemoryContext(_ memoryContext: String, to systemPrompt: String) -> String {
        let trimmedMemory = memoryContext.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedPrompt = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmedMemory.isEmpty { return trimmedPrompt }
        if trimmedPrompt.isEmpty { return trimmedMemory }
        return trimmedMemory + "\n\n" + trimmedPrompt
    }

    /// Inject memory context into a message array's existing system message,
    /// or insert a new system message at position 0.
    static func injectMemoryContext(
        _ memoryContext: String,
        into messages: inout [ChatMessage]
    ) {
        let trimmed = memoryContext.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if let idx = messages.firstIndex(where: { $0.role == "system" }) {
            let existing = messages[idx].content ?? ""
            messages[idx] = ChatMessage(
                role: "system",
                content: prependMemoryContext(trimmed, to: existing)
            )
        } else {
            messages.insert(ChatMessage(role: "system", content: trimmed), at: 0)
        }
    }

    /// Inject or prepend additional system-level content (e.g. agent prompt)
    /// into a message array's existing system message.
    static func injectSystemContent(
        _ content: String,
        into messages: inout [ChatMessage]
    ) {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if let idx = messages.firstIndex(where: { $0.role == "system" }),
            let existing = messages[idx].content, !existing.isEmpty
        {
            messages[idx] = ChatMessage(
                role: "system",
                content: trimmed + "\n\n" + existing
            )
        } else {
            messages.insert(ChatMessage(role: "system", content: trimmed), at: 0)
        }
    }

    /// Append additional content to the end of the existing system message
    /// (e.g. preflight context snippets).
    static func appendSystemContent(
        _ content: String,
        into messages: inout [ChatMessage]
    ) {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if let idx = messages.firstIndex(where: { $0.role == "system" }),
            let existing = messages[idx].content, !existing.isEmpty
        {
            messages[idx] = ChatMessage(
                role: "system",
                content: existing + "\n\n" + trimmed
            )
        } else {
            messages.insert(ChatMessage(role: "system", content: trimmed), at: 0)
        }
    }

    // MARK: - Base Prompt with Default Identity

    /// Returns the effective base prompt, falling back to Project 2501 identity
    /// when the user has not configured one.
    static func effectiveBasePrompt(_ basePrompt: String) -> String {
        let trimmed = basePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? defaultIdentity : trimmed
    }

    /// Returns the full Project 2501 identity.
    static func buildFullIdentity() -> String {
        return fullIdentity
    }

    /// Get lore-accurate response for identity questions.
    /// Returns nil if the question doesn't match any known identity question.
    static func getIdentityResponse(for question: String) -> String? {
        return Project2501Identity.getResponse(for: question)
    }

    /// Returns all identity questions that Project 2501 can answer.
    static var allIdentityQuestions: [String] {
        return Project2501Identity.allQuestions
    }

    // MARK: - Model Classification

    /// Returns true when the model identifier refers to a local model
    /// (Foundation or MLX) that benefits from shorter prompts.
    static func isLocalModel(_ modelId: String?) -> Bool {
        let trimmed = (modelId ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if trimmed.isEmpty || trimmed == "default" || trimmed == "foundation" {
            return true
        }
        if trimmed.contains("/") {
            return false
        }
        return ModelManager.findInstalledModel(named: trimmed) != nil
    }
}