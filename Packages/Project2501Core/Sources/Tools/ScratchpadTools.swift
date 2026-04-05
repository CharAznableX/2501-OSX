//
//  ScratchpadTools.swift
//  project2501
//
//  Persistent scratchpad for agent notes that survive context compaction and session resume.
//

import Foundation

/// Saves notes to persistent storage (backed by IssueEvent).
final class SaveNotesTool: OsaurusTool, @unchecked Sendable {
    let name = "save_notes"
    let description =
        "Save important findings, decisions, or progress notes that persist across context resets. "
        + "Use this to record key discoveries, architectural decisions, file locations, error patterns, "
        + "or any information you'll need later. Notes are append-only within a task."

    let parameters: JSONValue? = .object([
        "type": .string("object"),
        "properties": .object([
            "content": .object([
                "type": .string("string"),
                "description": .string(
                    "The notes to save. Be specific — include file paths, values, decisions, and reasoning."
                ),
            ])
        ]),
        "required": .array([.string("content")]),
    ])

    func execute(argumentsJSON: String) async throws -> String {
        guard let args = parseArguments(argumentsJSON),
            let content = args["content"] as? String,
            !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return "Error: 'content' parameter is required and must be non-empty."
        }

        guard let issueId = WorkExecutionContext.currentIssueId else {
            return "Error: No active task context."
        }

        let event = IssueEvent.withPayload(
            issueId: issueId,
            eventType: .noteSaved,
            payload: EventPayload.NoteSaved(content: content)
        )
        _ = try? IssueStore.createEvent(event)
        return "Notes saved."
    }
}

/// Reads previously saved notes for the current task.
final class ReadNotesTool: OsaurusTool, @unchecked Sendable {
    let name = "read_notes"
    let description =
        "Read previously saved notes for the current task. "
        + "Use this at the start of resumed work or when you need to recall earlier findings."

    let parameters: JSONValue? = .object([
        "type": .string("object"),
        "properties": .object([:]),
    ])

    func execute(argumentsJSON: String) async throws -> String {
        guard let issueId = WorkExecutionContext.currentIssueId else {
            return "Error: No active task context."
        }
        return Self.loadNotes(issueId: issueId)
    }

    /// Shared helper so resume logic can also read notes without a tool call.
    static func loadNotes(issueId: String) -> String {
        guard let events = try? IssueStore.getEvents(issueId: issueId, ofType: .noteSaved),
            !events.isEmpty
        else {
            return "No notes saved yet for this task."
        }

        let decoder = JSONDecoder()
        let decoded =
            events
            .enumerated()
            .compactMap { index, event -> String? in
                guard let payload = event.payload,
                    let data = payload.data(using: .utf8),
                    let note = try? decoder.decode(EventPayload.NoteSaved.self, from: data)
                else { return nil }
                return "[\(index + 1)] \(note.content)"
            }

        if decoded.isEmpty {
            return "No notes saved yet for this task."
        }
        return decoded.joined(separator: "\n\n")
    }
}
