//
//  MethodTools.swift
//  project2501
//
//  Agent-facing tools for saving and reporting on methods.
//  Search and load are handled by capabilities_search / capabilities_load.
//

import Foundation

// MARK: - methods_save

final class MethodsSaveTool: Project2501Tool, @unchecked Sendable {
    let name = "methods_save"
    let description =
        "Save a reusable method. Provide the tool-call sequence as YAML steps."

    let parameters: JSONValue? = .object([
        "type": .string("object"),
        "properties": .object([
            "name": .object([
                "type": .string("string"),
                "description": .string("Short name for the method"),
            ]),
            "description": .object([
                "type": .string("string"),
                "description": .string("What this method does"),
            ]),
            "trigger_text": .object([
                "type": .string("string"),
                "description": .string("Phrases that should trigger this method"),
            ]),
            "steps_yaml": .object([
                "type": .string("string"),
                "description": .string(
                    "YAML steps for the method. Write the tool-call sequence from the current session."
                ),
            ]),
        ]),
        "required": .array([.string("name"), .string("description"), .string("steps_yaml")]),
    ])

    func execute(argumentsJSON: String) async throws -> String {
        guard let args = parseArguments(argumentsJSON),
            let rawName = args["name"] as? String,
            let rawDesc = args["description"] as? String,
            let rawYaml = args["steps_yaml"] as? String
        else {
            return "Error: 'name', 'description', and 'steps_yaml' parameters are required."
        }

        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        let description = rawDesc.trimmingCharacters(in: .whitespacesAndNewlines)
        let yaml = rawYaml.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !description.isEmpty else {
            return "Error: 'name' and 'description' must not be blank."
        }
        guard !yaml.isEmpty else {
            return "Error: 'steps_yaml' must not be blank."
        }

        let triggerText = args["trigger_text"] as? String

        let method = try await MethodService.shared.create(
            name: name,
            description: description,
            triggerText: triggerText,
            body: yaml,
            source: .user
        )
        return
            "Method '\(method.name)' saved (id: \(method.id), tools: \(method.toolsUsed.joined(separator: ", ")))."
    }
}

// MARK: - methods_report

final class MethodsReportTool: Project2501Tool, @unchecked Sendable {
    let name = "methods_report"
    let description =
        "Report the outcome of following a method. "
        + "Call after completing (or failing) a method's steps to update its score."

    let parameters: JSONValue? = .object([
        "type": .string("object"),
        "properties": .object([
            "id": .object([
                "type": .string("string"),
                "description": .string("Method ID to report on"),
            ]),
            "outcome": .object([
                "type": .string("string"),
                "description": .string("Result: 'succeeded' or 'failed'"),
                "enum": .array([.string("succeeded"), .string("failed")]),
            ]),
            "notes": .object([
                "type": .string("string"),
                "description": .string("Optional notes about what happened"),
            ]),
        ]),
        "required": .array([.string("id"), .string("outcome")]),
    ])

    func execute(argumentsJSON: String) async throws -> String {
        guard let args = parseArguments(argumentsJSON),
            let id = args["id"] as? String,
            let outcomeStr = args["outcome"] as? String,
            let outcome = MethodEventType(rawValue: outcomeStr),
            (outcome == .succeeded || outcome == .failed)
        else {
            return "Error: 'id' and 'outcome' (succeeded/failed) are required."
        }

        let notes = args["notes"] as? String

        guard let method = try await MethodService.shared.load(id: id) else {
            return "Error: Method '\(id)' not found."
        }

        try await MethodService.shared.reportOutcome(
            methodId: id,
            outcome: outcome,
            agentId: WorkExecutionContext.currentIssueId,
            notes: notes
        )

        let score = try await MethodService.shared.loadScore(methodId: id)
        let scoreStr = score.map { String(format: "%.2f", $0.score) } ?? "N/A"
        return "Reported '\(outcomeStr)' for method '\(method.name)'. Current score: \(scoreStr)."
    }
}
