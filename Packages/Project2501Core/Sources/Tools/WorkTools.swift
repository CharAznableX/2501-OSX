//
//  WorkTools.swift
//  project2501
//
//  Work-specific tools for task completion, issue creation, and artifact generation.
//

import Foundation

// MARK: - Complete Task Tool

/// Tool for work mode to mark the current task as complete
public struct CompleteTaskTool: Project2501Tool {
    public let name = "complete_task"
    public let description =
        "Mark the current task as complete with a summary of the verified result. IMPORTANT: Call share_artifact for any generated files BEFORE calling this tool."

    public let parameters: JSONValue? = .object([
        "type": .string("object"),
        "properties": .object([
            "summary": .object([
                "type": .string("string"),
                "description": .string("Brief one-line summary of what was accomplished"),
            ]),
            "success": .object([
                "type": .string("boolean"),
                "description": .string("Whether the task was fully successful"),
            ]),
            "artifact": .object([
                "type": .string("string"),
                "description": .string(
                    "Optional final artifact in markdown format. Include this when a richer report or formatted result would help the user."
                ),
            ]),
            "remaining_work": .object([
                "type": .string("string"),
                "description": .string("Any remaining work that wasn't completed (optional)"),
            ]),
        ]),
        "required": .array([.string("summary"), .string("success")]),
    ])

    public init() {}

    public func execute(argumentsJSON: String) async throws -> String {
        guard let data = argumentsJSON.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let summary = json["summary"] as? String,
            let success = coerceBool(json["success"])
        else {
            throw NSError(
                domain: "WorkTools",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Invalid completion format. Required: summary (string), success (true/false). Example: {\"summary\": \"Done\", \"success\": true}"
                ]
            )
        }

        let artifact =
            (json["artifact"] as? String)?
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")

        let remainingWork = json["remaining_work"] as? String

        var result = """
            Task completion reported:
            - Status: \(success ? "SUCCESS" : "PARTIAL")
            - Summary: \(summary)
            """

        if let artifact, !artifact.isEmpty {
            result += "\n- Artifact Length: \(artifact.count) characters"
        }

        if let remaining = remainingWork, !remaining.isEmpty {
            result += "\n- Remaining work: \(remaining)"
        }

        return result
    }
}

// MARK: - Share Artifact Tool

/// Unified tool for sharing files or inline content with the user.
/// Supports any file type, directories, and inline text content.
public struct ShareArtifactTool: Project2501Tool {
    public let name = "share_artifact"
    public let description =
        "Share a file, directory, or text content with the user. The user cannot see any files you create unless you call this tool. Always call this for generated images, charts, websites, reports, code output, etc."

    public let parameters: JSONValue? = .object([
        "type": .string("object"),
        "properties": .object([
            "path": .object([
                "type": .string("string"),
                "description": .string(
                    "Relative path to a file or directory to share. Resolved relative to your working directory."
                ),
            ]),
            "content": .object([
                "type": .string("string"),
                "description": .string(
                    "Inline text or markdown content to share directly. Use this when you want to share generated text without writing to a file first."
                ),
            ]),
            "filename": .object([
                "type": .string("string"),
                "description": .string(
                    "Filename for the artifact. Required when using 'content'. Optional with 'path' (defaults to the file/directory name)."
                ),
            ]),
            "description": .object([
                "type": .string("string"),
                "description": .string("Brief human-readable description of what this artifact is."),
            ]),
        ]),
        "required": .array([]),
    ])

    public init() {}

    public func execute(argumentsJSON: String) async throws -> String {
        guard let data = argumentsJSON.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw NSError(
                domain: "WorkTools",
                code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Invalid arguments. Provide at least one of: path (string), content (string)"
                ]
            )
        }

        let path = json["path"] as? String
        let rawContent = json["content"] as? String
        let filename = json["filename"] as? String
        let description = json["description"] as? String

        guard path != nil || rawContent != nil else {
            throw NSError(
                domain: "WorkTools",
                code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "At least one of 'path' or 'content' must be provided."
                ]
            )
        }

        if rawContent != nil {
            guard let fn = filename, !fn.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw NSError(
                    domain: "WorkTools",
                    code: 4,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "'filename' is required when using 'content' mode."
                    ]
                )
            }
        }

        let content = rawContent?
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")

        let resolvedFilename: String
        if let filename, !filename.isEmpty {
            resolvedFilename = filename.trimmingCharacters(in: .whitespacesAndNewlines)
        } else if let path {
            resolvedFilename = (path as NSString).lastPathComponent
        } else {
            resolvedFilename = "artifact.txt"
        }

        let mimeType = SharedArtifact.mimeType(from: resolvedFilename)

        var metadataDict: [String: Any] = [
            "filename": resolvedFilename,
            "mime_type": mimeType,
        ]
        if let path { metadataDict["path"] = path }
        if content != nil { metadataDict["has_content"] = true }
        if let description { metadataDict["description"] = description }

        let metadataJSON: String
        if let jsonData = try? JSONSerialization.data(withJSONObject: metadataDict),
            let jsonStr = String(data: jsonData, encoding: .utf8)
        {
            metadataJSON = jsonStr
        } else {
            metadataJSON = "{}"
        }

        var result = """
            Artifact shared:
            - Filename: \(resolvedFilename)
            - Type: \(mimeType)
            """
        if let description {
            result += "\n- Description: \(description)"
        }

        result += "\n\n---SHARED_ARTIFACT_START---\n"
        result += metadataJSON + "\n"
        if let content {
            result += content + "\n"
        }
        result += "---SHARED_ARTIFACT_END---"

        return result
    }
}

// MARK: - Create Issue Tool

/// Tool for creating follow-up issues discovered during execution
public struct CreateIssueTool: Project2501Tool {
    public let name = "create_issue"
    public let description =
        "Create a follow-up issue for work that was discovered but is outside the current task scope. Include detailed context about what you learned so the next execution can pick up without starting from scratch."

    public let parameters: JSONValue? = .object([
        "type": .string("object"),
        "properties": .object([
            "title": .object([
                "type": .string("string"),
                "description": .string("Short descriptive title for the issue"),
            ]),
            "description": .object([
                "type": .string("string"),
                "description": .string(
                    "Detailed description with full context. Include: what was discovered, why it's needed, relevant file paths, and any preliminary analysis."
                ),
            ]),
            "reason": .object([
                "type": .string("string"),
                "description": .string("Why this work was discovered/needed"),
            ]),
            "learnings": .object([
                "type": .string("array"),
                "items": .object(["type": .string("string")]),
                "description": .string("Key things learned that are relevant to this work"),
            ]),
            "relevant_files": .object([
                "type": .string("array"),
                "items": .object(["type": .string("string")]),
                "description": .string("File paths that are relevant to this issue"),
            ]),
            "priority": .object([
                "type": .string("string"),
                "enum": .array([
                    .string("p0"),
                    .string("p1"),
                    .string("p2"),
                    .string("p3"),
                ]),
                "description": .string("Priority level: p0 (urgent), p1 (high), p2 (medium), p3 (low)"),
            ]),
        ]),
        "required": .array([.string("title"), .string("description")]),
    ])

    public init() {}

    public func execute(argumentsJSON: String) async throws -> String {
        guard let data = argumentsJSON.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let title = json["title"] as? String,
            let description = json["description"] as? String
        else {
            throw NSError(
                domain: "WorkTools",
                code: 5,
                userInfo: [
                    NSLocalizedDescriptionKey: "Invalid issue format. Required: title (string), description (string)"
                ]
            )
        }

        let reason = json["reason"] as? String
        let learnings = coerceStringArray(json["learnings"])
        let relevantFiles = coerceStringArray(json["relevant_files"])

        let priorityStr = json["priority"] as? String ?? "p2"
        let priority: IssuePriority
        switch priorityStr.lowercased() {
        case "p0": priority = .p0
        case "p1": priority = .p1
        case "p3": priority = .p3
        default: priority = .p2
        }

        // Get current issue context for linking
        guard let currentIssueId = WorkExecutionContext.currentIssueId else {
            return """
                Issue creation recorded:
                - Title: \(title)
                - Priority: \(priorityStr.uppercased())
                - Description: \(description.prefix(200))...

                Note: No active execution context. Issue will be created when context is available.
                """
        }

        // Build rich handoff context
        let handoffContext = HandoffContext(
            title: title,
            description: description,
            reason: reason,
            learnings: learnings,
            relevantFiles: relevantFiles,
            constraints: nil,
            priority: priority,
            type: .discovery,
            isDiscoveredWork: true
        )

        // Create the issue with full context
        let newIssue = await IssueManager.shared.createIssueWithContextSafe(
            handoffContext,
            sourceIssueId: currentIssueId
        )

        guard let newIssue = newIssue else {
            return "Error: Failed to create issue. Please try again."
        }

        return """
            Successfully created follow-up issue:
            - ID: \(newIssue.id)
            - Title: \(title)
            - Priority: \(priorityStr.uppercased())
            - Status: Open

            The issue has been linked to the current task for tracking.
            Continue with your current task.
            """
    }
}

// MARK: - Request Clarification Tool

/// Tool for requesting clarification from the user when task is ambiguous
public struct RequestClarificationTool: Project2501Tool {
    public let name = "request_clarification"
    public let description =
        "Ask the user a question when the task is critically ambiguous. Only use this for ambiguities that would lead to wrong results if assumed incorrectly. Do NOT use for minor details or preferences."

    public let parameters: JSONValue? = .object([
        "type": .string("object"),
        "properties": .object([
            "question": .object([
                "type": .string("string"),
                "description": .string("Clear, specific question to ask the user"),
            ]),
            "options": .object([
                "type": .string("array"),
                "items": .object([
                    "type": .string("string")
                ]),
                "description": .string("Optional predefined choices for the user to select from"),
            ]),
            "context": .object([
                "type": .string("string"),
                "description": .string("Brief explanation of why this clarification is needed"),
            ]),
        ]),
        "required": .array([.string("question")]),
    ])

    public init() {}

    public func execute(argumentsJSON: String) async throws -> String {
        guard let data = argumentsJSON.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let question = json["question"] as? String
        else {
            throw NSError(
                domain: "WorkTools",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Invalid clarification format. Required: question (string)"]
            )
        }

        let options = coerceStringArray(json["options"])
        let context = json["context"] as? String
        var response = """
            Clarification requested:
            Question: \(question)
            """

        if let opts = options, !opts.isEmpty {
            response +=
                "\nOptions:\n" + opts.enumerated().map { "  \($0.offset + 1). \($0.element)" }.joined(separator: "\n")
        }

        if let ctx = context, !ctx.isEmpty {
            response += "\nContext: \(ctx)"
        }

        response += "\n\n---CLARIFICATION_NEEDED---"

        return response
    }
}

// MARK: - Tool Registration

/// Manager for work-specific tool registration
/// Uses reference counting to support multiple concurrent work sessions
@MainActor
public final class WorkToolManager {
    public static let shared = WorkToolManager()

    /// Cached tool instances (created once, reused)
    /// Note: SubmitPlanTool and ReportDiscoveryTool removed - no longer used with reasoning loop architecture
    private lazy var tools: [Project2501Tool] = [
        CompleteTaskTool(),
        CreateIssueTool(),
        RequestClarificationTool(),
        SaveNotesTool(),
        ReadNotesTool(),
    ]

    /// Reference count for active work sessions
    /// Tools stay registered while count > 0
    private var referenceCount = 0

    // MARK: - Folder Tools

    /// Folder tools (created dynamically based on folder context)
    private var folderTools: [Project2501Tool] = []

    /// Names of currently registered folder tools
    private var _folderToolNames: [String] = []

    /// Current folder context (if any)
    private var currentFolderContext: WorkFolderContext?

    private init() {}

    /// Whether work tools are currently registered
    public var isRegistered: Bool {
        referenceCount > 0
    }

    /// Returns the names of all work tools (excluding folder tools)
    public var toolNames: [String] {
        tools.map { $0.name }
    }

    /// Returns the names of currently registered folder tools
    public var folderToolNames: [String] {
        _folderToolNames
    }

    /// Whether folder tools are currently registered
    public var hasFolderTools: Bool {
        currentFolderContext != nil
    }

    /// Registers work-specific tools with the tool registry.
    /// Uses reference counting - safe to call multiple times from different sessions.
    public func registerTools() {
        referenceCount += 1
        guard referenceCount == 1 else { return }

        for tool in tools {
            ToolRegistry.shared.register(tool)
        }
    }

    /// Unregisters work-specific tools from the tool registry.
    /// Uses reference counting - only unregisters when last session leaves.
    public func unregisterTools() {
        guard referenceCount > 0 else { return }
        referenceCount -= 1
        guard referenceCount == 0 else { return }

        ToolRegistry.shared.unregister(names: toolNames)
        unregisterFolderTools()
    }

    /// Force unregisters all work tools regardless of reference count.
    /// Use for cleanup during app termination.
    public func forceUnregisterAll() {
        guard referenceCount > 0 else { return }
        ToolRegistry.shared.unregister(names: toolNames)
        referenceCount = 0
        unregisterFolderTools()
    }

    // MARK: - Folder Tool Registration

    /// Register folder-specific tools for the given context
    /// Called by WorkFolderContextService when folder is selected
    public func registerFolderTools(for context: WorkFolderContext) {
        // Unregister any existing folder tools first
        unregisterFolderTools()

        currentFolderContext = context

        // Build core tools (always)
        folderTools = WorkFolderToolFactory.buildCoreTools(rootPath: context.rootPath)

        // Add coding tools if known project type
        if context.projectType != .unknown {
            folderTools += WorkFolderToolFactory.buildCodingTools(rootPath: context.rootPath)
        }

        // Add git tools if git repo
        if context.isGitRepo {
            folderTools += WorkFolderToolFactory.buildGitTools(rootPath: context.rootPath)
        }

        _folderToolNames = folderTools.map { $0.name }
        for tool in folderTools {
            ToolRegistry.shared.register(tool)
        }
    }

    /// Unregister all folder tools
    /// Called by WorkFolderContextService when folder is cleared
    public func unregisterFolderTools() {
        guard !_folderToolNames.isEmpty else { return }
        ToolRegistry.shared.unregister(names: _folderToolNames)
        folderTools = []
        _folderToolNames = []
        currentFolderContext = nil
    }
}
