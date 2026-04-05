import Foundation
import Testing

@testable import Project2501Core

@Suite(.serialized)
@MainActor
struct WorkEngineResumeTests {
    @Test
    func provideClarification_resumesWithPreservedConversation() async throws {
        try await IssueManager.shared.initialize()
        let registry = ToolRegistry.shared
        registry.register(NoopResumeTestTool())
        registry.setEnabled(true, for: "noop_resume_test")
        defer { registry.unregister(names: ["noop_resume_test"]) }

        let chatEngine = RecordingWorkChatEngine(
            steps: [
                .tool("noop_resume_test", "{}"),
                .tool("request_clarification", #"{"question":"SQLite or PostgreSQL?"}"#),
                .tool("complete_task", #"{"summary":"done","success":true}"#),
            ]
        )
        let engine = WorkEngine(executionEngine: WorkExecutionEngine(chatEngine: chatEngine))

        let first = try await engine.run(
            query: "Build a database-backed API",
            model: "mock",
            systemPrompt: "Base",
            tools: [noopResumeToolSpec(), clarificationToolSpec(), completeTaskToolSpec()],
            executionMode: .none
        )

        #expect(first.isPaused)
        #expect(first.pauseReason == .clarificationNeeded(ClarificationRequest(question: "SQLite or PostgreSQL?")))

        let second = try await engine.provideClarification(
            issueId: first.issue.id,
            response: "PostgreSQL"
        )

        #expect(second.success)

        let lastMessages = await chatEngine.lastMessages()
        #expect(lastMessages.contains(where: { $0.role == "tool" && $0.content == "{}" }))
        #expect(
            lastMessages.contains(where: {
                $0.role == "user" && ($0.content?.contains("PostgreSQL") == true)
            })
        )
    }

    @Test
    func continueExecution_afterBudgetExhaustion_reusesConversation() async throws {
        try await IssueManager.shared.initialize()
        let originalConfig = ChatConfigurationStore.load()
        var limitedConfig = originalConfig
        limitedConfig.workMaxIterations = 1
        ChatConfigurationStore.save(limitedConfig)
        defer { ChatConfigurationStore.save(originalConfig) }

        let registry = ToolRegistry.shared
        registry.register(NoopResumeTestTool())
        registry.setEnabled(true, for: "noop_resume_test")
        defer { registry.unregister(names: ["noop_resume_test"]) }

        let chatEngine = RecordingWorkChatEngine(
            steps: [
                .tool("noop_resume_test", "{}"),
                .tool("complete_task", #"{"summary":"wrapped up","success":true}"#),
            ]
        )
        let engine = WorkEngine(executionEngine: WorkExecutionEngine(chatEngine: chatEngine))

        let first = try await engine.run(
            query: "Build and verify the service",
            model: "mock",
            systemPrompt: "Base",
            tools: [noopResumeToolSpec(), completeTaskToolSpec()],
            executionMode: .none
        )

        #expect(first.isPaused)
        #expect(first.pauseReason == .budgetExhausted)

        let second = try await engine.continueExecution(message: "Focus on the tests.")

        #expect(second.success)

        let lastMessages = await chatEngine.lastMessages()
        #expect(lastMessages.contains(where: { $0.role == "tool" && $0.content == "{}" }))
        #expect(
            lastMessages.contains(where: {
                $0.role == "user" && ($0.content?.contains("Focus on the tests.") == true)
            })
        )
    }

    @Test
    func persistedSession_restoresIntoFreshEngineAfterPause() async throws {
        try await IssueManager.shared.initialize()
        let originalConfig = ChatConfigurationStore.load()
        var limitedConfig = originalConfig
        limitedConfig.workMaxIterations = 1
        ChatConfigurationStore.save(limitedConfig)
        defer { ChatConfigurationStore.save(originalConfig) }

        let registry = ToolRegistry.shared
        registry.register(NoopResumeTestTool())
        registry.setEnabled(true, for: "noop_resume_test")
        defer { registry.unregister(names: ["noop_resume_test"]) }

        let firstChatEngine = RecordingWorkChatEngine(steps: [.tool("noop_resume_test", "{}")])
        let firstEngine = WorkEngine(executionEngine: WorkExecutionEngine(chatEngine: firstChatEngine))

        let paused = try await firstEngine.run(
            query: "Pause and recover this task",
            model: "mock",
            systemPrompt: "Base",
            tools: [noopResumeToolSpec(), completeTaskToolSpec()],
            executionMode: .none
        )

        #expect(paused.isPaused)
        #expect(paused.pauseReason == .budgetExhausted)

        let secondChatEngine = RecordingWorkChatEngine(
            steps: [.tool("complete_task", #"{"summary":"recovered","success":true}"#)]
        )
        let recoveredEngine = WorkEngine(executionEngine: WorkExecutionEngine(chatEngine: secondChatEngine))

        let restoredReason = await recoveredEngine.restorePersistedSessionIfNeeded(for: paused.issue.id)
        #expect(restoredReason == .budgetExhausted)

        let completed = try await recoveredEngine.continueExecution()
        #expect(completed.success)

        let lastMessages = await secondChatEngine.lastMessages()
        #expect(lastMessages.contains(where: { $0.role == "tool" && $0.content == "{}" }))
        #expect(
            lastMessages.contains(where: {
                $0.role == "user"
                    && ($0.content?.contains(WorkEngine.freshBudgetContinuation) == true)
            })
        )
    }
}

private actor RecordingWorkChatEngine: ChatEngineProtocol {
    enum Step {
        case tool(String, String)
    }

    private var steps: [Step]
    private var index = 0
    private var requests: [ChatCompletionRequest] = []

    init(steps: [Step]) {
        self.steps = steps
    }

    func streamChat(request: ChatCompletionRequest) async throws -> AsyncThrowingStream<String, Error> {
        requests.append(request)
        guard index < steps.count else {
            return AsyncThrowingStream { continuation in continuation.finish() }
        }

        let step = steps[index]
        index += 1
        switch step {
        case .tool(let name, let args):
            throw ServiceToolInvocation(toolName: name, jsonArguments: args)
        }
    }

    func completeChat(request _: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        throw NSError(domain: "WorkEngineResumeTests", code: 1)
    }

    func lastMessages() -> [ChatMessage] {
        requests.last?.messages ?? []
    }
}

private func completeTaskToolSpec() -> Tool {
    Tool(
        type: "function",
        function: ToolFunction(
            name: "complete_task",
            description: "Complete the task",
            parameters: .object(["type": .string("object")])
        )
    )
}

private func clarificationToolSpec() -> Tool {
    Tool(
        type: "function",
        function: ToolFunction(
            name: "request_clarification",
            description: "Clarify the task",
            parameters: .object(["type": .string("object")])
        )
    )
}

private func noopResumeToolSpec() -> Tool {
    Tool(
        type: "function",
        function: ToolFunction(
            name: "noop_resume_test",
            description: "No-op resume test tool.",
            parameters: .object(["type": .string("object")])
        )
    )
}

private struct NoopResumeTestTool: OsaurusTool {
    let name = "noop_resume_test"
    let description = "No-op resume test tool."
    let parameters: JSONValue? = .object(["type": .string("object")])

    func execute(argumentsJSON _: String) async throws -> String {
        "{}"
    }
}
