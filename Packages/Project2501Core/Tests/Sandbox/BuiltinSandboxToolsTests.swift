import Foundation
import Testing

@testable import Project2501Core

@Suite(.serialized)
struct BuiltinSandboxToolsTests {
    @Test @MainActor
    func sandboxPipInstall_bootstrapsPythonAndReturnsInstalledOnSuccess() async throws {
        let runner = MockSandboxToolCommandRunner(
            rootResults: [.init(stdout: "", stderr: "", exitCode: 0)],
            agentResults: [.init(stdout: "installed ok", stderr: "", exitCode: 0)]
        )

        let output = try await withRegisteredSandboxTools(runner: runner) {
            try await ToolRegistry.shared.execute(
                name: "sandbox_pip_install",
                argumentsJSON: #"{"packages":["flask","pytest"]}"#
            )
        }

        let payload = try #require(try parseJSON(output))
        let installed = try #require(payload["installed"] as? [String])
        #expect(installed == ["flask", "pytest"])
        #expect(payload["requested"] == nil)
        #expect(payload["exit_code"] as? Int == 0)

        let calls = await runner.calls
        #expect(calls.count == 2)
        #expect(calls[0] == .root("test -x /usr/bin/python3"))
        guard case .agent(_, let command) = calls[1] else {
            Issue.record("Expected agent install call")
            return
        }
        #expect(command.contains("/usr/bin/python3 -m venv"))
        #expect(command.contains(".venv/bin/python3"))
        #expect(command.contains("-m pip install flask pytest"))
    }

    @Test @MainActor
    func sandboxPipInstall_returnsErrorWhenPythonMissing() async throws {
        let runner = MockSandboxToolCommandRunner(
            rootResults: [.init(stdout: "", stderr: "", exitCode: 1)],
            agentResults: []
        )

        let output = try await withRegisteredSandboxTools(runner: runner) {
            try await ToolRegistry.shared.execute(
                name: "sandbox_pip_install",
                argumentsJSON: #"{"packages":["flask","pytest"]}"#
            )
        }

        let payload = try #require(try parseJSON(output))
        #expect(payload["error"] as? String == "python3 is not installed in the sandbox image")
        #expect(payload["installed"] == nil)

        let calls = await runner.calls
        #expect(calls.count == 1)
        #expect(calls[0] == .root("test -x /usr/bin/python3"))
    }

    @Test @MainActor
    func sandboxNpmInstall_returnsRequestedOnFailure() async throws {
        let runner = MockSandboxToolCommandRunner(
            rootResults: [.init(stdout: "", stderr: "", exitCode: 0)],
            agentResults: [.init(stdout: "", stderr: "npm: not found", exitCode: 127)]
        )

        let output = try await withRegisteredSandboxTools(runner: runner) {
            try await ToolRegistry.shared.execute(
                name: "sandbox_npm_install",
                argumentsJSON: #"{"packages":["vite"]}"#
            )
        }

        let payload = try #require(try parseJSON(output))
        #expect(payload["installed"] == nil)
        #expect(payload["requested"] as? [String] == ["vite"])
        #expect(payload["exit_code"] as? Int == 127)
        #expect((payload["output"] as? String)?.contains("npm: not found") == true)

        let calls = await runner.calls
        #expect(calls.count == 2)
        #expect(calls[0] == .root("test -x /usr/bin/node && test -x /usr/bin/npm"))
    }

    @Test @MainActor
    func sandboxRunScript_pythonUsesPythonInterpreter() async throws {
        let runner = MockSandboxToolCommandRunner(
            rootResults: [],
            agentResults: [],
            execResults: [.init(stdout: "ok", stderr: "", exitCode: 0)]
        )

        let output = try await withRegisteredSandboxTools(runner: runner) {
            try await ToolRegistry.shared.execute(
                name: "sandbox_run_script",
                argumentsJSON: #"{"language":"python","script":"print('hi')"}"#
            )
        }

        let payload = try #require(try parseJSON(output))
        #expect(payload["exit_code"] as? Int == 0)
        #expect((payload["output"] as? String)?.contains("ok") == true)

        let calls = await runner.calls
        guard case .exec(let user, let command, let env) = try #require(calls.first) else {
            Issue.record("Expected exec call")
            return
        }
        #expect(user == "agent-test-agent")
        #expect(command.contains("python3"))
        #expect(command.contains(".tmp/script_"))
        #expect(env["VIRTUAL_ENV"]?.contains(".venv") == true)
        #expect(env["PATH"]?.contains(".venv/bin") == true)
    }

    @Test @MainActor
    func sandboxExec_prefersAgentVenvOnPath() async throws {
        let runner = MockSandboxToolCommandRunner(
            rootResults: [],
            agentResults: [],
            execResults: [.init(stdout: "", stderr: "sh: pytest: not found", exitCode: 127)]
        )

        _ = try await withRegisteredSandboxTools(runner: runner) {
            try await ToolRegistry.shared.execute(
                name: "sandbox_exec",
                argumentsJSON: #"{"command":"pytest test_app.py -v"}"#
            )
        }

        let calls = await runner.calls
        guard case .exec(let user, let command, let env) = try #require(calls.first) else {
            Issue.record("Expected exec call")
            return
        }
        #expect(user == "agent-test-agent")
        #expect(command == "pytest test_app.py -v")
        #expect(env["VIRTUAL_ENV"]?.contains(".venv") == true)
        #expect(env["PATH"]?.contains(".venv/bin") == true)
    }

    @Test @MainActor
    func sandboxReadFile_supportsTailAndMaxChars() async throws {
        let runner = MockSandboxToolCommandRunner(
            rootResults: [],
            agentResults: [.init(stdout: "tail-output", stderr: "", exitCode: 0)]
        )

        let output = try await withRegisteredSandboxTools(runner: runner) {
            try await ToolRegistry.shared.execute(
                name: "sandbox_read_file",
                argumentsJSON: #"{"path":"build.log","tail_lines":20,"max_chars":1200}"#
            )
        }

        let payload = try #require(try parseJSON(output))
        #expect(payload["content"] as? String == "tail-output")
        #expect(payload["tail_lines"] as? Int == 20)
        #expect(payload["max_chars"] as? Int == 1200)

        let calls = await runner.calls
        guard case .agent(_, let command) = try #require(calls.first) else {
            Issue.record("Expected agent read call")
            return
        }
        #expect(command.contains("tail -n 20"))
        #expect(command.contains("| head -c 1200"))
    }
}

private actor MockSandboxToolCommandRunner: SandboxToolCommandRunning {
    enum Call: Equatable {
        case exec(String?, String, [String: String])
        case root(String)
        case agent(String, String)
    }

    private(set) var calls: [Call] = []
    private var execResults: [ContainerExecResult]
    private var rootResults: [ContainerExecResult]
    private var agentResults: [ContainerExecResult]

    init(
        rootResults: [ContainerExecResult],
        agentResults: [ContainerExecResult],
        execResults: [ContainerExecResult] = []
    ) {
        self.rootResults = rootResults
        self.agentResults = agentResults
        self.execResults = execResults
    }

    func exec(
        user: String?,
        command: String,
        env: [String: String],
        cwd _: String?,
        timeout _: TimeInterval,
        streamToLogs _: Bool,
        logSource _: String?
    ) async throws -> ContainerExecResult {
        calls.append(.exec(user, command, env))
        return execResults.isEmpty ? .init(stdout: "", stderr: "", exitCode: 0) : execResults.removeFirst()
    }

    func execAsRoot(
        command: String,
        timeout _: TimeInterval,
        streamToLogs _: Bool,
        logSource _: String?
    ) async throws -> ContainerExecResult {
        calls.append(.root(command))
        return rootResults.isEmpty ? .init(stdout: "", stderr: "", exitCode: 0) : rootResults.removeFirst()
    }

    func execAsAgent(
        _ agentName: String,
        command: String,
        pluginName _: String?,
        env _: [String: String],
        timeout _: TimeInterval,
        streamToLogs _: Bool,
        logSource _: String?
    ) async throws -> ContainerExecResult {
        calls.append(.agent(agentName, command))
        return agentResults.isEmpty ? .init(stdout: "", stderr: "", exitCode: 0) : agentResults.removeFirst()
    }
}

@MainActor
private func withRegisteredSandboxTools<T>(
    runner: some SandboxToolCommandRunning,
    _ body: () async throws -> T
) async throws -> T {
    let agentId = "test-agent"
    let config = AutonomousExecConfig(enabled: true, maxCommandsPerTurn: 10, commandTimeout: 30, pluginCreate: true)
    await SandboxToolCommandRunnerRegistry.shared.setRunner(runner)
    ToolRegistry.shared.unregisterAllSandboxTools()
    BuiltinSandboxTools.register(agentId: agentId, agentName: agentId, config: config)

    do {
        let result = try await body()
        ToolRegistry.shared.unregisterAllSandboxTools()
        await SandboxToolCommandRunnerRegistry.shared.reset()
        return result
    } catch {
        ToolRegistry.shared.unregisterAllSandboxTools()
        await SandboxToolCommandRunnerRegistry.shared.reset()
        throw error
    }
}

private func parseJSON(_ string: String) throws -> [String: Any]? {
    guard let data = string.data(using: .utf8) else { return nil }
    return try JSONSerialization.jsonObject(with: data) as? [String: Any]
}
