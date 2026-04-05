//
//  SandboxToolRegistrar.swift
//  project2501
//
//  Bridges the sandbox infrastructure with the ToolRegistry by
//  registering/unregistering sandbox tools in response to plugin
//  installs, and container lifecycle events.
//
//  Plugin tools are registered globally (agent-agnostic). Agent
//  identity is resolved at execution time via WorkExecutionContext.
//  Builtin sandbox tools remain per-agent.
//

import Combine
import Foundation

@MainActor
public final class SandboxToolRegistrar {
    public static let shared = SandboxToolRegistrar()

    private var observers: [NSObjectProtocol] = []
    private var statusCancellable: AnyCancellable?
    var provisionAgentOverride: ((UUID) async throws -> Void)?

    private init() {}

    // MARK: - Lifecycle

    /// Call once at app startup (after sandbox auto-start attempt).
    /// Sets up all notification observers and performs initial registration.
    public func start() {
        observers.append(
            NotificationCenter.default.addObserver(
                forName: .activeAgentChanged,
                object: nil,
                queue: .main
            ) { [weak self] _ in Task { @MainActor in await self?.handleAgentChanged() } }
        )

        observers.append(
            NotificationCenter.default.addObserver(
                forName: .sandboxPluginInstalled,
                object: nil,
                queue: .main
            ) { [weak self] note in
                let pluginId = note.userInfo?["pluginId"] as? String
                Task { @MainActor in await self?.handlePluginInstalled(pluginId: pluginId) }
            }
        )

        observers.append(
            NotificationCenter.default.addObserver(
                forName: .sandboxPluginUninstalled,
                object: nil,
                queue: .main
            ) { [weak self] note in
                let pluginId = note.userInfo?["pluginId"] as? String
                Task { @MainActor in await self?.handlePluginUninstalled(pluginId: pluginId) }
            }
        )

        statusCancellable = SandboxManager.State.shared.$status
            .removeDuplicates()
            .sink { [weak self] newStatus in
                Task { @MainActor in await self?.handleContainerStatusChanged(newStatus) }
            }

        observers.append(
            NotificationCenter.default.addObserver(
                forName: .agentUpdated,
                object: nil,
                queue: .main
            ) { [weak self] note in
                let agentId = note.object as? UUID
                Task { @MainActor in await self?.handleAgentUpdated(agentId: agentId) }
            }
        )

        Task { @MainActor in
            registerAllPluginTools()
            await registerTools(for: AgentManager.shared.activeAgent.id)
        }
    }

    // MARK: - Plugin Tools (Global)

    /// Register all sandbox plugin tools globally (agent-agnostic).
    /// Plugin tools are available to any agent and resolved at execution time.
    public func registerAllPluginTools() {
        let allPlugins = SandboxPluginManager.shared.allUniquePlugins()
        for plugin in allPlugins {
            ToolRegistry.shared.registerSandboxPluginTools(plugin: plugin)
        }
    }

    // MARK: - Builtin Tools (Per-Agent)

    /// Re-register builtin sandbox tools for a specific agent.
    /// This is the per-agent concern: provisioning + builtin tool registration.
    public func registerTools(for agentId: UUID) async {
        ToolRegistry.shared.unregisterAllBuiltinSandboxTools()

        let agent = AgentManager.shared.agent(for: agentId) ?? Agent.default
        let agentIdStr = agent.id.uuidString
        let agentName = SandboxAgentProvisioner.linuxName(for: agentIdStr)

        guard SandboxManager.State.shared.status == .running else { return }

        let execConfig = AgentManager.shared.effectiveAutonomousExec(for: agent.id)
        let plugins = SandboxPluginManager.shared.plugins(for: agentIdStr)
        let needsProvisioning = (execConfig?.enabled == true) || plugins.contains { $0.status == .ready }

        if needsProvisioning {
            do {
                try await ensureProvisioned(agentId: agent.id)
            } catch {
                NSLog("[SandboxToolRegistrar] Failed to provision agent sandbox: \(error.localizedDescription)")
                return
            }
        }

        BuiltinSandboxTools.register(
            agentId: agentIdStr,
            agentName: agentName,
            config: execConfig
        )
    }

    private func ensureProvisioned(agentId: UUID) async throws {
        if let provisionAgentOverride {
            try await provisionAgentOverride(agentId)
            return
        }
        try await SandboxAgentProvisioner.shared.ensureProvisioned(agentId: agentId)
    }

    // MARK: - Event Handlers

    private func handleAgentChanged() async {
        await registerTools(for: AgentManager.shared.activeAgent.id)
    }

    private func handleAgentUpdated(agentId: UUID?) async {
        guard agentId == nil || agentId == AgentManager.shared.activeAgent.id else { return }
        await registerTools(for: AgentManager.shared.activeAgent.id)
    }

    private func handlePluginInstalled(pluginId: String?) async {
        guard let pluginId else { return }
        guard let plugin = SandboxPluginLibrary.shared.plugin(id: pluginId) else { return }
        ToolRegistry.shared.registerSandboxPluginTools(plugin: plugin)
    }

    private func handlePluginUninstalled(pluginId: String?) async {
        guard let pluginId else { return }
        ToolRegistry.shared.unregisterSandboxPluginTools(pluginId: pluginId)
    }

    private func handleContainerStatusChanged(_ newStatus: ContainerStatus) async {
        if newStatus == .running {
            await SandboxPluginManager.shared.verifyAndRepairAllPlugins()
        }
        registerAllPluginTools()
        await registerTools(for: AgentManager.shared.activeAgent.id)
    }
}
