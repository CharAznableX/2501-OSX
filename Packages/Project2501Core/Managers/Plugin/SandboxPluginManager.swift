//
//  SandboxPluginManager.swift
//  project2501
//
//  Manages installation, setup, and lifecycle of sandbox plugins
//  that run inside the shared Linux container.
//

import Foundation
import Combine

@MainActor
public final class SandboxPluginManager: ObservableObject {
    public static let shared = SandboxPluginManager()

    /// agentId -> list of installed sandbox plugins
    @Published public var installedPlugins: [String: [InstalledSandboxPlugin]] = [:]
    @Published public var installProgress: [String: InstallProgress] = [:]

    public struct InstallProgress: Sendable {
        public let pluginName: String
        public let phase: String
        public let agentId: String
    }

    private init() {
        loadAllInstalled()
    }

    // MARK: - Install

    public func install(plugin: SandboxPlugin, for agentId: String) async throws {
        let errors = plugin.validateFilePaths()
        guard errors.isEmpty else {
            throw SandboxPluginError.invalidPlugin(errors.joined(separator: "; "))
        }

        let agentName = SandboxAgentProvisioner.linuxName(for: agentId)
        let key = progressKey(plugin: plugin.id, agent: agentId)

        setProgress(
            key: key,
            InstallProgress(
                pluginName: plugin.name,
                phase: "Preparing...",
                agentId: agentId
            )
        )

        var installed = InstalledSandboxPlugin(
            plugin: plugin,
            agentId: agentId,
            status: .installing,
            sourceContentHash: plugin.contentHash
        )

        updateInstalled(installed, for: agentId)

        do {
            setProgress(
                key: key,
                InstallProgress(
                    pluginName: plugin.name,
                    phase: "Provisioning agent sandbox...",
                    agentId: agentId
                )
            )
            try await SandboxAgentProvisioner.shared.ensureProvisioned(agentId: agentId)

            if plugin.dependencies != nil {
                setProgress(
                    key: key,
                    InstallProgress(
                        pluginName: plugin.name,
                        phase: "Installing system packages...",
                        agentId: agentId
                    )
                )
            }
            try await installSystemDependencies(for: plugin, agentName: agentName)

            setProgress(
                key: key,
                InstallProgress(
                    pluginName: plugin.name,
                    phase: "Creating plugin directory...",
                    agentId: agentId
                )
            )
            let pluginDir = Project2501Paths.inContainerPluginDir(agentName, plugin.id)
            let mkdirResult = try await SandboxManager.shared.execAsAgent(
                agentName,
                command: "mkdir -p \(pluginDir)"
            )
            guard mkdirResult.succeeded else {
                throw SandboxPluginError.setupFailed("mkdir failed: \(mkdirResult.stderr)")
            }

            if let files = plugin.files {
                setProgress(
                    key: key,
                    InstallProgress(
                        pluginName: plugin.name,
                        phase: "Seeding files...",
                        agentId: agentId
                    )
                )
                let hostPluginDir = self.hostPluginDir(agentName: agentName, pluginId: plugin.id)
                for (path, content) in files {
                    let fullPath = hostPluginDir.appendingPathComponent(path)
                    let dir = fullPath.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                    try content.write(to: fullPath, atomically: true, encoding: .utf8)
                }
                // Fix ownership inside the container so the agent user can access the files
                _ = try await SandboxManager.shared.execAsRoot(
                    command: "chown -R agent-\(agentName):agent-\(agentName) \(pluginDir)"
                )
            }

            if plugin.setup != nil {
                setProgress(
                    key: key,
                    InstallProgress(
                        pluginName: plugin.name,
                        phase: "Running setup...",
                        agentId: agentId
                    )
                )
            }
            try await runSetupCommand(for: plugin, agentName: agentName, agentId: agentId)

            installed.status = .ready
            updateInstalled(installed, for: agentId)
            saveInstalled(for: agentId)
            clearProgress(key: key)

            NotificationCenter.default.post(
                name: .sandboxPluginInstalled,
                object: nil,
                userInfo: [
                    "pluginId": plugin.id,
                    "agentId": agentId,
                ]
            )

        } catch {
            installed.status = .failed
            updateInstalled(installed, for: agentId)
            saveInstalled(for: agentId)
            clearProgress(key: key)
            throw error
        }
    }

    // MARK: - Uninstall

    public func uninstall(pluginId: String, from agentId: String) async throws {
        guard var list = installedPlugins[agentId],
            let index = list.firstIndex(where: { $0.id == pluginId })
        else { return }

        list[index].status = .uninstalling
        installedPlugins[agentId] = list

        let agentName = SandboxAgentProvisioner.linuxName(for: agentId)
        let pluginDir = Project2501Paths.inContainerPluginDir(agentName, pluginId)

        if await SandboxManager.shared.status().isRunning {
            _ = try? await SandboxManager.shared.execAsAgent(
                agentName,
                command: "rm -rf '\(pluginDir)'"
            )
        }

        try? FileManager.default.removeItem(at: hostPluginDir(agentName: agentName, pluginId: pluginId))

        list.remove(at: index)
        installedPlugins[agentId] = list
        saveInstalled(for: agentId)

        NotificationCenter.default.post(
            name: .sandboxPluginUninstalled,
            object: nil,
            userInfo: [
                "pluginId": pluginId,
                "agentId": agentId,
            ]
        )
    }

    // MARK: - Reinstall

    public func reinstall(plugin: SandboxPlugin, for agentId: String) async throws {
        try await uninstall(pluginId: plugin.id, from: agentId)
        try await install(plugin: plugin, for: agentId)
    }

    // MARK: - Verify & Repair

    /// Re-installs ephemeral dependencies and re-runs setup for all `.ready`
    /// plugins across all agents. Call after the container restarts so that
    /// system packages and setup side effects lost with the rootfs are restored.
    public func verifyAndRepairAllPlugins() async {
        guard await SandboxManager.shared.status().isRunning else { return }

        let snapshot = installedPlugins.flatMap { agentId, plugins in
            plugins.filter { $0.status == .ready }.map { (agentId, $0.plugin) }
        }
        guard !snapshot.isEmpty else { return }

        NSLog("[SandboxPluginManager] Verifying \(snapshot.count) installed plugin(s) after container start")

        for (agentId, plugin) in snapshot {
            await repairPlugin(plugin, for: agentId)
        }
    }

    /// Re-installs system dependencies and re-runs the setup command for a
    /// single plugin. If VirtioFS files are intact, only restores ephemeral
    /// deps. If files are missing, does a full reinstall.
    @discardableResult
    public func repairPlugin(_ plugin: SandboxPlugin, for agentId: String) async -> Bool {
        let agentName = SandboxAgentProvisioner.linuxName(for: agentId)
        let key = progressKey(plugin: plugin.id, agent: agentId)

        setProgress(
            key: key,
            InstallProgress(pluginName: plugin.name, phase: "Verifying plugin...", agentId: agentId)
        )
        defer { clearProgress(key: key) }

        guard hostFilesIntact(plugin: plugin, agentName: agentName) else {
            NSLog("[SandboxPluginManager] Plugin files missing for '\(plugin.id)' (agent \(agentId)), reinstalling")
            do {
                try await reinstall(plugin: plugin, for: agentId)
                return true
            } catch {
                NSLog("[SandboxPluginManager] Reinstall failed for '\(plugin.id)': \(error.localizedDescription)")
                markPluginFailed(plugin.id, for: agentId)
                return false
            }
        }

        do {
            try await SandboxAgentProvisioner.shared.ensureProvisioned(agentId: agentId)

            if plugin.dependencies != nil {
                setProgress(
                    key: key,
                    InstallProgress(pluginName: plugin.name, phase: "Restoring system packages...", agentId: agentId)
                )
            }
            try await installSystemDependencies(for: plugin, agentName: agentName)

            if plugin.setup != nil {
                setProgress(
                    key: key,
                    InstallProgress(pluginName: plugin.name, phase: "Re-running setup...", agentId: agentId)
                )
            }
            try await runSetupCommand(for: plugin, agentName: agentName, agentId: agentId)

            return true
        } catch {
            NSLog("[SandboxPluginManager] Repair failed for '\(plugin.id)': \(error.localizedDescription)")
            markPluginFailed(plugin.id, for: agentId)
            return false
        }
    }

    private func markPluginFailed(_ pluginId: String, for agentId: String) {
        guard var list = installedPlugins[agentId],
            let index = list.firstIndex(where: { $0.id == pluginId })
        else { return }
        list[index].status = .failed
        installedPlugins[agentId] = list
        saveInstalled(for: agentId)
    }

    // MARK: - Outdated Detection

    public func isOutdated(pluginId: String, agentId: String) -> Bool {
        guard let installed = plugin(id: pluginId, for: agentId),
            let libraryPlugin = SandboxPluginLibrary.shared.plugin(id: pluginId)
        else { return false }
        return installed.sourceContentHash != libraryPlugin.contentHash
    }

    public func hasAnyOutdated(pluginId: String, validAgentIds: Set<String>) -> Bool {
        installedPlugins.contains { agentId, plugins in
            validAgentIds.contains(agentId)
                && plugins.contains { $0.id == pluginId }
                && isOutdated(pluginId: pluginId, agentId: agentId)
        }
    }

    // MARK: - On-Demand Provisioning

    /// Ensures a plugin is installed and ready for a given agent.
    /// Verifies host-side files are intact (fast local FS check). If anything
    /// is missing — directory, files, or metadata — does a full reinstall.
    public func ensureReady(pluginId: String, plugin: SandboxPlugin, for agentId: String) async -> Bool {
        let agentName = SandboxAgentProvisioner.linuxName(for: agentId)
        let existing = self.plugin(id: pluginId, for: agentId)

        if existing?.status == .ready,
            hostFilesIntact(plugin: plugin, agentName: agentName)
        {
            return true
        }

        do {
            if existing != nil {
                NSLog("[SandboxPluginManager] Plugin '\(pluginId)' stale — reinstalling")
                try await uninstall(pluginId: pluginId, from: agentId)
            }
            try await install(plugin: plugin, for: agentId)
            return true
        } catch {
            NSLog("[SandboxPluginManager] On-demand provision failed for '\(pluginId)': \(error.localizedDescription)")
            return false
        }
    }

    private func hostFilesIntact(plugin: SandboxPlugin, agentName: String) -> Bool {
        let dir = hostPluginDir(agentName: agentName, pluginId: plugin.id)
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.path) else { return false }
        guard let files = plugin.files, !files.isEmpty else { return true }
        return files.allSatisfy { (path, _) in
            fm.fileExists(atPath: dir.appendingPathComponent(path).path)
        }
    }

    // MARK: - Global Plugin Listing

    /// Returns deduplicated plugin definitions across all agents (by plugin ID).
    public func allUniquePlugins() -> [SandboxPlugin] {
        var seen = Set<String>()
        return installedPlugins.values.flatMap { $0 }
            .filter { $0.status == .ready }
            .compactMap { installed in
                guard seen.insert(installed.plugin.id).inserted else { return nil }
                return installed.plugin
            }
    }

    // MARK: - Query

    public func plugins(for agentId: String) -> [InstalledSandboxPlugin] {
        installedPlugins[agentId] ?? []
    }

    public func plugin(id: String, for agentId: String) -> InstalledSandboxPlugin? {
        installedPlugins[agentId]?.first { $0.id == id }
    }

    // MARK: - Persistence & Cleanup

    /// Remove installed-plugin records for agents that no longer exist.
    public func purgeStaleAgents(validAgentIds: Set<String>) {
        let stale = Set(installedPlugins.keys).subtracting(validAgentIds)
        guard !stale.isEmpty else { return }
        for agentId in stale {
            installedPlugins.removeValue(forKey: agentId)
            try? FileManager.default.removeItem(at: storeFile(for: agentId))
        }
    }

    @discardableResult
    public func removeAgentState(for agentId: String) -> Bool {
        let agentName = SandboxAgentProvisioner.linuxName(for: agentId)
        let storeDir = storeDirectory(for: agentId)
        let hostPluginsDir = Project2501Paths.containerAgentDir(agentName)
            .appendingPathComponent("plugins", isDirectory: true)

        let hadInstalledState = installedPlugins.removeValue(forKey: agentId) != nil
        let progressKeys = installProgress.keys.filter { $0.hasPrefix("\(agentId):") }
        for key in progressKeys {
            installProgress.removeValue(forKey: key)
        }

        let fm = FileManager.default
        let hadStoreDir = fm.fileExists(atPath: storeDir.path)
        let hadHostPluginsDir = fm.fileExists(atPath: hostPluginsDir.path)
        if hadStoreDir {
            try? fm.removeItem(at: storeDir)
        }
        if hadHostPluginsDir {
            try? fm.removeItem(at: hostPluginsDir)
        }

        return hadInstalledState || hadStoreDir || hadHostPluginsDir || !progressKeys.isEmpty
    }

    private func storeDirectory(for agentId: String) -> URL {
        Project2501Paths.agents()
            .appendingPathComponent(agentId, isDirectory: true)
            .appendingPathComponent("sandbox-plugins", isDirectory: true)
    }

    private func storeFile(for agentId: String) -> URL {
        storeDirectory(for: agentId).appendingPathComponent("installed.json")
    }

    private func loadAllInstalled() {
        let fm = FileManager.default
        let agentsDir = Project2501Paths.agents()
        guard
            let agentDirs = try? fm.contentsOfDirectory(
                at: agentsDir,
                includingPropertiesForKeys: [.isDirectoryKey]
            )
        else { return }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        for dir in agentDirs {
            let agentId = dir.lastPathComponent
            let file = storeFile(for: agentId)
            guard let data = try? Data(contentsOf: file),
                let plugins = try? decoder.decode([InstalledSandboxPlugin].self, from: data)
            else { continue }
            installedPlugins[agentId] = plugins
        }
    }

    private func saveInstalled(for agentId: String) {
        let dir = storeDirectory(for: agentId)
        Project2501Paths.ensureExistsSilent(dir)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let plugins = installedPlugins[agentId] ?? []
        guard let data = try? encoder.encode(plugins) else { return }
        try? data.write(to: storeFile(for: agentId), options: .atomic)
    }

    // MARK: - Helpers

    private func updateInstalled(_ plugin: InstalledSandboxPlugin, for agentId: String) {
        var list = installedPlugins[agentId] ?? []
        if let index = list.firstIndex(where: { $0.id == plugin.id }) {
            list[index] = plugin
        } else {
            list.append(plugin)
        }
        installedPlugins[agentId] = list
    }

    private func installSystemDependencies(for plugin: SandboxPlugin, agentName: String) async throws {
        guard let deps = plugin.dependencies, !deps.isEmpty else { return }
        let depList = deps.joined(separator: " ")
        let result = try await SandboxManager.shared.execAsRoot(
            command: "apk add --no-cache \(depList)",
            timeout: 300,
            streamToLogs: true,
            logSource: plugin.id
        )
        guard result.succeeded else {
            throw SandboxPluginError.dependencyInstallFailed(result.stderr)
        }
    }

    private func runSetupCommand(for plugin: SandboxPlugin, agentName: String, agentId: String) async throws {
        guard let setup = plugin.setup else { return }
        let env = secretsEnvironment(agentId: agentId, pluginId: plugin.id)
        let result = try await SandboxManager.shared.execAsAgent(
            agentName,
            command: setup,
            pluginName: plugin.id,
            env: env,
            timeout: 300,
            streamToLogs: true,
            logSource: plugin.id
        )
        guard result.succeeded else {
            throw SandboxPluginError.setupFailed(result.stderr)
        }
    }

    private func secretsEnvironment(agentId: String, pluginId: String) -> [String: String] {
        guard let uuid = UUID(uuidString: agentId) else { return [:] }
        return AgentSecretsKeychain.mergedSecretsEnvironment(agentId: uuid, pluginId: pluginId)
    }

    private func hostPluginDir(agentName: String, pluginId: String) -> URL {
        Project2501Paths.containerWorkspace()
            .appendingPathComponent("agents/\(agentName)/plugins/\(pluginId)")
    }

    private func progressKey(plugin: String, agent: String) -> String {
        "\(agent):\(plugin)"
    }

    private func setProgress(key: String, _ progress: InstallProgress) {
        installProgress[key] = progress
    }

    private func clearProgress(key: String) {
        installProgress.removeValue(forKey: key)
    }

}

// MARK: - Notifications

extension Notification.Name {
    static let sandboxPluginInstalled = Notification.Name("SandboxPluginInstalled")
    static let sandboxPluginUninstalled = Notification.Name("SandboxPluginUninstalled")
}

// MARK: - Errors

public enum SandboxPluginError: Error, LocalizedError {
    case invalidPlugin(String)
    case dependencyInstallFailed(String)
    case setupFailed(String)
    case notInstalled

    public var errorDescription: String? {
        switch self {
        case .invalidPlugin(let msg): "Invalid plugin: \(msg)"
        case .dependencyInstallFailed(let msg): "Dependency install failed: \(msg)"
        case .setupFailed(let msg): "Setup failed: \(msg)"
        case .notInstalled: "Plugin is not installed"
        }
    }
}
