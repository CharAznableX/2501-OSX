//
//  Project2501Paths.swift
//  project2501
//
//  Centralized path management for all Project2501 app data.
//  Provides consistent directory structure across all components.
//

import Foundation

/// Centralized path management for all Project2501 app data.
/// All stores and services should use this module for path resolution.
public enum Project2501Paths {
    /// Optional root directory override for tests
    /// Note: nonisolated(unsafe) since this is only set during test setup before any concurrent access
    public nonisolated(unsafe) static var overrideRoot: URL?

    // MARK: - Root Directory

    private static let defaultRoot: URL = {
        let fm = FileManager.default
        let newRoot = fm.homeDirectoryForCurrentUser.appendingPathComponent(".project2501", isDirectory: true)
        let supportDir = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let oldRoot = supportDir.appendingPathComponent("com.cuadralabs.project2501", isDirectory: true)

        // Copy data from old Application Support on first access (never deletes the original).
        if fm.fileExists(atPath: oldRoot.path) {
            if !fm.fileExists(atPath: newRoot.path) {
                do {
                    try fm.copyItem(at: oldRoot, to: newRoot)
                    print("[Project2501] Copied data from \(oldRoot.path) to \(newRoot.path)")
                    return newRoot
                } catch {
                    print("[Project2501] Copy failed, falling back to merge: \(error)")
                }
            }
            mergeDirectory(from: oldRoot, into: newRoot)
            print("[Project2501] Merged data from \(oldRoot.path) into \(newRoot.path)")
        }

        return newRoot
    }()

    /// The root data directory for Project2501: `~/.project2501/`
    public static func root() -> URL {
        if let override = overrideRoot {
            return override
        }
        return defaultRoot
    }

    // MARK: - Directory Paths

    /// Configuration files directory
    public static func config() -> URL {
        root().appendingPathComponent("config", isDirectory: true)
    }

    /// Voice-related configuration directory
    public static func voiceConfig() -> URL {
        config().appendingPathComponent("voice", isDirectory: true)
    }

    /// Provider configurations directory
    public static func providers() -> URL {
        root().appendingPathComponent("providers", isDirectory: true)
    }

    /// Agents directory
    public static func agents() -> URL {
        root().appendingPathComponent("agents", isDirectory: true)
    }

    /// Themes directory
    public static func themes() -> URL {
        root().appendingPathComponent("themes", isDirectory: true)
    }

    /// Chat sessions directory
    public static func sessions() -> URL {
        root().appendingPathComponent("sessions", isDirectory: true)
    }

    /// Schedules directory
    public static func schedules() -> URL {
        root().appendingPathComponent("schedules", isDirectory: true)
    }

    /// Watchers directory
    public static func watchers() -> URL {
        root().appendingPathComponent("watchers", isDirectory: true)
    }

    /// Runtime state directory
    public static func runtime() -> URL {
        root().appendingPathComponent("runtime", isDirectory: true)
    }

    /// Cache directory
    public static func cache() -> URL {
        root().appendingPathComponent("cache", isDirectory: true)
    }

    /// Skills directory
    public static func skills() -> URL {
        root().appendingPathComponent("skills", isDirectory: true)
    }

    /// Artifacts directory
    public static func artifacts() -> URL {
        root().appendingPathComponent("artifacts", isDirectory: true)
    }

    /// Work data directory
    public static func workData() -> URL {
        root().appendingPathComponent("work", isDirectory: true)
    }

    /// Memory system data directory
    public static func memory() -> URL {
        root().appendingPathComponent("memory", isDirectory: true)
    }

    /// Methods system data directory
    public static func methods() -> URL {
        root().appendingPathComponent("methods", isDirectory: true)
    }

    /// Tool index data directory
    public static func toolIndex() -> URL {
        root().appendingPathComponent("tool-index", isDirectory: true)
    }

    /// Plugin binaries directory (`~/.project2501/Tools/`)
    public static func tools() -> URL {
        root().appendingPathComponent("Tools", isDirectory: true)
    }

    /// Plugin specifications directory (`~/.project2501/PluginSpecs/`)
    public static func toolSpecs() -> URL {
        root().appendingPathComponent("PluginSpecs", isDirectory: true)
    }

    /// Central sandbox plugin library (`~/.project2501/sandbox-plugins/`)
    public static func sandboxPluginLibrary() -> URL {
        root().appendingPathComponent("sandbox-plugins", isDirectory: true)
    }

    // MARK: - Container / Sandbox Paths

    /// Container root: `~/.project2501/container/`
    public static func container() -> URL {
        root().appendingPathComponent("container", isDirectory: true)
    }

    /// Kernel binary directory: `~/.project2501/container/kernel/`
    public static func containerKernelDir() -> URL {
        container().appendingPathComponent("kernel", isDirectory: true)
    }

    /// Path to the Linux kernel binary
    public static func containerKernelFile() -> URL {
        containerKernelDir().appendingPathComponent("vmlinux")
    }

    /// Path to the init filesystem image: `~/.project2501/container/initfs.ext4`
    public static func containerInitFSFile() -> URL {
        container().appendingPathComponent("initfs.ext4")
    }

    /// Mounted as `/workspace` inside the container
    public static func containerWorkspace() -> URL {
        container().appendingPathComponent("workspace", isDirectory: true)
    }

    /// Per-agent workspace directories inside the container workspace
    public static func containerAgentsDir() -> URL {
        containerWorkspace().appendingPathComponent("agents", isDirectory: true)
    }

    /// A specific agent's workspace directory (host-side path)
    public static func containerAgentDir(_ agentName: String) -> URL {
        containerAgentsDir().appendingPathComponent(agentName, isDirectory: true)
    }

    /// Shared workspace readable by all agents
    public static func containerSharedDir() -> URL {
        containerWorkspace().appendingPathComponent("shared", isDirectory: true)
    }

    // MARK: - Shared Artifacts

    /// Root directory for all shared artifacts: `~/.project2501/artifacts/`
    public static func artifactsDir() -> URL {
        root().appendingPathComponent("artifacts", isDirectory: true)
    }

    /// Per-context artifacts directory: `~/.project2501/artifacts/{contextId}/`
    public static func contextArtifactsDir(contextId: String) -> URL {
        artifactsDir().appendingPathComponent(contextId, isDirectory: true)
    }

    /// In-container absolute path for an agent's home directory
    public static func inContainerAgentHome(_ agentName: String) -> String {
        "/workspace/agents/\(agentName)"
    }

    /// In-container absolute path for a plugin directory
    public static func inContainerPluginDir(_ agentName: String, _ pluginName: String) -> String {
        "/workspace/agents/\(agentName)/plugins/\(pluginName)"
    }

    // MARK: - Configuration Files

    public static func chatConfigFile() -> URL { config().appendingPathComponent("chat.json") }
    public static func serverConfigFile() -> URL { config().appendingPathComponent("server.json") }
    public static func toolConfigFile() -> URL { config().appendingPathComponent("tools.json") }
    public static func toastConfigFile() -> URL { config().appendingPathComponent("toast.json") }
    public static func sandboxConfigFile() -> URL { config().appendingPathComponent("sandbox.json") }
    public static func speechConfigFile() -> URL { voiceConfig().appendingPathComponent("speech.json") }
    public static func vadConfigFile() -> URL { voiceConfig().appendingPathComponent("vad.json") }
    public static func transcriptionConfigFile() -> URL { voiceConfig().appendingPathComponent("transcription.json") }
    public static func remoteProviderConfigFile() -> URL { providers().appendingPathComponent("remote.json") }
    public static func mcpProviderConfigFile() -> URL { providers().appendingPathComponent("mcp.json") }
    public static func workDatabaseFile() -> URL { workData().appendingPathComponent("work.db") }
    public static func memoryDatabaseFile() -> URL { memory().appendingPathComponent("memory.sqlite") }
    public static func methodsDatabaseFile() -> URL { methods().appendingPathComponent("methods.sqlite") }
    public static func toolIndexDatabaseFile() -> URL { toolIndex().appendingPathComponent("tool_index.sqlite") }
    public static func memoryConfigFile() -> URL { config().appendingPathComponent("memory.json") }
    public static func relayConfigFile() -> URL { config().appendingPathComponent("relay.json") }

    // MARK: - File Path Helpers

    public static func agentFile(for id: UUID) -> URL {
        agents().appendingPathComponent("\(id.uuidString).json")
    }

    public static func themeFile(for id: UUID) -> URL {
        themes().appendingPathComponent("\(id.uuidString).json")
    }

    public static func sessionFile(for id: UUID) -> URL {
        sessions().appendingPathComponent("\(id.uuidString).json")
    }

    public static func scheduleFile(for id: UUID) -> URL {
        schedules().appendingPathComponent("\(id.uuidString).json")
    }

    public static func watcherFile(for id: UUID) -> URL {
        watchers().appendingPathComponent("\(id.uuidString).json")
    }

    public static func pluginDirectory(for pluginId: String) -> URL {
        tools().appendingPathComponent(pluginId, isDirectory: true)
    }

    /// Per-plugin data directory for sandboxed SQLite storage
    public static func pluginDataDirectory(for pluginId: String) -> URL {
        pluginDirectory(for: pluginId).appendingPathComponent("data", isDirectory: true)
    }

    /// Per-plugin SQLite database file
    public static func pluginDatabaseFile(for pluginId: String) -> URL {
        pluginDataDirectory(for: pluginId).appendingPathComponent("data.db")
    }

    public static func runtimeInstance(_ instanceId: String) -> URL {
        runtime().appendingPathComponent(instanceId, isDirectory: true)
    }

    // MARK: - Legacy Resolution

    /// Resolves a path, preferring the legacy location if it exists and the new location doesn't.
    public static func resolvePath(new newPath: URL, legacy legacyName: String) -> URL {
        let legacyPath = root().appendingPathComponent(legacyName)
        let fm = FileManager.default
        if fm.fileExists(atPath: legacyPath.path) && !fm.fileExists(atPath: newPath.path) {
            return legacyPath
        }
        return newPath
    }

    // MARK: - Directory Creation

    /// Ensures a directory exists, creating it if necessary
    public static func ensureExists(_ url: URL) throws {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        if !fm.fileExists(atPath: url.path, isDirectory: &isDir) {
            try fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
    }

    /// Ensures a directory exists (non-throwing version)
    public static func ensureExistsSilent(_ url: URL) {
        try? ensureExists(url)
    }

    // MARK: - File Utilities

    /// Computes the total size of all files in a directory tree.
    public static func directorySize(at url: URL) -> Int {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += size
            }
        }
        return total
    }

    // MARK: - Migration

    /// Recursively copy the contents of `src` into `dest` (never deletes from `src`).
    /// When both source and destination files exist, the newer one wins.
    private static func mergeDirectory(from src: URL, into dest: URL) {
        let fm = FileManager.default
        ensureExistsSilent(dest)
        let keys: Set<URLResourceKey> = [.isDirectoryKey, .contentModificationDateKey]
        guard let contents = try? fm.contentsOfDirectory(at: src, includingPropertiesForKeys: Array(keys)) else {
            return
        }
        for item in contents {
            let target = dest.appendingPathComponent(item.lastPathComponent)
            let isDir = (try? item.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false

            if fm.fileExists(atPath: target.path) {
                if isDir {
                    mergeDirectory(from: item, into: target)
                } else {
                    let srcDate =
                        (try? item.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                        ?? .distantPast
                    let destDate =
                        (try? target.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                        ?? .distantPast
                    if srcDate > destDate {
                        try? fm.removeItem(at: target)
                        try? fm.copyItem(at: item, to: target)
                    }
                }
            } else {
                try? fm.copyItem(at: item, to: target)
            }
        }
    }

}
