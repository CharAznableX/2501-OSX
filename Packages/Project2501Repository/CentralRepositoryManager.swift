//
//  CentralRepositoryManager.swift
//  project2501
//
//  Manages the central plugin repository, including cloning, refreshing, and querying plugin specifications.
//

import Foundation

public struct CentralRepository {
    public let url: String
    public let branch: String?
    public init(url: String, branch: String? = nil) {
        self.url = url
        self.branch = branch
    }
}

public final class CentralRepositoryManager: @unchecked Sendable {
    public static let shared = CentralRepositoryManager()
    private init() {}

    public var central: CentralRepository = .init(
        url: "https://github.com/project2501-ai/project2501-tools.git",
        branch: nil
    )

    private func tapCloneDirectory() -> URL {
        ToolsPaths.pluginSpecsRoot().appendingPathComponent("central", isDirectory: true)
    }

    /// Refreshes the local clone of the central plugin repository.
    /// Returns `true` if git operations succeeded, `false` if they failed (e.g. network unreachable).
    /// When a fast-forward pull fails (e.g. after a force-push), the broken clone is
    /// deleted and re-cloned so the cache never stays permanently stale.
    @discardableResult
    public func refresh() -> Bool {
        let fm = FileManager.default
        let root = ToolsPaths.pluginSpecsRoot()
        if !fm.fileExists(atPath: root.path) {
            try? fm.createDirectory(at: root, withIntermediateDirectories: true)
        }
        let cloneDir = tapCloneDirectory()
        if fm.fileExists(atPath: cloneDir.appendingPathComponent(".git").path) {
            let (fetchStatus, _) = runGit(in: cloneDir, args: ["fetch", "--all", "--tags"])
            let (pullStatus, _) = runGit(in: cloneDir, args: ["pull", "--ff-only", "origin"])
            if let branch = central.branch {
                _ = runGit(in: cloneDir, args: ["checkout", branch])
            }
            if fetchStatus == 0 && pullStatus == 0 && validateCloneIntegrity(cloneDir) {
                return true
            }
            NSLog(
                "[Project2501] Registry pull failed or integrity check failed (fetch=%d, pull=%d), re-cloning",
                fetchStatus,
                pullStatus
            )
            try? fm.removeItem(at: cloneDir)
            return cloneFresh(root: root, cloneDir: cloneDir)
        } else {
            return cloneFresh(root: root, cloneDir: cloneDir)
        }
    }

    private func cloneFresh(root: URL, cloneDir: URL) -> Bool {
        var args = ["clone", "--depth", "1", central.url, cloneDir.path]
        if let branch = central.branch {
            args = ["clone", "--depth", "1", "--branch", branch, central.url, cloneDir.path]
        }
        let (status, _) = runGit(in: root, args: args)
        return status == 0
    }

    private func runGit(in directory: URL, args: [String]) -> (Int32, String) {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = ["git"] + args
        task.currentDirectoryURL = directory
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        do {
            try task.run()
            task.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let s = String(data: data, encoding: .utf8) ?? ""
            return (task.terminationStatus, s)
        } catch {
            return (-1, "\(error)")
        }
    }

    /// The `plugins/` directory must exist and contain at least one
    /// JSON file that decodes as a valid `PluginSpec`.
    private func validateCloneIntegrity(_ cloneDir: URL) -> Bool {
        let fm = FileManager.default
        let pluginsDir = cloneDir.appendingPathComponent("plugins", isDirectory: true)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: pluginsDir.path, isDirectory: &isDir), isDir.boolValue,
            let enumerator = fm.enumerator(
                at: pluginsDir,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        else {
            NSLog("[Project2501] Clone integrity check failed for %@", cloneDir.path)
            return false
        }
        for case let fileURL as URL in enumerator {
            if fileURL.pathExtension.lowercased() == "json",
                let data = try? Data(contentsOf: fileURL),
                (try? JSONDecoder().decode(PluginSpec.self, from: data)) != nil
            {
                return true
            }
        }
        NSLog("[Project2501] Clone integrity check failed: no valid spec JSON in %@", pluginsDir.path)
        return false
    }

    public func listAllSpecs() -> [PluginSpec] {
        let fm = FileManager.default
        var specs: [PluginSpec] = []
        let base = tapCloneDirectory().appendingPathComponent("plugins", isDirectory: true)
        guard
            let enumr = fm.enumerator(
                at: base,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return specs
        }
        for case let fileURL as URL in enumr {
            if fileURL.pathExtension.lowercased() == "json" {
                if let data = try? Data(contentsOf: fileURL),
                    let spec = try? JSONDecoder().decode(PluginSpec.self, from: data)
                {
                    specs.append(spec)
                }
            }
        }
        return specs
    }

    public func spec(for pluginId: String) -> PluginSpec? {
        return listAllSpecs().first(where: { $0.plugin_id == pluginId })
    }
}
