//
//  ToolsReset.swift
//  project2501
//
//  Resets plugin state to recover from crashes, corrupted data, or stale caches.
//

import Foundation
import Project2501Repository

public struct ToolsReset {
    public static func execute(args: [String]) {
        let flags = Set(args)
        let resetPlugins = flags.contains("--plugins") || flags.contains("--all")
        let resetAll = flags.contains("--all")
        let fm = FileManager.default

        let registryClone = ToolsPaths.pluginSpecsRoot().appendingPathComponent("central", isDirectory: true)
        if fm.fileExists(atPath: registryClone.path) {
            do {
                try fm.removeItem(at: registryClone)
                print("Reset plugin registry cache")
            } catch {
                fputs("Warning: failed to remove registry cache: \(error)\n", stderr)
            }
        }

        let toolsRoot = ToolsPaths.toolsRootDirectory()
        for name in [".quarantine", ".currently_loading"] {
            let url = toolsRoot.appendingPathComponent(name, isDirectory: false)
            if fm.fileExists(atPath: url.path) {
                try? fm.removeItem(at: url)
                print("Cleared \(name)")
            }
        }

        if resetPlugins {
            if let entries = try? fm.contentsOfDirectory(
                at: toolsRoot,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) {
                var count = 0
                for entry in entries where entry.hasDirectoryPath {
                    try? fm.removeItem(at: entry)
                    count += 1
                }
                print("Removed \(count) installed plugin(s)")
            }
        }

        if resetAll {
            let defaults = UserDefaults.standard
            defaults.removeObject(forKey: "LaunchGuard.startupInProgress")
            defaults.removeObject(forKey: "LaunchGuard.consecutiveCrashCount")
            defaults.synchronize()
            print("Reset crash recovery state")
        }

        AppControl.postDistributedNotification(name: "com.cuadralabs.project2501.control.toolsReload", userInfo: [:])
        print("Done. Restart Project2501 to apply changes.")
        exit(EXIT_SUCCESS)
    }
}
