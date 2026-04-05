//
//  ToolConfigurationStore.swift
//  project2501
//
//  Persistence for ToolConfiguration
//

import Foundation

@MainActor
enum ToolConfigurationStore {
    /// When set, configuration reads/writes use this directory instead of the default path.
    static var overrideDirectory: URL?

    static func load() -> ToolConfiguration {
        let url = configurationFileURL()
        if FileManager.default.fileExists(atPath: url.path) {
            do {
                return try JSONDecoder().decode(ToolConfiguration.self, from: Data(contentsOf: url))
            } catch {
                print("[Project2501] Failed to load ToolConfiguration: \(error)")
            }
        }
        let defaults = ToolConfiguration()
        save(defaults)
        return defaults
    }

    static func save(_ configuration: ToolConfiguration) {
        let url = configurationFileURL()
        Project2501Paths.ensureExistsSilent(url.deletingLastPathComponent())
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(configuration).write(to: url, options: [.atomic])
        } catch {
            print("[Project2501] Failed to save ToolConfiguration: \(error)")
        }
    }

    private static func configurationFileURL() -> URL {
        if let dir = overrideDirectory {
            return dir.appendingPathComponent("tools.json")
        }
        return Project2501Paths.resolvePath(new: Project2501Paths.toolConfigFile(), legacy: "ToolConfiguration.json")
    }
}
