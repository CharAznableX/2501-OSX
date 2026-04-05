//
//  Configuration.swift
//  project2501
//
//  Service for reading CLI configuration including server port and tools directory paths.
//

import Foundation
import Project2501Repository

public struct Configuration {
    /// Root data directory for Project2501 (`~/.project2501/`)
    public static func root() -> URL {
        ToolsPaths.root()
    }

    /// The previous root directory before migration to `~/.project2501/`.
    private static func legacyRoot() -> URL {
        let fm = FileManager.default
        let supportDir = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return supportDir.appendingPathComponent("com.cuadralabs.project2501", isDirectory: true)
    }

    public static func resolveConfiguredPort() -> Int? {
        if let env = ProcessInfo.processInfo.environment["OSU_PORT"], let p = Int(env) {
            return p
        }

        let fm = FileManager.default
        let root = root()
        let oldRoot = legacyRoot()

        // Check ~/.project2501/config/server.json first, then legacy Application Support locations
        let candidates: [URL] = [
            root.appendingPathComponent("config/server.json"),
            root.appendingPathComponent("ServerConfiguration.json"),
            oldRoot.appendingPathComponent("config/server.json"),
            oldRoot.appendingPathComponent("ServerConfiguration.json"),
        ]

        guard let configURL = candidates.first(where: { fm.fileExists(atPath: $0.path) }) else {
            return nil
        }

        struct PartialConfig: Decodable { let port: Int? }
        do {
            let data = try Data(contentsOf: configURL)
            let cfg = try JSONDecoder().decode(PartialConfig.self, from: data)
            return cfg.port
        } catch {
            return nil
        }
    }

    public static func toolsRootDirectory() -> URL {
        ToolsPaths.toolsRootDirectory()
    }
}
