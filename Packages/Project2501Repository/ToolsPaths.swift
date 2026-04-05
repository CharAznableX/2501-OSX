//
//  ToolsPaths.swift
//  project2501
//
//  Path management for plugin storage and specifications.
//  Mirrors OsaurusPaths.root() for use in the Project2501Repository package.
//

import Foundation

public enum ToolsPaths {
    /// Optional root directory override for tests
    /// Note: nonisolated(unsafe) since this is only set during test setup before any concurrent access
    public nonisolated(unsafe) static var overrideRoot: URL?

    /// The root data directory for Osaurus: `~/.project2501/`
    public static func root() -> URL {
        if let override = overrideRoot {
            return override
        }
        let fm = FileManager.default
        return fm.homeDirectoryForCurrentUser.appendingPathComponent(".project2501", isDirectory: true)
    }

    /// Tools directory (plugins)
    /// `~/.project2501/Tools/`
    public static func toolsRootDirectory() -> URL {
        root().appendingPathComponent("Tools", isDirectory: true)
    }

    /// Plugin specifications directory
    /// `~/.project2501/PluginSpecs/`
    public static func pluginSpecsRoot() -> URL {
        root().appendingPathComponent("PluginSpecs", isDirectory: true)
    }

    /// Ensures a directory exists, creating it if necessary
    /// - Parameter url: The directory URL to ensure exists
    public static func ensureExists(_ url: URL) {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        if !fm.fileExists(atPath: url.path, isDirectory: &isDir) {
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
    }
}
