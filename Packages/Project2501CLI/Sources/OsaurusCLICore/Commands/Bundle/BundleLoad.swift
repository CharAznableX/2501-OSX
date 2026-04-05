//
//  BundleLoad.swift
//  project2501
//
//  Load and start an MCP Bundle (.mcpb file).
//

import Foundation
import MCP

enum BundleLoadError: Error, CustomStringConvertible {
    case missingPath
    case fileNotFound(String)
    case invalidExtension
    case extractionFailed(String)
    case missingManifest
    case invalidManifest(String)
    case serverLaunchFailed(String)
    case toolDiscoveryFailed(String)

    var description: String {
        switch self {
        case .missingPath:
            return "Error: Bundle path required\n  Usage: project2501 bundle load <path.mcpb>"
        case let .fileNotFound(path):
            return "Error: File not found: \(path)"
        case .invalidExtension:
            return "Error: File must have .mcpb extension"
        case let .extractionFailed(reason):
            return "Error: Failed to extract bundle: \(reason)"
        case .missingManifest:
            return "Error: manifest.json not found in bundle"
        case let .invalidManifest(reason):
            return "Error: Invalid manifest format: \(reason)"
        case let .serverLaunchFailed(reason):
            return "Error: Failed to start MCP server: \(reason)"
        case let .toolDiscoveryFailed(reason):
            return "Error: Failed to discover tools: \(reason)"
        }
    }
}

struct BundleLoad {
    static func execute(args: [String]) async {
        do {
            try await run(args: args)
        } catch {
            fputs("\(error)\n", stderr)
            exit(EXIT_FAILURE)
        }
    }

    private static func run(args: [String]) async throws {
        // Parse arguments
        var bundlePath: String?
        var displayName: String?
        var i = 0
        while i < args.count {
            let arg = args[i]
            if arg == "--name" {
                if i + 1 < args.count {
                    displayName = args[i + 1]
                    i += 2
                } else {
                    throw BundleLoadError.missingPath
                }
            } else if !arg.hasPrefix("--") {
                bundlePath = arg
                i += 1
            } else {
                i += 1
            }
        }

        guard let path = bundlePath else {
            throw BundleLoadError.missingPath
        }

        // Validate file
        guard FileManager.default.fileExists(atPath: path) else {
            throw BundleLoadError.fileNotFound(path)
        }

        guard path.lowercased().hasSuffix(".mcpb") else {
            throw BundleLoadError.invalidExtension
        }

        // Extract bundle
        let bundleInfo = try MCPBundleManager.extract(path)
        defer { bundleInfo.cleanup() }

        // Parse manifest
        let manifest = try bundleInfo.parseManifest()

        // Display bundle info
        print("Bundle: \(displayName ?? manifest.displayName ?? manifest.name)")
        print("Version: \(manifest.version)")
        if let description = manifest.description {
            print("Description: \(description)")
        }
        print("")

        // Launch server
        let serverInfo = try await bundleInfo.launchServer(workingDirectory: bundleInfo.extractedPath)
        defer {
            serverInfo.shutdown()
        }

        // Discover tools via MCP SDK
        do {
            let tools = try await serverInfo.discoverTools()

            if tools.isEmpty {
                print("No tools discovered.")
            } else {
                print("Discovered \(tools.count) tool(s):")
                for tool in tools {
                    print("  - \(tool.name): \(tool.description ?? "")")
                }
            }
        } catch {
            print("Warning: Could not discover tools: \(error)")
        }

        print("")
        print("Server running. Press Ctrl+C to stop.")
        print("")

        // Keep alive until interrupt
        signal(SIGINT) { _ in
            exit(EXIT_SUCCESS)
        }

        while true {
            try await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }
}
