//
//  Version.swift
//  project2501
//
//  Command to display the Project2501 version and build number from environment variables.
//

import Foundation

public struct VersionCommand: Command {
    public static let name = "version"

    public static func execute(args: [String]) async {
        var versionString: String?
        var buildString: String?

        if let v = ProcessInfo.processInfo.environment["PROJECT2501_VERSION"] { versionString = v }
        if let b = ProcessInfo.processInfo.environment["PROJECT2501_BUILD_NUMBER"] { buildString = b }

        let output: String
        if let v = versionString, let b = buildString, !b.isEmpty {
            output = "Project2501 \(v) (\(b))"
        } else if let v = versionString {
            output = "Project2501 \(v)"
        } else {
            output = "Project2501 dev"
        }
        print(output)
        exit(EXIT_SUCCESS)
    }
}
