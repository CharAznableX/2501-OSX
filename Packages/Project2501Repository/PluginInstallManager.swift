//
//  PluginInstallManager.swift
//  project2501
//
//  Handles plugin installation workflow including download, verification, extraction, and receipt generation.
//

import Foundation
import CryptoKit

public enum PluginInstallError: Error, CustomStringConvertible, LocalizedError {
    case specNotFound(String)
    case resolutionFailed(String)
    case downloadFailed(String)
    case checksumMismatch
    case signatureRequired
    case signatureInvalid
    case authorKeyMismatch
    case unzipFailed(String)
    case layoutInvalid(String)

    public var description: String {
        switch self {
        case .specNotFound(let id): return "Spec not found: \(id)"
        case .resolutionFailed(let msg): return "Resolution failed: \(msg)"
        case .downloadFailed(let msg): return "Download failed: \(msg)"
        case .checksumMismatch: return "Checksum mismatch"
        case .signatureRequired: return "Plugin requires a minisign signature for installation"
        case .signatureInvalid: return "Signature verification failed"
        case .authorKeyMismatch:
            return
                "The signing key for this plugin has changed. Uninstall the plugin and reinstall it to accept the new key."
        case .unzipFailed(let msg): return "Unzip failed: \(msg)"
        case .layoutInvalid(let msg): return "Invalid artifact layout: \(msg)"
        }
    }

    public var errorDescription: String? { description }
}

public final class PluginInstallManager: @unchecked Sendable {
    public static let shared = PluginInstallManager()
    private init() {}

    public struct InstallResult: Sendable {
        public let receipt: PluginReceipt
        public let installDirectory: URL
        public let dylibURL: URL
        /// SKILL.md files found in the artifact and copied to the install directory
        public let skillFiles: [URL]
    }

    @discardableResult
    public func install(pluginId: String, preferredVersion: SemanticVersion? = nil) async throws -> InstallResult {
        let refreshed = CentralRepositoryManager.shared.refresh()
        guard let spec = CentralRepositoryManager.shared.spec(for: pluginId) else {
            if !refreshed {
                throw PluginInstallError.specNotFound("\(pluginId) (registry unavailable)")
            }
            throw PluginInstallError.specNotFound(pluginId)
        }

        // TOFU: when the author's signing key changes, remove the old version so
        // the new artifact is verified against the updated key from the registry.
        // This allows recovery from accidental bad-key pushes without manual uninstall.
        if let latest = InstalledPluginsStore.shared.latestInstalledVersion(pluginId: spec.plugin_id),
            let existing = InstalledPluginsStore.shared.receipt(pluginId: spec.plugin_id, version: latest),
            let existingKey = existing.public_keys?["minisign"],
            let newKey = spec.public_keys?["minisign"],
            existingKey != newKey
        {
            NSLog(
                "[Project2501] Signing key changed for %@ — removing old version %@ to accept new key",
                spec.plugin_id,
                latest.description
            )
            let oldVersionDir = PluginInstallManager.toolsVersionDirectory(pluginId: spec.plugin_id, version: latest)
            try? FileManager.default.removeItem(at: oldVersionDir)
        }

        let targetPlatform: Platform = .macos
        // Arm64 only per project policy
        let targetArch: CPUArch = .arm64

        let resolution: PluginResolution
        do {
            resolution = try spec.resolveBestVersion(
                targetPlatform: targetPlatform,
                targetArch: targetArch,
                minimumProject2501Version: nil,
                preferredVersion: preferredVersion
            )
        } catch {
            throw PluginInstallError.resolutionFailed("\(error)")
        }

        let artifact = resolution.artifact
        guard artifact.arch == CPUArch.arm64.rawValue else {
            throw PluginInstallError.resolutionFailed(
                "No arm64 artifact for \(pluginId) @ \(resolution.version.version)"
            )
        }
        guard let url = URL(string: artifact.url) else {
            throw PluginInstallError.downloadFailed("Invalid URL: \(artifact.url)")
        }

        let (tmpZip, bytes) = try await download(toTempFileFrom: url)
        defer { try? FileManager.default.removeItem(at: tmpZip) }

        let digest = SHA256.hash(data: bytes)
        let checksum = Data(digest).map { String(format: "%02x", $0) }.joined()
        if checksum.lowercased() != artifact.sha256.lowercased() {
            throw PluginInstallError.checksumMismatch
        }

        guard let ms = artifact.minisign, let pubKey = spec.public_keys?["minisign"] else {
            throw PluginInstallError.signatureRequired
        }
        do {
            _ = try MinisignVerifier.verify(publicKey: pubKey, signature: ms.signature, data: bytes)
        } catch {
            NSLog("[Project2501] Minisign verification failed for \(pluginId): \(error)")
            throw PluginInstallError.signatureInvalid
        }
        NSLog("[Project2501] Minisign signature verified for \(pluginId)")

        let tmpDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        try unzip(zipURL: tmpZip, to: tmpDir)

        guard let dylibURL = findFirstDylib(in: tmpDir) else {
            throw PluginInstallError.layoutInvalid("No .dylib found in archive")
        }

        let installDir = PluginInstallManager.toolsVersionDirectory(
            pluginId: spec.plugin_id,
            version: resolution.version.version
        )
        try ensureDirectoryExists(installDir)
        let finalDylibURL = installDir.appendingPathComponent(dylibURL.lastPathComponent, isDirectory: false)
        if FileManager.default.fileExists(atPath: finalDylibURL.path) {
            try FileManager.default.removeItem(at: finalDylibURL)
        }
        try FileManager.default.copyItem(at: dylibURL, to: finalDylibURL)

        // Copy any SKILL.md files found in the artifact
        let skillFileURLs = findSkillFiles(in: tmpDir)
        var installedSkillFiles: [URL] = []
        if !skillFileURLs.isEmpty {
            let skillsDir = installDir.appendingPathComponent("skills", isDirectory: true)
            try ensureDirectoryExists(skillsDir)
            for skillURL in skillFileURLs {
                // Use parent directory name as prefix for disambiguation, or just the filename
                let relativePath = skillURL.deletingLastPathComponent().lastPathComponent
                let destName: String
                if relativePath != tmpDir.lastPathComponent && relativePath != "skills" {
                    destName = "\(relativePath)_SKILL.md"
                } else {
                    destName = "SKILL.md"
                }
                let destURL = skillsDir.appendingPathComponent(destName, isDirectory: false)
                if FileManager.default.fileExists(atPath: destURL.path) {
                    try FileManager.default.removeItem(at: destURL)
                }
                try FileManager.default.copyItem(at: skillURL, to: destURL)
                installedSkillFiles.append(destURL)
                NSLog("[Project2501] Installed SKILL.md for plugin \(pluginId): \(destName)")
            }
        }

        // Copy documentation files (README.md, CHANGELOG.md) from the artifact
        for docName in ["README.md", "CHANGELOG.md"] {
            if let docURL = findDocFile(named: docName, in: tmpDir) {
                let destURL = installDir.appendingPathComponent(docName, isDirectory: false)
                if FileManager.default.fileExists(atPath: destURL.path) {
                    try FileManager.default.removeItem(at: destURL)
                }
                try FileManager.default.copyItem(at: docURL, to: destURL)
                NSLog("[Project2501] Installed \(docName) for plugin \(pluginId)")
            }
        }

        let dylibData = try Data(contentsOf: finalDylibURL)
        let dylibDigest = SHA256.hash(data: dylibData)
        let dylibSha = Data(dylibDigest).map { String(format: "%02x", $0) }.joined()

        let receipt = PluginReceipt(
            plugin_id: spec.plugin_id,
            version: resolution.version.version,
            installed_at: Date(),
            dylib_filename: finalDylibURL.lastPathComponent,
            dylib_sha256: dylibSha,
            platform: targetPlatform.rawValue,
            arch: targetArch.rawValue,
            public_keys: spec.public_keys,
            artifact: .init(
                url: artifact.url,
                sha256: artifact.sha256,
                minisign: artifact.minisign,
                size: artifact.size
            )
        )
        let receiptURL = installDir.appendingPathComponent("receipt.json", isDirectory: false)
        let receiptData = try JSONEncoder().encode(receipt)
        try receiptData.write(to: receiptURL)

        // Auto-grant user consent for plugins installed through the verified flow
        let consentURL = installDir.appendingPathComponent(".user_consent", isDirectory: false)
        try Data().write(to: consentURL)

        try Self.updateCurrentSymlink(pluginId: spec.plugin_id, version: resolution.version.version)

        return InstallResult(
            receipt: receipt,
            installDirectory: installDir,
            dylibURL: finalDylibURL,
            skillFiles: installedSkillFiles
        )
    }

    // MARK: - Paths
    public static func toolsRootDirectory() -> URL {
        return ToolsPaths.toolsRootDirectory()
    }

    public static func toolsPluginDirectory(pluginId: String) -> URL {
        toolsRootDirectory().appendingPathComponent(pluginId, isDirectory: true)
    }

    public static func toolsVersionDirectory(pluginId: String, version: SemanticVersion) -> URL {
        toolsPluginDirectory(pluginId: pluginId).appendingPathComponent(version.description, isDirectory: true)
    }

    public static func currentSymlinkURL(pluginId: String) -> URL {
        toolsPluginDirectory(pluginId: pluginId).appendingPathComponent("current", isDirectory: false)
    }

    public static func updateCurrentSymlink(pluginId: String, version: SemanticVersion) throws {
        let fm = FileManager.default
        let link = currentSymlinkURL(pluginId: pluginId)
        let dir = toolsPluginDirectory(pluginId: pluginId)
        if !fm.fileExists(atPath: dir.path) {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        if fm.fileExists(atPath: link.path) {
            try fm.removeItem(at: link)
        }
        try fm.createSymbolicLink(atPath: link.path, withDestinationPath: version.description)
    }

    private func ensureDirectoryExists(_ url: URL) throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
    }

    // MARK: - Download / unzip
    private func download(toTempFileFrom url: URL) async throws -> (fileURL: URL, data: Data) {
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw PluginInstallError.downloadFailed("HTTP \(statusCode) from \(url.absoluteString)")
        }
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".zip")
        try data.write(to: tmp)
        return (tmp, data)
    }

    private func unzip(zipURL: URL, to destination: URL) throws {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        task.arguments = ["unzip", "-o", zipURL.path, "-d", destination.path]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        try task.run()
        task.waitUntilExit()
        if task.terminationStatus != 0 {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let s = String(data: data, encoding: .utf8) ?? ""
            throw PluginInstallError.unzipFailed(s)
        }
    }

    private func makeTempDirectory() throws -> URL {
        let fm = FileManager.default
        let base = fm.temporaryDirectory
        let dir = base.appendingPathComponent("project2501-plugin-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Finds a documentation file (case-insensitive) in the extracted archive directory
    private func findDocFile(named filename: String, in directory: URL) -> URL? {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return nil
        }
        let target = filename.lowercased()
        for case let fileURL as URL in enumerator {
            if fileURL.lastPathComponent.lowercased() == target {
                return fileURL
            }
        }
        return nil
    }

    /// Finds all SKILL.md files in the extracted archive directory
    private func findSkillFiles(in directory: URL) -> [URL] {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return []
        }
        var results: [URL] = []
        for case let fileURL as URL in enumerator {
            if fileURL.lastPathComponent.uppercased() == "SKILL.MD" {
                results.append(fileURL)
            }
        }
        return results
    }

    private func findFirstDylib(in directory: URL) -> URL? {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        else {
            return nil
        }
        for case let fileURL as URL in enumerator {
            if fileURL.pathExtension == "dylib" {
                return fileURL
            }
        }
        return nil
    }

}
