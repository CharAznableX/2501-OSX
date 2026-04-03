//
//  PythonEnvironmentManager.swift
//  osaurus
//
//  Manages the Python inference environment lifecycle. On first launch,
//  provisions Python 3.12 + MLX dependencies via the embedded uv tool
//  into ~/Library/Application Support/Osaurus/python/. Subsequent launches
//  detect the existing environment and skip provisioning.
//

import Foundation
import os.log

private let logger = Logger(subsystem: "ai.osaurus", category: "PythonEnv")

@MainActor
public final class PythonEnvironmentManager: ObservableObject {
    static let shared = PythonEnvironmentManager()

    // MARK: - Types

    enum State: Equatable {
        case checking
        case notProvisioned
        case provisioning(step: ProvisionStep, detail: String)
        case ready
        case failed(message: String)
    }

    enum ProvisionStep: Int, CaseIterable, Equatable {
        case installingPython = 1
        case creatingEnvironment = 2
        case installingPackages = 3
        case applyingPatches = 4
        case installingEngine = 5
        case verifying = 6

        var label: String {
            switch self {
            case .installingPython: return "Installing Python runtime..."
            case .creatingEnvironment: return "Creating environment..."
            case .installingPackages: return "Installing ML packages..."
            case .applyingPatches: return "Applying compatibility patches..."
            case .installingEngine: return "Installing inference engine..."
            case .verifying: return "Verifying installation..."
            }
        }

        static var totalSteps: Int { allCases.count }
    }

    // MARK: - Published State

    @Published private(set) var state: State = .checking

    var isReady: Bool { state == .ready }

    var shouldShowOverlay: Bool {
        switch state {
        case .notProvisioned, .provisioning, .failed: return true
        default: return false
        }
    }

    var isProvisioning: Bool {
        if case .provisioning = state { return true }
        return false
    }

    // MARK: - Paths

    private var envRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Osaurus/python")
    }

    private var pythonBin: String {
        envRoot.appendingPathComponent("bin/python3").path
    }

    private var versionFile: URL {
        envRoot.appendingPathComponent("OSAURUS_VERSION")
    }

    // MARK: - Init

    private init() {
        check()
    }

    // MARK: - Check

    func check() {
        state = .checking
        let hasBinary = FileManager.default.fileExists(atPath: pythonBin)
        let hasVersionMarker = FileManager.default.fileExists(atPath: versionFile.path)
        if hasBinary && hasVersionMarker {
            logger.info("Python environment found at \(self.pythonBin)")
            state = .ready
        } else {
            if hasBinary && !hasVersionMarker {
                logger.warning("Partial Python environment detected, cleaning up")
                try? FileManager.default.removeItem(at: envRoot)
            }
            logger.info("Python environment not found, provisioning required")
            state = .notProvisioned
        }
    }

    // MARK: - Python Path (for VMLXProcessManager)

    func pythonPath() -> String? {
        guard isReady else { return nil }
        return pythonBin
    }

    // MARK: - Provision

    func provision() async {
        guard !isProvisioning else { return }

        let uvPath = resolveUVPath()
        guard let uvPath else {
            state = .failed(message: "uv binary not found in app bundle")
            return
        }

        let engineSourcePath = resolveEngineSourcePath()
        let requirementsPath = resolveRequirementsPath()
        let patchesPath = resolvePatchesPath()

        do {
            // Step 1: Install Python
            state = .provisioning(step: .installingPython, detail: "Downloading Python 3.12")
            try await runProcess(
                uvPath,
                arguments: ["python", "install", "3.12"],
                description: "uv python install"
            )

            // Step 2: Create venv
            state = .provisioning(step: .creatingEnvironment, detail: "Creating virtual environment")
            let envPath = envRoot.path
            try FileManager.default.createDirectory(
                atPath: envRoot.deletingLastPathComponent().path,
                withIntermediateDirectories: true
            )
            try await runProcess(
                uvPath,
                arguments: ["venv", "--python", "3.12", envPath],
                description: "uv venv"
            )

            // Step 3: Install packages
            state = .provisioning(step: .installingPackages, detail: "This may take a minute")
            var pipArgs = ["pip", "install", "--python", pythonBin]
            if let reqPath = requirementsPath {
                pipArgs += ["-r", reqPath]
            } else {
                pipArgs += [
                    "mlx>=0.29.0", "mlx-lm>=0.30.2", "mlx-vlm>=0.1.0",
                    "transformers>=4.40.0", "tokenizers>=0.19.0", "huggingface-hub>=0.23.0",
                    "numpy>=1.24.0", "pillow>=10.0.0", "opencv-python-headless>=4.8.0",
                    "fastapi>=0.100.0", "python-multipart>=0.0.6", "uvicorn>=0.23.0", "jsonschema>=4.0.0",
                    "psutil>=5.9.0", "tqdm>=4.66.0", "pyyaml>=6.0",
                    "requests>=2.28.0", "tabulate>=0.9.0", "mlx-embeddings>=0.0.5",
                ]
            }
            try await runProcess(uvPath, arguments: pipArgs, description: "uv pip install")

            // Step 4: Apply patches
            state = .provisioning(step: .applyingPatches, detail: "Fixing torch-free compatibility")
            if let patchScript = patchesPath {
                let siteDir = findSitePackages()
                if let siteDir {
                    try await runProcess(
                        pythonBin,
                        arguments: [patchScript, siteDir],
                        description: "post_install_patches.py"
                    )
                }
            }

            // Step 5: Install engine (pure Python — direct copy into site-packages)
            state = .provisioning(step: .installingEngine, detail: "Installing vmlx-engine")
            if let enginePath = engineSourcePath, let siteDir = findSitePackages() {
                let srcPackage = (enginePath as NSString).appendingPathComponent("vmlx_engine")
                let dstPackage = (siteDir as NSString).appendingPathComponent("vmlx_engine")
                if FileManager.default.fileExists(atPath: dstPackage) {
                    try FileManager.default.removeItem(atPath: dstPackage)
                }
                try FileManager.default.copyItem(atPath: srcPackage, toPath: dstPackage)
                logger.info("Copied vmlx_engine to \(dstPackage)")
            }

            // Step 6: Verify
            state = .provisioning(step: .verifying, detail: "Checking installation")
            try await runProcess(
                pythonBin,
                arguments: ["-c", "import vmlx_engine; print(f'vmlx_engine {vmlx_engine.__version__}')"],
                description: "verify import"
            )

            // Write version marker
            let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
            try? version.write(to: versionFile, atomically: true, encoding: .utf8)

            logger.info("Python environment provisioned successfully")
            state = .ready

        } catch {
            logger.error("Provisioning failed: \(error.localizedDescription)")
            logger.info("Removing partial environment at \(self.envRoot.path)")
            try? FileManager.default.removeItem(at: envRoot)
            state = .failed(message: error.localizedDescription)
        }
    }

    // MARK: - Repair

    func repair() async {
        logger.info("Repairing Python environment — removing and re-provisioning")
        try? FileManager.default.removeItem(at: envRoot)
        await provision()
    }

    // MARK: - Process Execution

    private func runProcess(_ executable: String, arguments: [String], description: String) async throws {
        logger.info("[\(description)] \(executable) \(arguments.joined(separator: " "))")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments

        var env = ProcessInfo.processInfo.environment
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        process.environment = env

        let stderrPipe = Pipe()
        process.standardOutput = FileHandle.nullDevice
        process.standardError = stderrPipe

        try process.run()

        let stderrData = try await Task.detached {
            stderrPipe.fileHandleForReading.readDataToEndOfFile()
        }.value

        process.waitUntilExit()

        if process.terminationStatus != 0 {
            let stderr = String(decoding: stderrData, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let lastLine = stderr.components(separatedBy: .newlines).last ?? stderr
            throw PythonEnvError.commandFailed(step: description, detail: lastLine)
        }
    }

    // MARK: - Path Resolution

    private static let projectRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // → Inference/
        .deletingLastPathComponent()  // → Services/
        .deletingLastPathComponent()  // → OsaurusCore/
        .deletingLastPathComponent()  // → Packages/
        .deletingLastPathComponent()  // → project root

    private func resolveResource(_ relativePath: String) -> String? {
        if let resourcePath = Bundle.main.resourcePath {
            let bundled = (resourcePath as NSString).appendingPathComponent(relativePath)
            if FileManager.default.fileExists(atPath: bundled) {
                return bundled
            }
        }
        let devPath = Self.projectRoot.appendingPathComponent("Resources/\(relativePath)").path
        if FileManager.default.fileExists(atPath: devPath) {
            return devPath
        }
        return nil
    }

    private func resolveUVPath() -> String? {
        if let path = resolveResource("uv") { return path }
        // Fallback: system PATH
        let whichProc = Process()
        whichProc.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        whichProc.arguments = ["uv"]
        let pipe = Pipe()
        whichProc.standardOutput = pipe
        whichProc.standardError = FileHandle.nullDevice
        do {
            try whichProc.run()
            whichProc.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let path = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            if whichProc.terminationStatus == 0, !path.isEmpty {
                return path
            }
        } catch {}
        return nil
    }

    private func resolveEngineSourcePath() -> String? {
        resolveResource("vmlx_engine")
    }

    private func resolveRequirementsPath() -> String? {
        resolveResource("vmlx_engine/requirements.txt")
    }

    private func resolvePatchesPath() -> String? {
        resolveResource("vmlx_engine/post_install_patches.py")
    }

    private func findSitePackages() -> String? {
        let base = envRoot.appendingPathComponent("lib").path
        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: base) else {
            return nil
        }
        if let pyDir = contents.first(where: { $0.hasPrefix("python3") }) {
            let sitePath = (base as NSString)
                .appendingPathComponent(pyDir)
                .appending("/site-packages")
            if FileManager.default.fileExists(atPath: sitePath) {
                return sitePath
            }
        }
        return nil
    }
}

// MARK: - Errors

enum PythonEnvError: Error, LocalizedError {
    case commandFailed(step: String, detail: String)
    case pythonNotProvisioned

    var errorDescription: String? {
        switch self {
        case .commandFailed(let step, let detail):
            return "\(step) failed: \(detail)"
        case .pythonNotProvisioned:
            return "Python environment is not set up. Please set up local inference first."
        }
    }
}
