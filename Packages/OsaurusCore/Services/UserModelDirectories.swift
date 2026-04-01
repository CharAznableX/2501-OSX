//
//  UserModelDirectories.swift
//  osaurus
//
//  Manages user-configured additional model directories.
//  Persists via UserDefaults, syncs to ModelDetector.additionalDirectories.
//

import AppKit
import Foundation
import VMLXRuntime

@MainActor
final class UserModelDirectories: ObservableObject {
    static let shared = UserModelDirectories()

    @Published var directories: [URL] = []

    private let key = "UserModelDirectories"

    private init() {
        load()
        syncToDetector()
    }

    func addDirectory(_ url: URL) {
        guard !directories.contains(url) else { return }
        directories.append(url)
        save()
        syncToDetector()
        NotificationCenter.default.post(name: .localModelsChanged, object: nil)
    }

    func removeDirectory(at index: Int) {
        guard index < directories.count else { return }
        directories.remove(at: index)
        save()
        syncToDetector()
        NotificationCenter.default.post(name: .localModelsChanged, object: nil)
    }

    func removeDirectory(_ url: URL) {
        directories.removeAll { $0 == url }
        save()
        syncToDetector()
        NotificationCenter.default.post(name: .localModelsChanged, object: nil)
    }

    /// Show folder picker and add selected directory
    func pickAndAddDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = false
        panel.allowsMultipleSelection = false
        panel.title = "Add Model Directory"
        panel.message = "Select a directory containing model folders"

        guard panel.runModal() == .OK, let url = panel.url else { return }
        addDirectory(url)
    }

    private func load() {
        guard let paths = UserDefaults.standard.stringArray(forKey: key) else { return }
        directories = paths.map { URL(fileURLWithPath: $0, isDirectory: true) }
    }

    private func save() {
        let paths = directories.map(\.path)
        UserDefaults.standard.set(paths, forKey: key)
    }

    private func syncToDetector() {
        ModelDetector.additionalDirectories = directories
    }
}
