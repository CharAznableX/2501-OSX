//
//  ModelOptionsStore.swift
//  project2501
//
//  Persists model-specific option preferences (like Thinking mode)
//  to UserDefaults so they are remembered per LLM.
//

import Foundation
import Combine

@MainActor
final class ModelOptionsStore: ObservableObject {
    static let shared = ModelOptionsStore()

    private let userDefaults = UserDefaults.standard
    private let prefix = "model_options_"

    private init() {}

    /// Load persisted options for a specific model ID
    func loadOptions(for modelId: String) -> [String: ModelOptionValue]? {
        guard let data = userDefaults.data(forKey: prefix + modelId) else { return nil }

        do {
            let decoder = JSONDecoder()
            return try decoder.decode([String: ModelOptionValue].self, from: data)
        } catch {
            print("[ModelOptionsStore] Failed to decode options for \(modelId): \(error)")
            return nil
        }
    }

    /// Save options for a specific model ID
    func saveOptions(_ options: [String: ModelOptionValue], for modelId: String) {
        do {
            let encoder = JSONEncoder()
            let data = try encoder.encode(options)
            userDefaults.set(data, forKey: prefix + modelId)
        } catch {
            print("[ModelOptionsStore] Failed to encode options for \(modelId): \(error)")
        }
    }
}
