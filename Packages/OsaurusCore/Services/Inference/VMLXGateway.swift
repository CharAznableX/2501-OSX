//
//  VMLXGateway.swift
//  osaurus
//
//  Gateway/router that tracks running vmlx engine instances.
//  Each model gets its own Python subprocess on a unique port.
//  The gateway maps model names to their port for request routing.
//

import Foundation

/// Tracks a running vmlx engine instance.
struct VMLXInstance: Sendable {
    let modelName: String
    let modelPath: String
    let port: Int
    let processIdentifier: Int32
    let startedAt: Date
}

/// Actor that manages the mapping of model names to running engine instances.
/// Thread-safe via Swift actor isolation.
actor VMLXGateway {
    static let shared = VMLXGateway()

    /// Active instances keyed by model name
    private var instances: [String: VMLXInstance] = [:]

    // MARK: - Registration

    /// Register a newly launched engine instance.
    /// Registers under both the model name and full path for reliable lookup.
    func register(_ instance: VMLXInstance) {
        instances[instance.modelName] = instance
        // Also register under the full path so lookups by path succeed
        if !instance.modelPath.isEmpty && instance.modelPath != instance.modelName {
            instances[instance.modelPath] = instance
        }
    }

    /// Unregister an instance by model name (removes all aliases).
    func unregister(model: String) {
        // Find the instance first so we can remove all keys pointing to it
        if let instance = instances[model] {
            let port = instance.port
            instances = instances.filter { $0.value.port != port }
        } else {
            instances.removeValue(forKey: model)
        }
    }

    // MARK: - Routing

    /// Get the port for a running model, or nil if not loaded.
    func port(for model: String) -> Int? {
        // Try exact match first
        if let instance = instances[model] {
            return instance.port
        }
        // Try case-insensitive match
        for (key, instance) in instances {
            if key.caseInsensitiveCompare(model) == .orderedSame {
                return instance.port
            }
        }
        // Try matching the last path component (e.g. "Llama-3.2-3B-Instruct-4bit"
        // matches "mlx-community/Llama-3.2-3B-Instruct-4bit")
        let modelSuffix = model.split(separator: "/").last.map(String.init) ?? model
        for (key, instance) in instances {
            let keySuffix = key.split(separator: "/").last.map(String.init) ?? key
            if keySuffix.caseInsensitiveCompare(modelSuffix) == .orderedSame {
                return instance.port
            }
        }
        return nil
    }

    /// List all running instances.
    func allInstances() -> [VMLXInstance] {
        Array(instances.values)
    }

    /// Number of running instances.
    var count: Int {
        instances.count
    }
}
