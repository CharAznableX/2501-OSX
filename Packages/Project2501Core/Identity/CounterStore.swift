//
//  CounterStore.swift
//  project2501
//
//  Per-device monotonic counter persisted to UserDefaults.
//  Each device maintains its own counter; the server rejects replayed values.
//

import Foundation

public struct CounterStore: Sendable {
    public static let shared = CounterStore()

    private let key = "com.project2501.device.counter"

    public var current: UInt64 {
        UInt64(UserDefaults.standard.integer(forKey: key))
    }

    public func next() -> UInt64 {
        let value = current + 1
        UserDefaults.standard.set(value, forKey: key)
        return value
    }

    /// Sync the local counter with the server's last-seen value (Phase 1b).
    public func sync(to serverCounter: UInt64) {
        UserDefaults.standard.set(serverCounter, forKey: key)
    }
}
