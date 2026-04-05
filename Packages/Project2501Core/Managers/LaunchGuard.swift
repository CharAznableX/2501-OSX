//
//  LaunchGuard.swift
//  project2501
//
//  Detects repeated startup crashes and enters safe mode to break crash loops.
//  Uses UserDefaults to track whether the previous launch completed successfully.
//

import Foundation

@MainActor
enum LaunchGuard {
    private static let startupInProgressKey = "LaunchGuard.startupInProgress"
    private static let crashCountKey = "LaunchGuard.consecutiveCrashCount"
    private static let crashThreshold = 3

    private(set) static var isSafeMode = false

    /// Call at the very start of `applicationDidFinishLaunching`, before any
    /// plugin or repository work.
    @discardableResult
    static func checkOnLaunch() -> Bool {
        let defaults = UserDefaults.standard

        if defaults.bool(forKey: startupInProgressKey) {
            let count = defaults.integer(forKey: crashCountKey) + 1
            defaults.set(count, forKey: crashCountKey)
            NSLog("[Project2501] Previous launch did not complete (consecutive crashes: %d)", count)
            if count >= crashThreshold {
                isSafeMode = true
            }
        }

        defaults.set(true, forKey: startupInProgressKey)
        defaults.synchronize()
        return isSafeMode
    }

    /// Call after startup completes successfully. Resets the crash counter.
    static func markStartupComplete() {
        let defaults = UserDefaults.standard
        defaults.set(false, forKey: startupInProgressKey)
        defaults.set(0, forKey: crashCountKey)
        isSafeMode = false
    }
}
