//
//  ManagementStateManager.swift
//  project2501
//
//  Manages the session state for the management interface.
//

import Foundation
import Combine

/// Manages the session state for the management interface.
@MainActor
public final class ManagementStateManager: ObservableObject {
    public static let shared = ManagementStateManager()

    /// Persists the last selected tab within the current app session.
    @Published public var selectedTab: ManagementTab = .settings

    private init() {}
}
