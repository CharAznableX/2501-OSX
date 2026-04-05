//
//  ManagementTab.swift
//  project2501
//
//  Defines all available tabs in the management sidebar.
//

import Foundation
import SwiftUI

/// Defines all available tabs in the management sidebar.
public enum ManagementTab: String, CaseIterable, Identifiable {
    case models
    case providers
    case agents
    case plugins
    case sandbox
    case tools
    case skills
    case memory
    case schedules
    case watchers
    case voice
    case themes
    case insights
    case server
    case permissions
    case identity
    case settings

    public var id: String { rawValue }

    public var icon: String {
        switch self {
        case .models: "cube.box.fill"
        case .providers: "cloud.fill"
        case .agents: "person.2.fill"
        case .plugins: "puzzlepiece.extension.fill"
        case .sandbox: "shippingbox.fill"
        case .tools: "wrench.and.screwdriver.fill"
        case .skills: "sparkles"
        case .memory: "brain.head.profile.fill"
        case .schedules: "calendar.badge.clock"
        case .watchers: "eye.fill"
        case .voice: "waveform"
        case .themes: "paintpalette.fill"
        case .insights: "chart.bar.doc.horizontal"
        case .server: "server.rack"
        case .permissions: "lock.shield.fill"
        case .identity: "person.badge.key.fill"
        case .settings: "gearshape.fill"
        }
    }

    public var label: String {
        switch self {
        case .models: "Models"
        case .providers: "Providers"
        case .agents: "Agents"
        case .plugins: "Plugins"
        case .sandbox: "Sandbox"
        case .tools: "Tools"
        case .skills: "Skills"
        case .memory: "Memory"
        case .schedules: "Schedules"
        case .watchers: "Watchers"
        case .voice: "Voice"
        case .themes: "Themes"
        case .insights: "Insights"
        case .server: "Server"
        case .permissions: "Permissions"
        case .identity: "Identity"
        case .settings: "Settings"
        }
    }

    /// Creates a sidebar item for this tab with an optional badge count and highlight state.
    func sidebarItem(badge: Int? = nil, badgeHighlight: Bool = false) -> SidebarItemData {
        SidebarItemData(
            id: rawValue,
            icon: icon,
            label: label,
            badge: badge,
            badgeHighlight: badgeHighlight
        )
    }
}
