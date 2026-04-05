//
//  ToolsReload.swift
//  project2501
//
//  Command to send a reload notification to the app, triggering a rescan of installed plugins.
//

import Foundation

public struct ToolsReload {
    public static func execute(args: [String]) {
        AppControl.postDistributedNotification(name: "com.cuadralabs.project2501.control.toolsReload", userInfo: [:])
        print("Reload signal sent.")
        exit(EXIT_SUCCESS)
    }
}
