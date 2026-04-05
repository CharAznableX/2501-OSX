//
//  WorkExecutionMode.swift
//  project2501
//
//  First-class execution mode for work sessions.
//

import Foundation

public enum WorkExecutionMode: Sendable {
    case hostFolder(WorkFolderContext)
    case sandbox
    case none

    public var folderContext: WorkFolderContext? {
        guard case .hostFolder(let context) = self else { return nil }
        return context
    }

    public var usesHostFolderTools: Bool {
        if case .hostFolder = self {
            return true
        }
        return false
    }

    public var usesSandboxTools: Bool {
        if case .sandbox = self {
            return true
        }
        return false
    }
}
