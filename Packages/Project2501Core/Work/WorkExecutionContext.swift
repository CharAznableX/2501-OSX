//
//  WorkExecutionContext.swift
//  project2501
//
//  TaskLocal context for tracking the current issue during work tool execution.
//

import Foundation

/// Execution context for work operations using TaskLocal storage
public enum WorkExecutionContext {
    /// The current issue ID being executed (available during tool calls)
    @TaskLocal public static var currentIssueId: String?

    /// The current batch ID for grouped operations (nil for non-batch operations)
    @TaskLocal public static var currentBatchId: UUID?

    /// The agent ID whose context is active for the current execution
    @TaskLocal public static var currentAgentId: UUID?
}
