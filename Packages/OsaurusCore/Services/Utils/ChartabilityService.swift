//
//  ChartabilityService.swift
//  osaurus
//

import Foundation

public struct ChartabilityService: Sendable {
    public static let shared = ChartabilityService()

    private init() {}

    /// Analyzes an attachment to determine if it can be visualized.
    public func isChartable(_ attachment: Attachment) -> Bool {
        guard case .document(let filename, let content, _) = attachment.kind else {
            return false
        }

        let ext = (filename as NSString).pathExtension.lowercased()
        guard ext == "csv" || ext == "json" else {
            return false
        }

        // basic heuristic is to check for numeric content or tabular structure
        if ext == "csv" {
            let lines = content.components(separatedBy: .newlines)
                .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            return lines.count >= 3 // Header + at least 2 data rows
        } else if ext == "json" {
            // very basic JSON check: is it an array of objects or an object with arrays?
            let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.hasPrefix("[") || (trimmed.hasPrefix("{") && trimmed.contains("["))
        }

        return false
    }

    /// Provides a recommendation for visualization if the attachment is chartable.
    public func recommendVisualization(for attachment: Attachment) -> String? {
        guard isChartable(attachment) else { return nil }
        guard case .document(let filename, _, _) = attachment.kind else { return nil }

        return "The file '\(filename)' contains structured data that can be visualized as a chart (e.g., line, bar, or pie chart)."
    }
}
