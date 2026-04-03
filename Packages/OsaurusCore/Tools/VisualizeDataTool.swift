//
//  VisualizeDataTool.swift
//  osaurus
//

import Foundation
import AAInfographics

struct VisualizeDataTool: OsaurusTool {
    let name = "visualize_data"
    let description = "Generate a chart visualization from structured data (CSV or JSON)."

    var parameters: JSONValue? {
        .object([
            "type": .string("object"),
            "properties": .object([
                "attachmentId": .object([
                    "type": .string("string"),
                    "description": .string("The ID of the attachment containing the data."),
                ]),
                "chartType": .object([
                    "type": .string("string"),
                    "enum": .array(ChartType.allCases.map { .string($0.rawValue) }),
                    "description": .string("The type of chart to generate."),
                ]),
                "title": .object([
                    "type": .string("string"),
                    "description": .string("The title of the chart."),
                ]),
                "subtitle": .object([
                    "type": .string("string"),
                    "description": .string("Optional subtitle for the chart."),
                ]),
                "categories": .object([
                    "type": .string("array"),
                    "items": .object(["type": .string("string")]),
                    "description": .string("Optional X-axis category labels."),
                ]),
                "series": .object([
                    "type": .string("array"),
                    "items": .object([
                        "type": .string("object"),
                        "properties": .object([
                            "name": .object(["type": .string("string")]),
                            "data": .object([
                                "type": .string("array"),
                                "items": .object(["type": .string("number")]),
                            ]),
                        ]),
                    ]),
                    "description": .string("The data series to plot."),
                ]),
                "xAxisTitle": .object([
                    "type": .string("string"),
                    "description": .string("Optional X-axis title."),
                ]),
                "yAxisTitle": .object([
                    "type": .string("string"),
                    "description": .string("Optional Y-axis title."),
                ]),
            ]),
            "required": .array([.string("chartType"), .string("title"), .string("series")]),
        ])
    }

    func execute(argumentsJSON: String) async throws -> String {
        // the tool doesn't "do" anything on the backend, it just returns the configuration
        // which the UI (generateBlocks) intercepts to render a chart.
        // we validate it's valid JSON matching our ChartConfiguration.
        guard let data = argumentsJSON.data(using: .utf8) else {
            throw NSError(domain: "VisualizeDataTool", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid JSON"])
        }

        let decoder = JSONDecoder()
        do {
            _ = try decoder.decode(ChartConfiguration.self, from: data)
            // return the JSON as the result. generateBlocks will look for this.
            return argumentsJSON
        } catch {
            throw NSError(domain: "VisualizeDataTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid ChartConfiguration: \(error.localizedDescription)"])
        }
    }
}
