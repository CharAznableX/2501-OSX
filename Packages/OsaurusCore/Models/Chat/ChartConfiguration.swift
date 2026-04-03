//
//  ChartConfiguration.swift
//  osaurus
//
//  Models for data visualization using AAChartKit.
//

import Foundation
import AAInfographics

/// Supported chart types for visualization
public enum ChartType: String, Codable, Sendable, CaseIterable {
    case column
    case bar
    case area
    case areaspline
    case line
    case spline
    case pie
    case bubble
    case scatter
    case pyramid
    case funnel
    case arearange
    case columnrange

    var aaType: AAChartType {
        switch self {
        case .column: return .column
        case .bar: return .bar
        case .area: return .area
        case .areaspline: return .areaspline
        case .line: return .line
        case .spline: return .spline
        case .pie: return .pie
        case .bubble: return .bubble
        case .scatter: return .scatter
        case .pyramid: return .pyramid
        case .funnel: return .funnel
        case .arearange: return .arearange
        case .columnrange: return .columnrange
        }
    }
}

/// Represents a single series of data in a chart
public struct ChartSeries: Codable, Sendable, Equatable {
    public let name: String
    public let data: [Double]
    public let color: String?

    public init(name: String, data: [Double], color: String? = nil) {
        self.name = name
        self.data = data
        self.color = color
    }

    func toAAElement() -> AASeriesElement {
        AASeriesElement()
            .name(name)
            .data(data)
            .color(color as Any)
    }
}

/// Complete configuration for a chart block
public struct ChartConfiguration: Codable, Sendable, Equatable {
    public let type: ChartType
    public let title: String
    public let subtitle: String?
    public let categories: [String]?
    public let series: [ChartSeries]
    public let xAxisTitle: String?
    public let yAxisTitle: String?

    private enum CodingKeys: String, CodingKey {
        case chartType
        case type
        case title
        case subtitle
        case categories
        case series
        case xAxisTitle
        case yAxisTitle
    }

    public init(
        type: ChartType,
        title: String,
        subtitle: String? = nil,
        categories: [String]? = nil,
        series: [ChartSeries],
        xAxisTitle: String? = nil,
        yAxisTitle: String? = nil
    ) {
        self.type = type
        self.title = title
        self.subtitle = subtitle
        self.categories = categories
        self.series = series
        self.xAxisTitle = xAxisTitle
        self.yAxisTitle = yAxisTitle
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let chartType = try c.decodeIfPresent(ChartType.self, forKey: .chartType) {
            type = chartType
        } else {
            type = try c.decode(ChartType.self, forKey: .type)
        }
        title = try c.decode(String.self, forKey: .title)
        subtitle = try c.decodeIfPresent(String.self, forKey: .subtitle)
        categories = try c.decodeIfPresent([String].self, forKey: .categories)
        series = try c.decode([ChartSeries].self, forKey: .series)
        xAxisTitle = try c.decodeIfPresent(String.self, forKey: .xAxisTitle)
        yAxisTitle = try c.decodeIfPresent(String.self, forKey: .yAxisTitle)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(type, forKey: .chartType)
        try c.encode(title, forKey: .title)
        try c.encodeIfPresent(subtitle, forKey: .subtitle)
        try c.encodeIfPresent(categories, forKey: .categories)
        try c.encode(series, forKey: .series)
        try c.encodeIfPresent(xAxisTitle, forKey: .xAxisTitle)
        try c.encodeIfPresent(yAxisTitle, forKey: .yAxisTitle)
    }
}
