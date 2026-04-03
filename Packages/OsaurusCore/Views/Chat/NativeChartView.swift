//
//  NativeChartView.swift
//  osaurus
//

import AppKit
import AAInfographics
import SwiftUI

/// A native AppKit view that renders a chart using AAChartKit.
final class NativeChartView: NSView {
    private let aaChartView = AAChartView()
    private var currentConfig: ChartConfiguration?
    private var currentTheme: (any ThemeProtocol)?

    var onHeightChanged: (() -> Void)?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupView()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupView() {
        aaChartView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(aaChartView)

        NSLayoutConstraint.activate([
            aaChartView.leadingAnchor.constraint(equalTo: leadingAnchor),
            aaChartView.trailingAnchor.constraint(equalTo: trailingAnchor),
            aaChartView.topAnchor.constraint(equalTo: topAnchor),
            aaChartView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }

    func configure(config: ChartConfiguration, theme: any ThemeProtocol) {
        let themeChanged = currentTheme?.isDark != theme.isDark || currentTheme?.accentColor != theme.accentColor
        let configChanged = currentConfig != config

        self.currentTheme = theme
        self.currentConfig = config

        if themeChanged || configChanged {
            let aaChartModel = AAChartModel()
                .chartType(config.type.aaType)
                .title(config.title)
                .subtitle(config.subtitle ?? "")
                .categories(config.categories ?? [])
                .series(config.series.map { $0.toAAElement() })
                .backgroundColor(theme.primaryBackground.toHexString())
                .dataLabelsEnabled(true)
                .animationType(.bounce)
                .zoomType(.xy)

            // customize colors based on theme
            let primaryColor = theme.accentColor.toHexString()
            aaChartModel.colorsTheme([primaryColor, "#fe117c", "#ffc069", "#06caf4", "#7dffc0"])

            // style titles and labels
            let textColor = theme.primaryText.toHexString()
            let labelStyle = AAStyle().color(textColor).fontSize(11)
            
            aaChartModel.titleStyle(AAStyle().color(textColor).fontSize(16).fontWeight(.bold))
            aaChartModel.subtitleStyle(AAStyle().color(theme.secondaryText.toHexString()).fontSize(12))
            aaChartModel.xAxisLabelsStyle(labelStyle)
            aaChartModel.yAxisLabelsStyle(labelStyle)

            if themeChanged && !configChanged {
                aaChartView.aa_refreshChartWholeContentWithChartModel(aaChartModel)
            } else {
                aaChartView.aa_drawChartWithChartModel(aaChartModel)
            }
        }
    }

    func measuredHeight() -> CGFloat {
        return 320
    }
}

// Helper to convert Color/NSColor to Hex for AAChartKit
extension Color {
    func toHexString() -> String {
        guard let nsColor = NSColor(self).usingColorSpace(.deviceRGB) else { return "#000000" }
        let r = Int(nsColor.redComponent * 255)
        let g = Int(nsColor.greenComponent * 255)
        let b = Int(nsColor.blueComponent * 255)
        return String(format: "#%02X%02X%02X", r, g, b)
    }
}
