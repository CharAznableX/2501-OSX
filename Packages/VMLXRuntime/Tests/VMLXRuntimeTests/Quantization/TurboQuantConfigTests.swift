import Testing
@testable import VMLXRuntime

@Suite("TurboQuantConfig")
struct TurboQuantConfigTests {

    @Test("Default config values")
    func defaults() {
        let config = TurboQuantConfig()
        #expect(config.defaultKeyBits == 3)
        #expect(config.defaultValueBits == 3)
        #expect(config.criticalKeyBits == 4)
        #expect(config.criticalValueBits == 4)
        #expect(config.seed == 42)
        #expect(config.layerPattern == nil)
        #expect(!config.isMLA)
    }

    @Test("Standard layer gets default bits")
    func standardLayer() {
        let config = TurboQuantConfig()
        // Layer 5 in a 32-layer model (not critical)
        #expect(config.keyBits(forLayer: 5, totalLayers: 32) == 3)
        #expect(config.valueBits(forLayer: 5, totalLayers: 32) == 3)
    }

    @Test("Critical layers get higher precision")
    func criticalLayers() {
        let config = TurboQuantConfig()
        // First 3 layers
        #expect(config.keyBits(forLayer: 0, totalLayers: 32) == 4)
        #expect(config.keyBits(forLayer: 1, totalLayers: 32) == 4)
        #expect(config.keyBits(forLayer: 2, totalLayers: 32) == 4)
        // Last 3 layers (negative indices resolve to 29, 30, 31)
        #expect(config.keyBits(forLayer: 29, totalLayers: 32) == 4)
        #expect(config.keyBits(forLayer: 30, totalLayers: 32) == 4)
        #expect(config.keyBits(forLayer: 31, totalLayers: 32) == 4)
        // Middle layer
        #expect(config.keyBits(forLayer: 15, totalLayers: 32) == 3)
    }

    @Test("SSM layers return nil (no KV to compress)")
    func ssmLayersNil() {
        let pattern: [LayerType] = [.ssm, .ssm, .ssm, .attention, .ssm, .attention]
        let config = TurboQuantConfig(layerPattern: pattern)

        #expect(config.keyBits(forLayer: 0, totalLayers: 6) == nil)  // SSM
        #expect(config.keyBits(forLayer: 1, totalLayers: 6) == nil)  // SSM
        #expect(config.keyBits(forLayer: 3, totalLayers: 6) != nil)  // Attention
        #expect(config.keyBits(forLayer: 5, totalLayers: 6) != nil)  // Attention
    }

    @Test("MLA dimensions")
    func mlaDimensions() {
        let config = TurboQuantConfig(mlaKeyDim: 192, mlaValueDim: 128)
        #expect(config.isMLA)
        #expect(config.keyDim == 192)
        #expect(config.valueDim == 128)
    }

    @Test("Attention layer count with hybrid pattern")
    func attentionCount() {
        // Nemotron-H style: 36 SSM + 12 attention in 48 layers
        var pattern: [LayerType] = []
        for i in 0..<48 {
            pattern.append(i % 4 == 3 ? .attention : .ssm)
        }
        let config = TurboQuantConfig(layerPattern: pattern)
        #expect(config.attentionLayerCount(totalLayers: 48) == 12)
        #expect(config.attentionLayerIndices(totalLayers: 48).count == 12)
    }

    @Test("No pattern means all attention")
    func noPatternAllAttention() {
        let config = TurboQuantConfig()
        #expect(config.attentionLayerCount(totalLayers: 32) == 32)
        #expect(config.attentionLayerIndices(totalLayers: 32) == Array(0..<32))
    }

    @Test("Expert layers return nil")
    func expertLayersNil() {
        let pattern: [LayerType] = [.attention, .expert, .attention]
        let config = TurboQuantConfig(layerPattern: pattern)
        #expect(config.keyBits(forLayer: 1, totalLayers: 3) == nil)  // Expert
    }
}
