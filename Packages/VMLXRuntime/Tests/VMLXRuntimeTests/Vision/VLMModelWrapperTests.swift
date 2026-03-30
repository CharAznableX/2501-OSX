import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("VLMConfig")
struct VLMConfigTests {

    @Test("Detect Qwen2.5-VL")
    func detectQwen() {
        let config = VLMConfigRegistry.detect(modelName: "Qwen2.5-VL-72B-JANG")
        #expect(config != nil)
        #expect(config?.family == "qwen2.5-vl")
        #expect(config?.usesGridTiling == true)
        #expect(config?.imageTokenStrategy == .replacement)
    }

    @Test("Detect Pixtral")
    func detectPixtral() {
        let config = VLMConfigRegistry.detect(modelName: "Pixtral-Large-Instruct")
        #expect(config != nil)
        #expect(config?.imageTokenStrategy == .interleave)
    }

    @Test("Detect LLaVA")
    func detectLLaVA() {
        let config = VLMConfigRegistry.detect(modelName: "llava-v1.6-34b")
        #expect(config != nil)
        #expect(config?.imageTokenStrategy == .prepend)
        #expect(config?.imageTokenCount == 576)
    }

    @Test("Unknown model returns nil")
    func unknownModel() {
        let config = VLMConfigRegistry.detect(modelName: "totally-unknown")
        #expect(config == nil)
    }

    @Test("VLM config defaults")
    func configDefaults() {
        let config = VLMConfig(family: "test")
        #expect(config.maxImages == 5)
        #expect(config.maxImageSizeBytes == 50 * 1024 * 1024)
        #expect(config.maxVideoFrames == 64)
        #expect(!config.usesGridTiling)
    }

    @Test("VLMInput hasImages")
    func vlmInputHasImages() {
        let empty = VLMInput(tokenIds: [1, 2, 3])
        #expect(!empty.hasImages)

        let withImages = VLMInput(
            tokenIds: [1, 2, 3],
            imageEmbeddings: [MLXArray.zeros([1, 256, 768])],
            imagePositions: [1]
        )
        #expect(withImages.hasImages)
    }

    @Test("Registry has all expected models")
    func registryCompleteness() {
        #expect(VLMConfigRegistry.configs.count >= 7)
        #expect(VLMConfigRegistry.configs["qwen2.5-vl"] != nil)
        #expect(VLMConfigRegistry.configs["pixtral"] != nil)
        #expect(VLMConfigRegistry.configs["llava"] != nil)
        #expect(VLMConfigRegistry.configs["internvl"] != nil)
        #expect(VLMConfigRegistry.configs["phi-3-vision"] != nil)
    }

    @Test("Image token strategies cover all cases")
    func allStrategies() {
        let strategies: Set<VLMImageTokenStrategy> = [.replacement, .prepend, .interleave]
        let registeredStrategies = Set(VLMConfigRegistry.configs.values.map(\.imageTokenStrategy))
        #expect(registeredStrategies == strategies)
    }
}
