import Testing
import Foundation
@testable import VMLXRuntime

@Suite("GenerationEngine")
struct GenerationEngineTests {

    @Test("Common prefix detection — full match")
    func commonPrefixFull() {
        let len = GenerationEngine.commonPrefixLength([1, 2, 3], [1, 2, 3])
        #expect(len == 3)
    }

    @Test("Common prefix detection — partial match")
    func commonPrefixPartial() {
        let len = GenerationEngine.commonPrefixLength([1, 2, 3, 4, 5], [1, 2, 3, 6, 7])
        #expect(len == 3)
    }

    @Test("Common prefix detection — no match")
    func commonPrefixNone() {
        let len = GenerationEngine.commonPrefixLength([1, 2, 3], [4, 5, 6])
        #expect(len == 0)
    }

    @Test("Common prefix detection — empty")
    func commonPrefixEmpty() {
        #expect(GenerationEngine.commonPrefixLength([], [1, 2, 3]) == 0)
        #expect(GenerationEngine.commonPrefixLength([1, 2], []) == 0)
        #expect(GenerationEngine.commonPrefixLength([], []) == 0)
    }

    @Test("Common prefix detection — different lengths")
    func commonPrefixDiffLengths() {
        let len = GenerationEngine.commonPrefixLength([1, 2, 3], [1, 2, 3, 4, 5, 6])
        #expect(len == 3)
    }

    @Test("GenerationConfig defaults")
    func configDefaults() {
        let config = GenerationConfig()
        #expect(config.maxTokens == 2048)
        #expect(!config.isHybrid)
        #expect(!config.enableTQ)
        #expect(!config.enableThinking)
        #expect(config.genPromptLen == 0)
    }

    @Test("GenerationConfig thinking model")
    func configThinking() {
        let config = GenerationConfig(
            isHybrid: true,
            enableThinking: true,
            genPromptLen: 5
        )
        #expect(config.isHybrid)
        #expect(config.enableThinking)
        #expect(config.genPromptLen == 5)
    }

    @Test("GenerationResult captures all fields")
    func resultFields() {
        let result = GenerationResult(
            tokenIds: [100, 101, 102],
            text: "hello",
            finishReason: .stop,
            promptTokens: 10,
            cachedTokens: 5,
            completionTokens: 3,
            cacheDetail: .prefix,
            ssmCheckpoint: nil
        )
        #expect(result.tokenIds.count == 3)
        #expect(result.finishReason == .stop)
        #expect(result.cachedTokens == 5)
        #expect(result.cacheDetail == .prefix)
    }

    @Test("Stable boundary calculation for thinking models")
    func stableBoundary() {
        // 100 prompt tokens, 5 are gen_prompt (<think>\n)
        // Stable boundary = 100 - 5 = 95
        let config = GenerationConfig(
            isHybrid: true,
            enableThinking: true,
            genPromptLen: 5
        )
        let boundary = 100 - config.genPromptLen
        #expect(boundary == 95)
    }
}
