import Testing
import Foundation
@testable import VMLXRuntime

@Suite("MLLMScheduler")
struct MLLMSchedulerTests {

    @Test("Default config")
    func defaultConfig() {
        let config = MLLMSchedulerConfig()
        #expect(config.maxImagesPerRequest == 10)
        #expect(config.maxVideoFrames == 64)
        #expect(config.enableVisionEmbeddingCache == true)
    }

    @Test("Creates with embedding cache")
    func withCache() {
        let scheduler = MLLMScheduler()
        #expect(scheduler.embeddingCache != nil)
    }

    @Test("Creates without embedding cache")
    func withoutCache() {
        let config = MLLMSchedulerConfig(enableVisionEmbeddingCache: false)
        let scheduler = MLLMScheduler(config: config)
        #expect(scheduler.embeddingCache == nil)
    }

    @Test("Strip gen prompt tokens")
    func stripGenPrompt() {
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let stripped = MLLMScheduler.stripGenPrompt(tokens: tokens, genPromptLen: 3)
        #expect(stripped == [1, 2, 3, 4, 5, 6, 7])
    }

    @Test("Strip gen prompt — zero length")
    func stripGenPromptZero() {
        let tokens = [1, 2, 3]
        let stripped = MLLMScheduler.stripGenPrompt(tokens: tokens, genPromptLen: 0)
        #expect(stripped == [1, 2, 3])
    }

    @Test("Strip gen prompt — exceeds length")
    func stripGenPromptExceeds() {
        let tokens = [1, 2, 3]
        let stripped = MLLMScheduler.stripGenPrompt(tokens: tokens, genPromptLen: 10)
        #expect(stripped == [1, 2, 3])  // No change
    }

    @Test("Compute gen prompt len")
    func computeGenPromptLen() {
        let withGen = [1, 2, 3, 4, 5, 6, 7, 8]     // 8 tokens
        let withoutGen = [1, 2, 3, 4, 5]              // 5 tokens
        let gpl = MLLMScheduler.computeGenPromptLen(
            tokensWithGenPrompt: withGen,
            tokensWithoutGenPrompt: withoutGen
        )
        #expect(gpl == 3)
    }

    @Test("Text-only request passthrough")
    func textOnlyPassthrough() {
        let scheduler = MLLMScheduler()
        let req = InferenceRequest(requestId: "r1", promptTokenIds: [1, 2, 3])
        scheduler.addRequest(req)
        let output = scheduler.schedule()
        #expect(output.scheduledRequestIds == ["r1"])
    }

    @Test("Passthrough methods work")
    func passthroughMethods() {
        let scheduler = MLLMScheduler()
        scheduler.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        _ = scheduler.schedule()
        scheduler.recordOutput(requestId: "r1", tokenId: 100, text: "hi")
        scheduler.finishRequest("r1", reason: .stop)
        #expect(scheduler.runningCount == 0)
    }
}
