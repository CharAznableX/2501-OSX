import Testing
import Foundation
@testable import VMLXRuntime

@Suite("VMLXRuntimeActor")
struct VMLXRuntimeActorTests {

    @Test("Initial state: no model loaded")
    func initialState() async {
        let runtime = VMLXRuntimeActor()
        let loaded = await runtime.isModelLoaded
        #expect(!loaded)
        let name = await runtime.currentModelName
        #expect(name == nil)
    }

    @Test("Load model with invalid name throws modelLoadFailed")
    func loadInvalidName() async {
        let runtime = VMLXRuntimeActor()
        do {
            try await runtime.loadModel(name: "nonexistent-model-xyz")
            Issue.record("Expected modelLoadFailed error")
        } catch let error as VMLXRuntimeError {
            if case .modelLoadFailed = error {
                // Expected
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            // Also acceptable (e.g., underlying loader error)
        }
    }

    @Test("Load model from invalid path throws modelLoadFailed")
    func loadInvalidPath() async {
        let runtime = VMLXRuntimeActor()
        let fakePath = URL(fileURLWithPath: "/tmp/vmlx-nonexistent-model-\(UUID().uuidString)")
        do {
            try await runtime.loadModel(from: fakePath)
            Issue.record("Expected modelLoadFailed error")
        } catch let error as VMLXRuntimeError {
            if case .modelLoadFailed = error {
                // Expected
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            // Also acceptable
        }
    }

    @Test("Unload on fresh runtime is safe")
    func unloadWithoutLoad() async {
        let runtime = VMLXRuntimeActor()
        await runtime.unloadModel()
        let loaded = await runtime.isModelLoaded
        #expect(!loaded)
    }

    @Test("Generate without model throws noModelLoaded")
    func generateNoModel() async {
        let runtime = VMLXRuntimeActor()
        let request = VMLXChatCompletionRequest(
            messages: [VMLXChatMessage(role: "user", content: "Hi")]
        )
        do {
            _ = try await runtime.generateStream(request: request)
            Issue.record("Expected error")
        } catch let error as VMLXRuntimeError {
            if case .noModelLoaded = error {
                // Expected
            } else {
                Issue.record("Wrong error: \(error)")
            }
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test("Error types have descriptions")
    func errorDescriptions() {
        let errors: [VMLXRuntimeError] = [
            .noModelLoaded,
            .modelLoadFailed("test"),
            .generationFailed("test"),
            .cacheCorruption("test"),
            .tokenizationFailed
        ]
        for error in errors {
            #expect(error.errorDescription != nil)
        }
    }

    @Test("Cache stats accessible")
    func cacheStats() async {
        let runtime = VMLXRuntimeActor()
        let stats = await runtime.cacheStats
        #expect(stats.memoryCacheHits == 0)
    }

    @Test("Auto-detect config applied")
    func autoDetectConfig() async {
        let runtime = VMLXRuntimeActor()
        let config = await runtime.config
        #expect(config.prefillStepSize >= 1024)
    }

    @Test("Clear cache does not crash when no model loaded")
    func clearCacheNoModel() async {
        await VMLXRuntimeActor().clearCache()
        // No crash = pass
    }

    @Test("Container is nil when no model loaded")
    func containerNilInitially() async {
        let runtime = VMLXRuntimeActor()
        let c = await runtime.container
        #expect(c == nil)
    }

    @Test("Non-streaming generate without model throws")
    func nonStreamingNoModel() async {
        let runtime = VMLXRuntimeActor()
        let request = VMLXChatCompletionRequest(
            messages: [VMLXChatMessage(role: "user", content: "Hi")]
        )
        do {
            _ = try await runtime.generate(request: request)
            Issue.record("Expected error")
        } catch {
            // Expected
        }
    }
}
