import Testing
import Foundation
import MLX
import MLXNN
@testable import VMLXRuntime

@Suite("MiniMax M2.5")
struct MiniMaxTests {
    @Test("Load and forward pass MiniMax M2.5 JANG_2L")
    func loadAndForward() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent("jang/models/MiniMax-M2.5-JANG_2L")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP: MiniMax not found")
            return
        }
        print("Loading MiniMax...")
        let loaded = try await ModelLoader.load(from: path)
        print("Loaded: vocab=\(loaded.vocabSize) layers=\(loaded.numLayers)")
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()
        print("Cache: \(cache.count) layers")
        let tokens = MLXArray([Int32(1)]).reshaped(1, 1)
        let logits = container.forward(tokens, cache: cache)
        MLX.eval(logits)
        print("Logits: \(logits.shape)")
        #expect(logits.dim(2) == loaded.vocabSize)
    }
}
