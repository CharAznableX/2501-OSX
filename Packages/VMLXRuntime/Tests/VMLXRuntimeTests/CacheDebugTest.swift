import Testing
import Foundation
import MLX
import MLXNN
@testable import VMLXRuntime

@Suite("Speed Check")
struct SpeedCheckTest {
    @Test("Check model dtype and speed")
    func checkDtypeAndSpeed() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent("jang/models/MiniMax-M2.5-JANG_2L")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP"); return
        }
        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()

        // Forward one token and check output dtype
        let input = MLXArray([Int32(1)]).reshaped(1, 1)
        let logits = container.forward(input, cache: cache)
        MLX.eval(logits)
        print("Logits dtype: \(logits.dtype)")

        // Speed test
        var y = logits[0,-1].argMax()
        MLX.eval(y)
        for _ in 0..<3 {
            let d = container.forward(MLXArray([Int32(y.item(Int.self))]).reshaped(1,1), cache: cache)
            y = d[0,-1].argMax()
            MLX.eval(y)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<20 {
            let tok = y.item(Int.self)
            let d = container.forward(MLXArray([Int32(tok)]).reshaped(1,1), cache: cache)
            y = d[0,-1].argMax()
            MLX.eval(y)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("MiniMax speed: \(String(format: "%.1f", 20/elapsed)) tok/s")
        
        // Check: what happens if we cast input to bfloat16?
        // (This might trigger bfloat16 compute path)
    }
}
