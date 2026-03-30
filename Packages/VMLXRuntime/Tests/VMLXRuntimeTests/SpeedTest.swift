import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("Speed")
struct SpeedTest {
    @Test("MiniMax detailed timing breakdown")
    func miniMaxBreakdown() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent("jang/models/MiniMax-M2.5-JANG_2L")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP"); return
        }

        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()

        let msgs = [VMLXChatMessage(role: "user", content: "Hi")]
        let tokenIds = try container.applyChatTemplate(messages: msgs, addGenerationPrompt: true, enableThinking: true)

        // Prefill
        let input = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, tokenIds.count)
        let logits = container.forward(input, cache: cache)
        MLX.eval(logits)
        var nextToken = logits[0, -1].argMax().item(Int.self)

        // Warm up 5 tokens
        for _ in 0..<5 {
            let ni = MLXArray([Int32(nextToken)]).reshaped(1, 1)
            let out = container.forward(ni, cache: cache)
            nextToken = out[0, -1].argMax().item(Int.self)
        }

        // Now measure 20 tokens with breakdown
        var forwardTimes: [Double] = []
        var sampleTimes: [Double] = []
        var decodeTimes: [Double] = []
        var totalTimes: [Double] = []

        for _ in 0..<20 {
            let t0 = CFAbsoluteTimeGetCurrent()

            // Forward
            let ni = MLXArray([Int32(nextToken)]).reshaped(1, 1)
            let out = container.forward(ni, cache: cache)
            let t1 = CFAbsoluteTimeGetCurrent()

            // Sample
            nextToken = out[0, -1].argMax().item(Int.self)
            let t2 = CFAbsoluteTimeGetCurrent()

            // Decode
            let _ = container.decode([nextToken])
            let t3 = CFAbsoluteTimeGetCurrent()

            forwardTimes.append(t1 - t0)
            sampleTimes.append(t2 - t1)
            decodeTimes.append(t3 - t2)
            totalTimes.append(t3 - t0)
        }

        let avgForward = forwardTimes.reduce(0, +) / 20
        let avgSample = sampleTimes.reduce(0, +) / 20
        let avgDecode = decodeTimes.reduce(0, +) / 20
        let avgTotal = totalTimes.reduce(0, +) / 20

        print("BREAKDOWN (avg of 20 tokens after warmup):")
        print("  forward:  \(String(format: "%.4f", avgForward))s (\(String(format: "%.0f", avgForward/avgTotal*100))%)")
        print("  sample:   \(String(format: "%.4f", avgSample))s (\(String(format: "%.0f", avgSample/avgTotal*100))%)")
        print("  decode:   \(String(format: "%.4f", avgDecode))s (\(String(format: "%.0f", avgDecode/avgTotal*100))%)")
        print("  total:    \(String(format: "%.4f", avgTotal))s = \(String(format: "%.1f", 1.0/avgTotal)) tok/s")
    }
}
