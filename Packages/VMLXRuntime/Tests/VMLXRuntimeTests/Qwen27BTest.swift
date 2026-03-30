import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("Model Switch")
struct ModelSwitchTest {
    @Test("MiniMax then 27B")
    func switchTest() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let mm = home.appendingPathComponent("jang/models/MiniMax-M2.5-JANG_2L")
        let q27 = home.appendingPathComponent("jang/models/Qwen3.5-27B-JANG_4S")
        guard FileManager.default.fileExists(atPath: mm.appendingPathComponent("config.json").path),
              FileManager.default.fileExists(atPath: q27.appendingPathComponent("config.json").path)
        else { print("SKIP"); return }

        let rt = VMLXRuntimeActor()
        print("Load MiniMax...")
        try await rt.loadModel(from: mm)
        let r1 = VMLXChatCompletionRequest(messages: [VMLXChatMessage(role: "user", content: "Hi")], model: nil, temperature: 0, maxTokens: 5, topP: 1.0, repetitionPenalty: 1.0, stop: [], stream: false)
        let o1 = try await rt.generate(request: r1)
        print("MiniMax: \(o1.prefix(40))")

        print("Switch to 27B...")
        try await rt.loadModel(from: q27)
        let r2 = VMLXChatCompletionRequest(messages: [VMLXChatMessage(role: "user", content: "Hi")], model: nil, temperature: 0, maxTokens: 5, topP: 1.0, repetitionPenalty: 1.0, stop: [], stream: false)
        let o2 = try await rt.generate(request: r2)
        print("27B: \(o2.prefix(40))")
        await rt.unloadModel()
    }
}
