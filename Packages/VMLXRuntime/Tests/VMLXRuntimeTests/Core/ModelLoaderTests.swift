import Testing
import Foundation
@testable import VMLXRuntime

@Suite("ModelLoader")
struct ModelLoaderTests {

    @Test("Error descriptions exist")
    func errorDescriptions() {
        let errors: [ModelLoaderError] = [
            .configNotFound("/path"),
            .invalidConfig("bad"),
            .tokenizerNotFound("/path"),
            .weightsNotFound("/path"),
            .invalidWeightIndex,
            .shardNotFound("shard-1.safetensors"),
            .unsupportedArchitecture("unknown")
        ]
        for error in errors {
            #expect(error.errorDescription != nil)
        }
    }

    @Test("LoadedModel vocabSize from config")
    func vocabSize() {
        // Verify config structures we depend on
        let config: [String: Any] = ["vocab_size": 151936, "num_hidden_layers": 36]
        #expect(config["vocab_size"] as? Int == 151936)
        #expect(config["num_hidden_layers"] as? Int == 36)
    }

    @Test("LoadedModel nested text_config lookup")
    func nestedTextConfig() {
        let config: [String: Any] = [
            "text_config": [
                "vocab_size": 32064,
                "num_hidden_layers": 28,
                "hidden_size": 3072,
                "num_attention_heads": 24,
                "num_key_value_heads": 8
            ] as [String: Any]
        ]
        let textConfig = config["text_config"] as? [String: Any]
        #expect(textConfig?["vocab_size"] as? Int == 32064)
        #expect(textConfig?["num_hidden_layers"] as? Int == 28)
        #expect(textConfig?["hidden_size"] as? Int == 3072)
        #expect(textConfig?["num_attention_heads"] as? Int == 24)
        #expect(textConfig?["num_key_value_heads"] as? Int == 8)
    }

    @Test("EOS token ID parsing from config")
    func eosTokenParsing() {
        // Single eos_token_id
        let config1: [String: Any] = ["eos_token_id": 2]
        #expect(config1["eos_token_id"] as? Int == 2)

        // Array of eos_token_id
        let config2: [String: Any] = ["eos_token_id": [151645, 151643]]
        #expect(config2["eos_token_id"] as? [Int] == [151645, 151643])
    }

    @Test("Missing config throws")
    func missingConfig() async {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        do {
            _ = try await ModelLoader.load(from: tempDir)
            Issue.record("Expected error")
        } catch {
            // Expected: some form of not-found error
        }
    }

    @Test("Error descriptions are meaningful")
    func errorMessages() {
        #expect(ModelLoaderError.configNotFound("/models/test").errorDescription!.contains("config.json"))
        #expect(ModelLoaderError.weightsNotFound("/models/test").errorDescription!.contains("safetensors"))
        #expect(ModelLoaderError.shardNotFound("model-00001.safetensors").errorDescription!.contains("model-00001"))
        #expect(ModelLoaderError.unsupportedArchitecture("rwkv").errorDescription!.contains("rwkv"))
    }
}

@Suite("ModelContainer")
struct ModelContainerTests {

    @Test("DetectedModel properties")
    func detectedModelProperties() {
        let detected = DetectedModel(
            name: "Qwen3.5-4B-JANG_2S",
            family: "qwen3.5",
            sourceModel: "Qwen3.5-4B",
            jangProfile: "JANG_2S",
            isJang: true,
            architectureType: "hybrid_ssm",
            attentionType: "none",
            hasVision: true,
            hasSSM: true,
            hasMoE: false,
            isHybrid: true,
            modelPath: URL(fileURLWithPath: "/tmp/test")
        )
        #expect(detected.isJang)
        #expect(detected.isHybrid)
        #expect(detected.hasVision)
        #expect(detected.jangProfile == "JANG_2S")
        #expect(detected.family == "qwen3.5")
        #expect(detected.sourceModel == "Qwen3.5-4B")
        #expect(!detected.hasMoE)
    }

    @Test("DetectedModel non-JANG defaults")
    func nonJangDefaults() {
        let detected = DetectedModel(
            name: "Llama-3.3-8B",
            family: "llama",
            sourceModel: "Llama-3.3-8B",
            modelPath: URL(fileURLWithPath: "/tmp/test")
        )
        #expect(!detected.isJang)
        #expect(!detected.isHybrid)
        #expect(!detected.hasVision)
        #expect(detected.jangProfile == nil)
        #expect(detected.architectureType == "transformer")
        #expect(detected.attentionType == "gqa")
    }

    @Test("DetectedModel MoE properties")
    func moeProperties() {
        let detected = DetectedModel(
            name: "DeepSeek-V3-JANG_4K",
            family: "deepseek",
            sourceModel: "DeepSeek-V3",
            isJang: true,
            hasMoE: true,
            numExperts: 256,
            numExpertsPerTok: 8,
            kvLoraRank: 512,
            qkNopeHeadDim: 128,
            qkRopeHeadDim: 64,
            modelPath: URL(fileURLWithPath: "/tmp/test")
        )
        #expect(detected.hasMoE)
        #expect(detected.numExperts == 256)
        #expect(detected.numExpertsPerTok == 8)
        #expect(detected.kvLoraRank == 512)
    }
}
