import Testing
import Foundation
@testable import VMLXRuntime

@Suite("MoEConfig")
struct MoEConfigTests {

    @Test("Parse Qwen3.5 MoE 122B config")
    func parseQwen35_122B() {
        let config: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": [
                "hidden_size": 6144,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 1024,
                "shared_expert_intermediate_size": 1024,
                "intermediate_size": 1024
            ] as [String: Any]
        ]
        let moe = MoEConfig.from(config: config)
        #expect(moe.numExperts == 256)
        #expect(moe.numExpertsPerTok == 8)
        #expect(moe.hiddenSize == 6144)
        #expect(moe.moeIntermediateSize == 1024)
        #expect(moe.hasSharedExpert == true)
        #expect(moe.sharedExpertIntermediateSize == 1024)
        #expect(moe.hasSharedExpertGate == true)
    }

    @Test("Parse Qwen3.5 MoE 35B config")
    func parseQwen35_35B() {
        let config: [String: Any] = [
            "text_config": [
                "hidden_size": 4096,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512
            ] as [String: Any]
        ]
        let moe = MoEConfig.from(config: config)
        #expect(moe.numExperts == 256)
        #expect(moe.numExpertsPerTok == 8)
        #expect(moe.hiddenSize == 4096)
        #expect(moe.moeIntermediateSize == 512)
        #expect(moe.hasSharedExpert == true)
        #expect(moe.sharedExpertIntermediateSize == 512)
    }

    @Test("Parse MiniMax M2.5 config (no shared expert)")
    func parseMiniMax() {
        let config: [String: Any] = [
            "hidden_size": 6144,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "intermediate_size": 1536
        ]
        let moe = MoEConfig.from(config: config)
        #expect(moe.numExperts == 256)
        #expect(moe.numExpertsPerTok == 8)
        #expect(moe.hiddenSize == 6144)
        #expect(moe.moeIntermediateSize == 1536)
        #expect(moe.hasSharedExpert == false)
        #expect(moe.sharedExpertIntermediateSize == nil)
        #expect(moe.hasSharedExpertGate == false)
    }

    @Test("Top-level keys override text_config")
    func topLevelOverride() {
        let config: [String: Any] = [
            "hidden_size": 8192,
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 2048,
            "text_config": [
                "hidden_size": 4096,
                "num_experts": 64,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 1024
            ] as [String: Any]
        ]
        let moe = MoEConfig.from(config: config)
        // Top-level should win
        #expect(moe.numExperts == 128)
        #expect(moe.numExpertsPerTok == 4)
        #expect(moe.hiddenSize == 8192)
        #expect(moe.moeIntermediateSize == 2048)
    }

    @Test("Defaults for missing fields")
    func defaults() {
        let moe = MoEConfig.from(config: [:])
        #expect(moe.numExperts == 256)
        #expect(moe.numExpertsPerTok == 8)
        #expect(moe.hiddenSize == 4096)
        #expect(moe.moeIntermediateSize == 1024)
        #expect(moe.hasSharedExpert == false)
    }

    @Test("Fallback from num_experts to num_local_experts")
    func expertKeyFallback() {
        let config: [String: Any] = [
            "num_local_experts": 64,
            "intermediate_size": 2048
        ]
        let moe = MoEConfig.from(config: config)
        #expect(moe.numExperts == 64)
        #expect(moe.moeIntermediateSize == 2048)
    }

    @Test("Shared expert gate tracks shared expert presence")
    func sharedExpertGate() {
        // With shared expert
        let withShared = MoEConfig.from(config: [
            "shared_expert_intermediate_size": 512
        ] as [String: Any])
        #expect(withShared.hasSharedExpert == true)
        #expect(withShared.hasSharedExpertGate == true)

        // Without shared expert
        let withoutShared = MoEConfig.from(config: [:])
        #expect(withoutShared.hasSharedExpert == false)
        #expect(withoutShared.hasSharedExpertGate == false)
    }

    @Test("Direct initializer")
    func directInit() {
        let moe = MoEConfig(
            numExperts: 16,
            numExpertsPerTok: 2,
            hiddenSize: 512,
            moeIntermediateSize: 256,
            hasSharedExpert: true,
            sharedExpertIntermediateSize: 128,
            hasSharedExpertGate: true
        )
        #expect(moe.numExperts == 16)
        #expect(moe.numExpertsPerTok == 2)
        #expect(moe.hiddenSize == 512)
        #expect(moe.moeIntermediateSize == 256)
        #expect(moe.hasSharedExpert == true)
        #expect(moe.sharedExpertIntermediateSize == 128)
        #expect(moe.hasSharedExpertGate == true)
    }
}
