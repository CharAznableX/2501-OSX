import Testing
import MLX
@testable import VMLXRuntime

@Suite("Sampler")
struct SamplerTests {

    @Test("Greedy returns argmax")
    func greedy() {
        // Create logits where index 5 has highest value
        var logitsData = [Float](repeating: -10.0, count: 100)
        logitsData[5] = 100.0
        let logits = MLXArray(logitsData)

        let params = SamplingParams(temperature: 0.0)  // Greedy
        let token = Sampler.sample(logits: logits, params: params)
        #expect(token == 5)
    }

    @Test("ArgMax works on simple array")
    func argMax() {
        let logits = MLXArray([1.0, 5.0, 3.0, 2.0] as [Float])
        #expect(Sampler.argMax(logits) == 1)
    }
}
