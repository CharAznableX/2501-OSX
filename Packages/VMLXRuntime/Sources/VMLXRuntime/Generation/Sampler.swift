import Foundation
import MLX
import MLXRandom

/// Token sampler supporting multiple sampling strategies.
/// Operates on logits (raw model output) to produce token IDs.
public struct Sampler: Sendable {

    /// Sample a token from logits using the given parameters.
    /// - Parameters:
    ///   - logits: Raw model output, shape [vocabSize] or [1, vocabSize]
    ///   - params: Sampling parameters (temperature, topP, topK, etc.)
    ///   - previousTokens: Previously generated tokens (for repetition penalty)
    /// - Returns: Sampled token ID
    public static func sample(
        logits: MLXArray,
        params: SamplingParams,
        previousTokens: [Int] = []
    ) -> Int {
        var logits = logits

        // Ensure 1D
        if logits.ndim > 1 {
            logits = logits.squeezed()
        }

        // Apply repetition penalty
        if params.repetitionPenalty != 1.0 && !previousTokens.isEmpty {
            logits = applyRepetitionPenalty(
                logits: logits, tokens: previousTokens, penalty: params.repetitionPenalty)
        }

        // Greedy: just take argmax
        if params.isGreedy {
            return argMax(logits)
        }

        // Temperature scaling
        if params.temperature > 0 && params.temperature != 1.0 {
            logits = logits / params.temperature
        }

        // Top-k filtering
        if params.topK > 0 {
            logits = topKFilter(logits: logits, k: params.topK)
        }

        // Min-p filtering
        if params.minP > 0 {
            logits = minPFilter(logits: logits, p: params.minP)
        }

        // Top-p (nucleus) filtering
        if params.topP > 0 && params.topP < 1.0 {
            logits = topPFilter(logits: logits, p: params.topP)
        }

        // Sample from categorical distribution (takes unnormalized logits).
        // categorical needs at least 2D input: [1, vocabSize]
        let logits2D = logits.reshaped([1, logits.shape[0]])
        let tokenArray = MLXRandom.categorical(logits2D)
        return tokenArray.item(Int.self)
    }

    // MARK: - Filtering Operations

    /// Apply repetition penalty to previously seen tokens.
    public static func applyRepetitionPenalty(
        logits: MLXArray, tokens: [Int], penalty: Float
    ) -> MLXArray {
        let uniqueTokens = Array(Set(tokens))
        guard !uniqueTokens.isEmpty else { return logits }

        let vocabSize = logits.shape[0]

        // Build a boolean mask: true at positions that appeared in tokens
        var maskData = [Float](repeating: 0.0, count: vocabSize)
        for tokenId in uniqueTokens {
            if tokenId >= 0 && tokenId < vocabSize {
                maskData[tokenId] = 1.0
            }
        }
        let mask = MLXArray(maskData) .> 0

        // Compute penalized logits for every position (applied selectively via mask).
        // If score < 0, multiply by penalty; if score > 0, divide by penalty.
        let penaltyArray = MLXArray(penalty)
        let penalizedNeg = logits * penaltyArray
        let penalizedPos = logits / penaltyArray
        let penalized = which(logits .< 0, penalizedNeg, penalizedPos)

        // Apply penalty only at token positions that appeared previously
        return which(mask, penalized, logits)
    }

    /// Keep only top-k logits, set rest to -inf.
    public static func topKFilter(logits: MLXArray, k: Int) -> MLXArray {
        let vocabSize = logits.shape[0]
        let k = min(k, vocabSize)
        let sortedVals = sorted(logits)
        let threshold = sortedVals[vocabSize - k]
        let mask = logits .>= threshold
        return which(mask, logits, MLXArray(Float(-1e9)))
    }

    /// Filter tokens below min_p * max_probability.
    public static func minPFilter(logits: MLXArray, p: Float) -> MLXArray {
        let probs = softmax(logits)
        let maxProb = probs.max()
        let threshold = maxProb * MLXArray(p)
        let mask = probs .>= threshold
        return which(mask, logits, MLXArray(Float(-1e9)))
    }

    /// Keep tokens with cumulative probability <= top_p (nucleus sampling).
    public static func topPFilter(logits: MLXArray, p: Float) -> MLXArray {
        let probs = softmax(logits)
        let sortedIndices = argSort(probs)
        let sortedProbs = probs.take(sortedIndices)
        let cumProbs = sortedProbs.cumsum()

        // Find cutoff: keep tokens where cumulative prob exceeds (1 - p).
        // Tokens in the ascending-sorted order below the (1-p) threshold
        // are the ones we want to zero out.
        let cutoffMask = cumProbs .> MLXArray(1.0 - p)

        // Map the keep-mask back to the original token order.
        // We need to "unsort": create an inverse permutation of sortedIndices.
        let inversePerm = argSort(sortedIndices)
        let keepInOriginal = cutoffMask.take(inversePerm)

        return which(keepInOriginal, logits, MLXArray(Float(-1e9)))
    }

    /// Greedy argmax.
    public static func argMax(_ logits: MLXArray) -> Int {
        logits.argMax().item(Int.self)
    }
}
