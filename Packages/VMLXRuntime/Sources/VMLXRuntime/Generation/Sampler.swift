import Foundation
import MLX
import MLXRandom

/// Compiled categorical sampler — fused temperature scaling + sampling.
/// `compile(shapeless: true)` generates a single GPU kernel instead of
/// separate division + categorical ops, avoiding intermediate allocations.
public let compiledCategoricalSample: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
    (logits: MLXArray, temperature: MLXArray) -> MLXArray in
    let scaled = logits / temperature
    return MLXRandom.categorical(scaled.expandedDimensions(axis: 0))
}

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
    /// Uses index-gather instead of full-vocab mask allocation:
    /// only reads/writes the unique token positions, O(unique_tokens) not O(vocab_size).
    public static func applyRepetitionPenalty(
        logits: MLXArray, tokens: [Int], penalty: Float
    ) -> MLXArray {
        let uniqueTokens = Array(Set(tokens))
        guard !uniqueTokens.isEmpty else { return logits }

        // Index-gather: extract logits at seen positions only
        let indices = MLXArray(uniqueTokens.map { Int32($0) })
        let gathered = logits[indices]

        // Penalize: negative scores * penalty, positive scores / penalty
        let penaltyVal = MLXArray(penalty)
        let penalizedNeg = gathered * penaltyVal
        let penalizedPos = gathered / penaltyVal
        let penalized = which(gathered .< 0, penalizedNeg, penalizedPos)

        // Scatter-write penalized values back at the token positions
        var result = logits[0...]  // shallow copy
        result[indices] = penalized
        return result
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
    /// Uses a threshold approach: find the minimum probability that stays within the
    /// top-p nucleus, then apply that threshold to the original (unsorted) logits.
    public static func topPFilter(logits: MLXArray, p: Float) -> MLXArray {
        let probs = softmax(logits)

        // Sort probabilities ascending (default)
        let sortedProbs = sorted(probs)
        let cumProbs = sortedProbs.cumsum()

        // Tokens in the ascending tail where cumulative prob < (1 - p) should be removed.
        // The remaining tokens (cumProbs >= 1-p) form the top-p nucleus.
        let removeMask = cumProbs .< MLXArray(1.0 - p)

        // Find threshold: the smallest probability that survives the filter.
        // Replace removed probs with infinity so they don't affect the min.
        let survivingProbs = which(removeMask, MLXArray(Float.infinity), sortedProbs)
        let threshold = survivingProbs.min()

        // Apply threshold to original logits: keep tokens whose prob >= threshold
        let keepMask = probs .>= threshold
        return which(keepMask, logits, MLXArray(Float(-1e9)))
    }

    /// Greedy argmax.
    public static func argMax(_ logits: MLXArray) -> Int {
        logits.argMax().item(Int.self)
    }
}
