import Foundation

/// Detects stop sequences in streaming text output.
/// Holds back characters that could be the start of a stop sequence,
/// only emitting them once confirmed safe.
public struct StopSequenceDetector: Sendable {

    private let stopSequences: [String]
    private let maxStopLen: Int  // Length of longest stop sequence

    /// Buffer of characters not yet emitted (might be start of stop sequence).
    private var buffer: String = ""

    /// Whether a stop sequence has been detected.
    public private(set) var stopped: Bool = false

    /// Which stop sequence was matched (if any).
    public private(set) var matchedSequence: String?

    public init(stopSequences: [String]) {
        self.stopSequences = stopSequences
        self.maxStopLen = stopSequences.map(\.count).max() ?? 0
    }

    /// Process new text. Returns the safe-to-emit portion.
    /// May return empty string if text is being buffered (potential stop sequence start).
    public mutating func process(_ newText: String) -> String {
        guard !stopped else { return "" }
        guard !stopSequences.isEmpty else { return newText }

        buffer += newText

        // Check for complete stop sequence match
        for seq in stopSequences {
            if let range = buffer.range(of: seq) {
                // Found stop sequence -- emit everything before it, stop
                let safe = String(buffer[buffer.startIndex..<range.lowerBound])
                stopped = true
                matchedSequence = seq
                buffer = ""
                return safe
            }
        }

        // Check if buffer ends with a partial match of any stop sequence
        let holdBack = partialMatchLength()

        if holdBack > 0 && holdBack <= buffer.count {
            // Emit safe prefix, keep potential match in buffer
            let safeEndIdx = buffer.index(buffer.endIndex, offsetBy: -holdBack)
            let safe = String(buffer[buffer.startIndex..<safeEndIdx])
            buffer = String(buffer[safeEndIdx...])
            return safe
        }

        // No partial match -- emit everything
        let safe = buffer
        buffer = ""
        return safe
    }

    /// Flush any remaining buffered text (call at end of generation).
    /// Returns buffered text that was being held back.
    public mutating func flush() -> String {
        let remaining = buffer
        buffer = ""
        return remaining
    }

    /// Reset detector state for reuse.
    public mutating func reset() {
        buffer = ""
        stopped = false
        matchedSequence = nil
    }

    // MARK: - Private

    /// Find the longest suffix of buffer that matches a prefix of any stop sequence.
    private func partialMatchLength() -> Int {
        var maxMatch = 0
        for seq in stopSequences {
            let checkLen = min(buffer.count, seq.count)
            for len in 1...checkLen {
                let bufferSuffix = String(buffer.suffix(len))
                let seqPrefix = String(seq.prefix(len))
                if bufferSuffix == seqPrefix {
                    maxMatch = max(maxMatch, len)
                }
            }
        }
        return maxMatch
    }
}
