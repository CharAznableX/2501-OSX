import Testing
@testable import VMLXRuntime

@Suite("StopSequenceDetector")
struct StopSequenceDetectorTests {

    @Test("No stop sequences passes everything through")
    func noStopSequences() {
        var det = StopSequenceDetector(stopSequences: [])
        #expect(det.process("hello world") == "hello world")
        #expect(!det.stopped)
    }

    @Test("Detects stop sequence")
    func detectsStop() {
        var det = StopSequenceDetector(stopSequences: ["<|endoftext|>"])
        let r1 = det.process("Hello world<|endoftext|>more")
        #expect(r1 == "Hello world")
        #expect(det.stopped)
        #expect(det.matchedSequence == "<|endoftext|>")
    }

    @Test("Holds back partial match")
    func holdsBackPartial() {
        var det = StopSequenceDetector(stopSequences: ["stop"])
        let r1 = det.process("hello st")
        // "st" is prefix of "stop", should be held back
        #expect(r1 == "hello ")

        // Complete with non-matching continuation
        let r2 = det.process("uff")
        // "stuff" doesn't match "stop", release "st" + "uff"
        #expect(r2.contains("st"))
    }

    @Test("Cross-token-boundary detection")
    func crossBoundary() {
        var det = StopSequenceDetector(stopSequences: ["END"])
        let r1 = det.process("hello E")
        let r2 = det.process("ND")
        // Should detect "END" across two process() calls
        #expect(det.stopped)
    }

    @Test("Flush returns held-back buffer")
    func flush() {
        var det = StopSequenceDetector(stopSequences: ["stop"])
        _ = det.process("hello sto")
        let flushed = det.flush()
        #expect(flushed.contains("sto"))
    }

    @Test("Reset clears state")
    func reset() {
        var det = StopSequenceDetector(stopSequences: ["end"])
        _ = det.process("the end")
        #expect(det.stopped)
        det.reset()
        #expect(!det.stopped)
        #expect(det.matchedSequence == nil)
    }

    @Test("Multiple stop sequences")
    func multipleStopSeqs() {
        var det = StopSequenceDetector(stopSequences: ["<|end|>", "<|stop|>", "###"])
        let r1 = det.process("output###trailing")
        #expect(r1 == "output")
        #expect(det.stopped)
        #expect(det.matchedSequence == "###")
    }

    @Test("After stop, process returns empty")
    func afterStopEmpty() {
        var det = StopSequenceDetector(stopSequences: ["x"])
        _ = det.process("ax")
        #expect(det.stopped)
        #expect(det.process("more text") == "")
    }

    // MARK: - Think Block Stop Skip

    @Test("Skips stop sequence inside unclosed think block")
    func thinkBlockSkipsStop() {
        var det = StopSequenceDetector(stopSequences: ["<|endoftext|>"])
        _ = det.process("<think>")
        let r1 = det.process("reasoning here <|endoftext|> more thinking")
        // Should NOT stop -- we're inside an unclosed <think> block
        #expect(!det.stopped)
        #expect(r1.contains("<|endoftext|>"))
    }

    @Test("Matches stop sequence after think block closes")
    func thinkBlockClosedMatchesStop() {
        var det = StopSequenceDetector(stopSequences: ["<|endoftext|>"])
        _ = det.process("<think>reasoning</think>")
        let r1 = det.process("answer<|endoftext|>")
        // Think block is closed, stop sequence should match
        #expect(det.stopped)
        #expect(r1 == "answer")
    }

    @Test("Think block open then close then stop")
    func thinkOpenCloseThenStop() {
        var det = StopSequenceDetector(stopSequences: ["END"])
        _ = det.process("<think>thinking...")
        // Still inside think -- stop should not match
        let r1 = det.process("END inside think")
        #expect(!det.stopped)
        #expect(r1.contains("END"))

        // Close think block
        _ = det.process("</think>Now answering")
        // Now stop should match
        let r2 = det.process("final END")
        #expect(det.stopped)
        #expect(det.matchedSequence == "END")
    }

    @Test("No think block still matches stop normally")
    func noThinkBlockNormalBehavior() {
        var det = StopSequenceDetector(stopSequences: ["STOP"])
        let r1 = det.process("output STOP trailing")
        #expect(det.stopped)
        #expect(r1 == "output ")
    }

    @Test("Reset clears think block tracking")
    func resetClearsThinkState() {
        var det = StopSequenceDetector(stopSequences: ["END"])
        _ = det.process("<think>open")
        // Inside think, stop skipped
        _ = det.process("END")
        #expect(!det.stopped)

        det.reset()
        // After reset, no think block active
        _ = det.process("END")
        #expect(det.stopped)
    }
}
