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
}
