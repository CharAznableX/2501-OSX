import Testing
@testable import VMLXRuntime

@Suite("RequestQueue")
struct RequestQueueTests {

    @Test("Add and schedule")
    func addAndSchedule() {
        let queue = RequestQueue()
        queue.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1, 2, 3]))
        queue.addRequest(InferenceRequest(requestId: "r2", promptTokenIds: [4, 5]))
        #expect(queue.waitingCount == 2)
        #expect(queue.runningCount == 0)

        let scheduled = queue.schedule(maxNumSeqs: 10, maxBatchedTokens: 1000)
        #expect(scheduled.count == 2)
        #expect(queue.waitingCount == 0)
        #expect(queue.runningCount == 2)
    }

    @Test("Respects maxNumSeqs")
    func maxSeqs() {
        let queue = RequestQueue()
        for i in 0..<10 {
            queue.addRequest(InferenceRequest(requestId: "r\(i)", promptTokenIds: [1]))
        }
        let scheduled = queue.schedule(maxNumSeqs: 3, maxBatchedTokens: 10000)
        #expect(scheduled.count == 3)
        #expect(queue.waitingCount == 7)
    }

    @Test("Respects maxBatchedTokens")
    func maxTokens() {
        let queue = RequestQueue()
        queue.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: Array(0..<500)))
        queue.addRequest(InferenceRequest(requestId: "r2", promptTokenIds: Array(0..<500)))
        queue.addRequest(InferenceRequest(requestId: "r3", promptTokenIds: Array(0..<500)))

        let scheduled = queue.schedule(maxNumSeqs: 100, maxBatchedTokens: 1000)
        #expect(scheduled.count == 2)  // 500+500=1000, third would exceed
    }

    @Test("Finish and drain")
    func finishAndDrain() {
        let queue = RequestQueue()
        queue.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        _ = queue.schedule(maxNumSeqs: 10, maxBatchedTokens: 1000)

        queue.finishRequest("r1", reason: .stop)
        #expect(queue.runningCount == 0)

        let finished = queue.drainFinished()
        #expect(finished.contains("r1"))
        #expect(queue.drainFinished().isEmpty)  // Already drained
    }

    @Test("Abort removes from waiting")
    func abortWaiting() {
        let queue = RequestQueue()
        queue.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        queue.abortRequest("r1")
        #expect(queue.waitingCount == 0)
    }

    @Test("Abort removes from running")
    func abortRunning() {
        let queue = RequestQueue()
        queue.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        _ = queue.schedule(maxNumSeqs: 10, maxBatchedTokens: 1000)
        queue.abortRequest("r1")
        #expect(queue.runningCount == 0)
        #expect(queue.drainFinished().contains("r1"))
    }

    @Test("Update running request")
    func updateRunning() {
        let queue = RequestQueue()
        queue.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        _ = queue.schedule(maxNumSeqs: 10, maxBatchedTokens: 1000)

        queue.updateRequest("r1") { req in
            req.appendOutputToken(100)
            req.appendOutputToken(101)
        }

        let req = queue.getRequest("r1")
        #expect(req?.numOutputTokens == 2)
    }

    @Test("FCFS order preserved")
    func fcfsOrder() {
        let queue = RequestQueue()
        queue.addRequest(InferenceRequest(requestId: "first", promptTokenIds: [1]))
        queue.addRequest(InferenceRequest(requestId: "second", promptTokenIds: [2]))
        queue.addRequest(InferenceRequest(requestId: "third", promptTokenIds: [3]))

        let scheduled = queue.schedule(maxNumSeqs: 2, maxBatchedTokens: 1000)
        #expect(scheduled == ["first", "second"])
    }
}
