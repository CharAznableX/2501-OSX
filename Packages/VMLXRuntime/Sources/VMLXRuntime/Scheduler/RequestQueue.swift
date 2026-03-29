import Foundation
import os

/// Manages request lifecycle: waiting -> running -> finished.
/// FCFS ordering with optional priority override.
public final class RequestQueue: @unchecked Sendable {

    private let lock = OSAllocatedUnfairLock()

    /// Pending requests in FCFS order (front = next to schedule).
    private var waiting: [InferenceRequest] = []

    /// Currently active requests, keyed by requestId.
    private var running: [String: InferenceRequest] = [:]

    /// All requests ever added (for lookup), keyed by requestId.
    private var allRequests: [String: InferenceRequest] = [:]

    /// Recently finished request IDs (cleared after schedule cycle).
    private var finishedIds: Set<String> = []

    public init() {}

    // MARK: - Add/Remove

    /// Add a new request to the waiting queue.
    public func addRequest(_ request: InferenceRequest) {
        lock.withLock {
            allRequests[request.requestId] = request
            waiting.append(request)
        }
    }

    /// Abort a request (remove from waiting or running).
    public func abortRequest(_ requestId: String) {
        lock.withLock {
            waiting.removeAll { $0.requestId == requestId }
            if var req = running.removeValue(forKey: requestId) {
                req.finish(reason: .abort)
                allRequests[requestId] = req
                finishedIds.insert(requestId)
            }
        }
    }

    // MARK: - Scheduling

    /// Move requests from waiting to running, respecting maxNumSeqs and maxBatchedTokens.
    /// Returns the newly scheduled request IDs.
    public func schedule(maxNumSeqs: Int, maxBatchedTokens: Int) -> [String] {
        lock.withLock {
            var scheduled: [String] = []
            var totalTokens = running.values.reduce(0) { $0 + $1.numPromptTokens }

            while !waiting.isEmpty && running.count < maxNumSeqs {
                let request = waiting[0]
                let newTotal = totalTokens + request.numPromptTokens

                // Check token budget (allow at least 1 request)
                if !running.isEmpty && newTotal > maxBatchedTokens {
                    break
                }

                var req = waiting.removeFirst()
                req.status = .running
                running[req.requestId] = req
                allRequests[req.requestId] = req
                scheduled.append(req.requestId)
                totalTokens = newTotal
            }

            return scheduled
        }
    }

    // MARK: - Finish

    /// Mark a request as finished.
    public func finishRequest(_ requestId: String, reason: FinishReason) {
        lock.withLock {
            if var req = running.removeValue(forKey: requestId) {
                req.finish(reason: reason)
                allRequests[requestId] = req
                finishedIds.insert(requestId)
            }
        }
    }

    /// Get and clear recently finished request IDs.
    public func drainFinished() -> Set<String> {
        lock.withLock {
            let finished = finishedIds
            finishedIds.removeAll()
            return finished
        }
    }

    // MARK: - Queries

    /// Get a request by ID.
    public func getRequest(_ requestId: String) -> InferenceRequest? {
        lock.withLock { allRequests[requestId] }
    }

    /// Update a running request (e.g., append output tokens).
    public func updateRequest(_ requestId: String, _ mutate: @Sendable (inout InferenceRequest) -> Void) {
        lock.withLock {
            if var req = running[requestId] {
                mutate(&req)
                running[requestId] = req
                allRequests[requestId] = req
            }
        }
    }

    public var waitingCount: Int { lock.withLock { waiting.count } }
    public var runningCount: Int { lock.withLock { running.count } }
    public var totalCount: Int { lock.withLock { allRequests.count } }

    /// All currently running request IDs.
    public var runningRequestIds: [String] {
        lock.withLock { Array(running.keys) }
    }

    /// All currently running requests.
    public var runningRequests: [InferenceRequest] {
        lock.withLock { Array(running.values) }
    }
}
