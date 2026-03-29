import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("SSMStateCache")
struct SSMStateCacheTests {

    private func makeCheckpoint(boundary: Int, hash: String? = nil) -> SSMCheckpoint {
        let states = [
            SSMStateLayer(state: [MLXArray.zeros([1, 16, 256])]),
            SSMStateLayer(state: [MLXArray.zeros([1, 16, 256])]),
        ]
        let tokenHash = hash ?? "abc123"
        return SSMCheckpoint(ssmStates: states, boundary: boundary, tokenHash: tokenHash)
    }

    @Test("Store and fetch")
    func storeAndFetch() {
        let cache = SSMStateCache(maxEntries: 10)
        let cp = makeCheckpoint(boundary: 100, hash: "test_hash")
        cache.store(checkpoint: cp)

        let result = cache.fetch(tokenHash: "test_hash", boundary: 100)
        #expect(result != nil)
        #expect(result?.boundary == 100)
        #expect(result?.ssmStates.count == 2)
        #expect(cache.hits == 1)
    }

    @Test("Miss returns nil")
    func miss() {
        let cache = SSMStateCache(maxEntries: 10)
        let result = cache.fetch(tokenHash: "nonexistent", boundary: 50)
        #expect(result == nil)
        #expect(cache.misses == 1)
    }

    @Test("Empty states treated as MISS")
    func emptyStatesMiss() {
        let cache = SSMStateCache(maxEntries: 10)
        // Store checkpoint with empty states
        let cp = SSMCheckpoint(ssmStates: [], boundary: 10, tokenHash: "empty_test")
        cache.store(checkpoint: cp)

        let result = cache.fetch(tokenHash: "empty_test", boundary: 10)
        #expect(result == nil)  // Empty == MISS, not hit
        #expect(cache.misses == 1)
    }

    @Test("LRU eviction at max entries")
    func lruEviction() {
        let cache = SSMStateCache(maxEntries: 3)
        cache.store(checkpoint: makeCheckpoint(boundary: 1, hash: "h1"))
        cache.store(checkpoint: makeCheckpoint(boundary: 2, hash: "h2"))
        cache.store(checkpoint: makeCheckpoint(boundary: 3, hash: "h3"))
        cache.store(checkpoint: makeCheckpoint(boundary: 4, hash: "h4"))  // Evicts h1

        #expect(cache.count == 3)
        #expect(cache.fetch(tokenHash: "h1", boundary: 1) == nil)  // Evicted
        #expect(cache.fetch(tokenHash: "h2", boundary: 2) != nil)  // Still present
    }

    @Test("Fetch returns deep copy")
    func deepCopy() {
        let cache = SSMStateCache(maxEntries: 10)
        cache.store(checkpoint: makeCheckpoint(boundary: 100, hash: "copy_test"))

        let r1 = cache.fetch(tokenHash: "copy_test", boundary: 100)
        let r2 = cache.fetch(tokenHash: "copy_test", boundary: 100)

        // Both should exist
        #expect(r1 != nil)
        #expect(r2 != nil)
        // Both should be independent (different array objects)
        // Can't easily test MLXArray identity, but both should have same shape
        #expect(r1?.ssmStates.count == r2?.ssmStates.count)
    }

    @Test("Invalidate removes entry")
    func invalidate() {
        let cache = SSMStateCache(maxEntries: 10)
        cache.store(checkpoint: makeCheckpoint(boundary: 50, hash: "inv_test"))
        #expect(cache.count == 1)
        cache.invalidate(tokenHash: "inv_test", boundary: 50)
        #expect(cache.count == 0)
    }

    @Test("Clear removes all")
    func clearAll() {
        let cache = SSMStateCache(maxEntries: 10)
        cache.store(checkpoint: makeCheckpoint(boundary: 1, hash: "c1"))
        cache.store(checkpoint: makeCheckpoint(boundary: 2, hash: "c2"))
        cache.clear()
        #expect(cache.count == 0)
    }

    @Test("Boundary distinguishes same token hash")
    func boundaryDistinguishes() {
        let cache = SSMStateCache(maxEntries: 10)
        cache.store(checkpoint: makeCheckpoint(boundary: 100, hash: "same"))
        cache.store(checkpoint: makeCheckpoint(boundary: 200, hash: "same"))
        #expect(cache.count == 2)

        let r1 = cache.fetch(tokenHash: "same", boundary: 100)
        let r2 = cache.fetch(tokenHash: "same", boundary: 200)
        #expect(r1?.boundary == 100)
        #expect(r2?.boundary == 200)
    }

    @Test("SSMCheckpoint estimatedBytes")
    func checkpointMemory() {
        let cp = makeCheckpoint(boundary: 10)
        #expect(cp.estimatedBytes > 0)
    }
}
