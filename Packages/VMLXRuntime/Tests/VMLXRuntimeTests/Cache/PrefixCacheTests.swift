import Testing
import MLX
@testable import VMLXRuntime

@Suite("PrefixCache")
struct PrefixCacheTests {

    /// Build a pure-attention HybridCache with the given number of tokens (4 layers).
    private func makeCache(tokenCount: Int) -> HybridCache {
        let layers: [LayerCacheEntry] = (0..<4).map { _ in
            .attention(KVCacheLayer(
                keys: MLXArray.zeros([1, 8, tokenCount, 128]),
                values: MLXArray.zeros([1, 8, tokenCount, 128]),
                offset: tokenCount
            ))
        }
        return HybridCache(layers: layers)
    }

    @Test("Exact match returns cache with no remaining tokens")
    func exactMatch() {
        let pc = PrefixCache(maxEntries: 10)
        let cache = makeCache(tokenCount: 5)
        pc.store(tokens: [1, 2, 3, 4, 5], cache: cache)

        let (result, remaining) = pc.fetch(tokens: [1, 2, 3, 4, 5])
        #expect(result != nil)
        #expect(remaining.isEmpty)
        #expect(pc.hits == 1)
    }

    @Test("Shorter prefix match returns cache and remaining tokens")
    func shorterPrefix() {
        let pc = PrefixCache(maxEntries: 10)
        let cache = makeCache(tokenCount: 3)
        pc.store(tokens: [1, 2, 3], cache: cache)

        let (result, remaining) = pc.fetch(tokens: [1, 2, 3, 4, 5])
        #expect(result != nil)
        #expect(remaining == [4, 5])
    }

    @Test("Complete miss returns nil and all tokens")
    func miss() {
        let pc = PrefixCache(maxEntries: 10)
        let cache = makeCache(tokenCount: 3)
        pc.store(tokens: [1, 2, 3], cache: cache)

        let (result, remaining) = pc.fetch(tokens: [7, 8, 9])
        #expect(result == nil)
        #expect(remaining == [7, 8, 9])
        #expect(pc.misses == 1)
    }

    @Test("LRU eviction removes oldest entry")
    func lruEviction() {
        let pc = PrefixCache(maxEntries: 2)
        pc.store(tokens: [1], cache: makeCache(tokenCount: 1))
        pc.store(tokens: [2], cache: makeCache(tokenCount: 1))
        pc.store(tokens: [3], cache: makeCache(tokenCount: 1))  // Should evict [1]

        let (r1, _) = pc.fetch(tokens: [1])
        #expect(r1 == nil)  // Evicted
        let (r2, _) = pc.fetch(tokens: [2])
        #expect(r2 != nil)  // Still present
    }

    @Test("Longer prefix with pure attention cache truncates successfully")
    func longerPrefixTruncation() {
        let pc = PrefixCache(maxEntries: 10)
        // Store a 5-token cache (pure attention, can truncate)
        let cache = makeCache(tokenCount: 5)
        pc.store(tokens: [1, 2, 3, 4, 5], cache: cache)

        // Request a 3-token prefix -- the cached entry is longer
        let (result, remaining) = pc.fetch(tokens: [1, 2, 3])
        #expect(result != nil)
        #expect(remaining.isEmpty)
        #expect(pc.hits == 1)
    }

    @Test("Hybrid cache (SSM layers) refuses truncation for longer prefix")
    func hybridRefusesTruncation() {
        let pc = PrefixCache(maxEntries: 10)
        // Create a hybrid cache with SSM layers that cannot be truncated
        let layers: [LayerCacheEntry] = [
            .attention(KVCacheLayer(
                keys: MLXArray.zeros([1, 8, 5, 128]),
                values: MLXArray.zeros([1, 8, 5, 128]),
                offset: 5
            )),
            .ssm(SSMStateLayer(state: [MLXArray.zeros([1, 16, 256])])),
        ]
        let hybridCache = HybridCache(layers: layers)
        pc.store(tokens: [1, 2, 3, 4, 5], cache: hybridCache)

        // Request shorter prefix -- would need truncation of the longer cached entry,
        // but hybrid (SSM) caches cannot truncate
        let (result, remaining) = pc.fetch(tokens: [1, 2, 3])
        #expect(result == nil)
        #expect(remaining == [1, 2, 3])
        #expect(pc.misses == 1)
    }

    @Test("Invalidate removes entry")
    func invalidate() {
        let pc = PrefixCache(maxEntries: 10)
        pc.store(tokens: [1, 2, 3], cache: makeCache(tokenCount: 3))
        #expect(pc.count == 1)

        pc.invalidate(tokens: [1, 2, 3])
        #expect(pc.count == 0)

        let (result, _) = pc.fetch(tokens: [1, 2, 3])
        #expect(result == nil)
    }

    @Test("Store overwrites existing entry without duplicating count")
    func overwrite() {
        let pc = PrefixCache(maxEntries: 10)
        pc.store(tokens: [1, 2, 3], cache: makeCache(tokenCount: 3))
        pc.store(tokens: [1, 2, 3], cache: makeCache(tokenCount: 3))
        #expect(pc.count == 1)
    }

    @Test("Multiple disjoint entries coexist")
    func multipleEntries() {
        let pc = PrefixCache(maxEntries: 10)
        pc.store(tokens: [1, 2], cache: makeCache(tokenCount: 2))
        pc.store(tokens: [3, 4], cache: makeCache(tokenCount: 2))
        pc.store(tokens: [5, 6], cache: makeCache(tokenCount: 2))
        #expect(pc.count == 3)

        let (r1, _) = pc.fetch(tokens: [1, 2])
        let (r2, _) = pc.fetch(tokens: [3, 4])
        let (r3, _) = pc.fetch(tokens: [5, 6])
        #expect(r1 != nil)
        #expect(r2 != nil)
        #expect(r3 != nil)
    }

    @Test("LRU touch promotes accessed entry")
    func lruTouch() {
        let pc = PrefixCache(maxEntries: 2)
        pc.store(tokens: [1], cache: makeCache(tokenCount: 1))
        pc.store(tokens: [2], cache: makeCache(tokenCount: 1))

        // Touch [1] to promote it
        let _ = pc.fetch(tokens: [1])

        // Insert [3] -- should evict [2] (now oldest), not [1]
        pc.store(tokens: [3], cache: makeCache(tokenCount: 1))

        let (r1, _) = pc.fetch(tokens: [1])
        #expect(r1 != nil)  // Promoted, not evicted
        let (r2, _) = pc.fetch(tokens: [2])
        #expect(r2 == nil)  // Evicted
    }
}
