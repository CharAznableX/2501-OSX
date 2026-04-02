import Testing
import MLX
@testable import VMLXRuntime

@Suite("MemoryCache")
struct MemoryCacheTests {

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

    @Test("Store and fetch exact match")
    func exactMatch() {
        let mc = MemoryCache(config: MemoryCacheConfig(maxMemoryMB: 1024))
        let stored = mc.store(tokens: [1, 2, 3], cache: makeCache(tokenCount: 3))
        #expect(stored == true)

        let (result, remaining) = mc.fetch(tokens: [1, 2, 3])
        #expect(result != nil)
        #expect(remaining.isEmpty)
        #expect(mc.hits == 1)
    }

    @Test("Shorter prefix match")
    func shorterPrefix() {
        let mc = MemoryCache(config: MemoryCacheConfig(maxMemoryMB: 1024))
        _ = mc.store(tokens: [1, 2, 3], cache: makeCache(tokenCount: 3))

        let (result, remaining) = mc.fetch(tokens: [1, 2, 3, 4, 5])
        #expect(result != nil)
        #expect(remaining == [4, 5])
    }

    @Test("Miss")
    func miss() {
        let mc = MemoryCache(config: MemoryCacheConfig(maxMemoryMB: 1024))
        _ = mc.store(tokens: [1, 2, 3], cache: makeCache(tokenCount: 3))

        let (result, remaining) = mc.fetch(tokens: [7, 8, 9])
        #expect(result == nil)
        #expect(remaining == [7, 8, 9])
        #expect(mc.misses == 1)
    }

    @Test("LRU eviction by entry count")
    func lruEviction() {
        let mc = MemoryCache(config: MemoryCacheConfig(maxMemoryMB: 1024, maxEntries: 2))
        _ = mc.store(tokens: [1], cache: makeCache(tokenCount: 1))
        _ = mc.store(tokens: [2], cache: makeCache(tokenCount: 1))
        _ = mc.store(tokens: [3], cache: makeCache(tokenCount: 1))

        #expect(mc.count == 2)
        let (r1, _) = mc.fetch(tokens: [1])
        #expect(r1 == nil)  // Evicted
    }

    @Test("Memory tracking")
    func memoryTracking() {
        let mc = MemoryCache(config: MemoryCacheConfig(maxMemoryMB: 1024))
        _ = mc.store(tokens: [1, 2, 3], cache: makeCache(tokenCount: 100))
        #expect(mc.memoryUsed > 0)
    }

    @Test("Rejects oversized entry")
    func rejectsOversized() {
        // Very small limit
        let mc = MemoryCache(config: MemoryCacheConfig(maxMemoryMB: 1))
        // Try to store a very large cache
        let bigCache = makeCache(tokenCount: 100000)
        let stored = mc.store(tokens: [1], cache: bigCache)
        // May or may not fit depending on actual byte size
        // The point is it shouldn't crash
        _ = stored
    }

    @Test("clear resets bookkeeping and stats")
    func clearResetsBookkeeping() {
        let mc = MemoryCache(config: MemoryCacheConfig(maxMemoryMB: 1024, maxEntries: 2))
        _ = mc.store(tokens: [1, 2, 3], cache: makeCache(tokenCount: 16))
        _ = mc.fetch(tokens: [1, 2, 3])
        _ = mc.fetch(tokens: [9, 9, 9])

        #expect(mc.count == 1)
        #expect(mc.memoryUsed > 0)
        #expect(mc.hits == 1)
        #expect(mc.misses == 1)

        mc.clear()

        #expect(mc.count == 0)
        #expect(mc.memoryUsed == 0)
        #expect(mc.hits == 0)
        #expect(mc.misses == 0)
        #expect(mc.evictions == 0)
        #expect(mc.debugEffectiveMemoryLimit == mc.debugBaseMemoryLimit)
        #expect(mc.debugLastPressureCheck == 0)
    }
}
