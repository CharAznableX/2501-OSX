import Testing
@testable import VMLXRuntime

@Suite("PagedCacheManager")
struct PagedCacheManagerTests {
    @Test("Allocate and free")
    func allocateAndFree() {
        let mgr = PagedCacheManager(blockSize: 64, maxBlocks: 100)
        #expect(mgr.stats.freeBlocks == 99)
        let block = mgr.allocateBlock()
        #expect(block != nil)
        #expect(block!.refCount == 1)
        #expect(mgr.stats.allocatedBlocks == 1)
        mgr.freeBlock(block!)
        #expect(mgr.stats.freeBlocks == 99)
    }

    @Test("Allocate by token count")
    func allocateByTokens() {
        let mgr = PagedCacheManager(blockSize: 64, maxBlocks: 100)
        let blocks = mgr.allocateBlocksByTokens(150) // ceil(150/64) = 3
        #expect(blocks.count == 3)
    }

    @Test("COW fork shares block")
    func cowFork() {
        let mgr = PagedCacheManager(blockSize: 64, maxBlocks: 100)
        let block = mgr.allocateBlock()!
        let hash = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [1, 2, 3])
        mgr.markCached(block: block, hash: hash)
        let forked = mgr.forkBlock(block, hash: hash)
        #expect(forked.blockId == block.blockId)
        #expect(block.refCount == 2)
    }

    @Test("Find cached block")
    func findCached() {
        let mgr = PagedCacheManager(blockSize: 64, maxBlocks: 100)
        let block = mgr.allocateBlock()!
        let hash = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [1, 2, 3])
        mgr.markCached(block: block, hash: hash)
        #expect(mgr.findCachedBlock(hash: hash)?.blockId == block.blockId)
    }

    @Test("Eviction when pool exhausted")
    func eviction() {
        let mgr = PagedCacheManager(blockSize: 64, maxBlocks: 5)
        // maxBlocks=5 means blocks 0(null),1,2,3,4 — 4 allocatable
        var blocks: [CacheBlock] = []
        for _ in 0..<4 {
            if let b = mgr.allocateBlock() { blocks.append(b) }
        }
        #expect(mgr.stats.freeBlocks == 0)

        // Mark all as cached and free them (refCount goes to 0, returns to free queue).
        // But we want to test eviction, so mark cached but keep allocated (refCount=1).
        // Re-allocate fresh:
        let mgr2 = PagedCacheManager(blockSize: 64, maxBlocks: 5)
        var blocks2: [CacheBlock] = []
        for i in 0..<4 {
            if let b = mgr2.allocateBlock() {
                let hash = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [i])
                mgr2.markCached(block: b, hash: hash)
                blocks2.append(b)
            }
        }
        #expect(mgr2.stats.freeBlocks == 0)

        // Now allocate one more — should evict the oldest cached block.
        let newBlock = mgr2.allocateBlock()
        #expect(newBlock != nil)
        #expect(mgr2.stats.evictions >= 1)
    }

    @Test("Delete block table frees blocks")
    func deleteBlockTable() {
        let mgr = PagedCacheManager(blockSize: 64, maxBlocks: 100)
        let blocks = mgr.allocateBlocksByTokens(200) // ceil(200/64) = 4
        mgr.registerBlockTable("req-1", blockIds: blocks.map(\.blockId))
        mgr.deleteBlockTable("req-1")
        #expect(mgr.stats.allocatedBlocks == 0)
    }

    @Test("Cache hit and miss stats")
    func cacheHitMissStats() {
        let mgr = PagedCacheManager(blockSize: 64, maxBlocks: 100)
        let block = mgr.allocateBlock()!
        let hash = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [10, 20])
        mgr.markCached(block: block, hash: hash)

        _ = mgr.findCachedBlock(hash: hash)
        #expect(mgr.stats.cacheHits == 1)

        let missingHash = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [99])
        _ = mgr.findCachedBlock(hash: missingHash)
        #expect(mgr.stats.cacheMisses == 1)
    }

    @Test("Allocate zero tokens returns empty")
    func allocateZeroTokens() {
        let mgr = PagedCacheManager(blockSize: 64, maxBlocks: 100)
        let blocks = mgr.allocateBlocksByTokens(0)
        #expect(blocks.isEmpty)
    }

    @Test("Fork increments COW stats")
    func forkCowStats() {
        let mgr = PagedCacheManager(blockSize: 64, maxBlocks: 100)
        let block = mgr.allocateBlock()!
        let hash = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [5])
        _ = mgr.forkBlock(block, hash: hash)
        _ = mgr.forkBlock(block, hash: hash)
        #expect(mgr.stats.cowCopies == 2)
        #expect(block.refCount == 3) // 1 original + 2 forks
    }
}
