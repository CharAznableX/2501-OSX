import Foundation
import os

/// Statistics tracking for the paged cache manager.
public struct CacheStats: Sendable {
    public var totalBlocks: Int = 0
    public var allocatedBlocks: Int = 0
    public var freeBlocks: Int = 0
    public var cacheHits: Int = 0
    public var cacheMisses: Int = 0
    public var cowCopies: Int = 0
    public var evictions: Int = 0
    public var diskHits: Int = 0
    public var diskMisses: Int = 0
}

/// Thread-safe paged cache manager with block allocation, COW forking,
/// hash-based prefix reuse, and LRU eviction.
///
/// Block 0 is reserved as a null sentinel and is never allocated.
/// Blocks 1..<maxBlocks are available for allocation via the free queue.
public final class PagedCacheManager: @unchecked Sendable {
    public let blockSize: Int
    public let maxBlocks: Int

    private let lock = OSAllocatedUnfairLock()
    private var blocks: [CacheBlock]
    private var freeQueue: FreeBlockQueue
    private var hashMap: BlockHashMap
    private var requestTables: [String: [Int]] = [:]
    private var allocatedBlocks: [Int: CacheBlock] = [:]
    private let nullBlock: CacheBlock

    public private(set) var stats: CacheStats

    public init(blockSize: Int = 64, maxBlocks: Int = 1000) {
        self.blockSize = blockSize
        self.maxBlocks = maxBlocks
        self.nullBlock = CacheBlock(blockId: 0, blockSize: blockSize)
        self.hashMap = BlockHashMap()
        self.freeQueue = FreeBlockQueue()

        // Create block pool: block 0 is the null sentinel, blocks 1..<maxBlocks are free.
        var pool = [CacheBlock]()
        pool.reserveCapacity(maxBlocks)
        pool.append(nullBlock)
        for i in 1..<maxBlocks {
            let block = CacheBlock(blockId: i, blockSize: blockSize)
            pool.append(block)
        }
        self.blocks = pool

        // Enqueue all non-sentinel blocks into the free queue.
        for i in 1..<maxBlocks {
            freeQueue.append(pool[i])
        }

        self.stats = CacheStats()
        self.stats.totalBlocks = maxBlocks
        self.stats.freeBlocks = maxBlocks - 1  // exclude null sentinel
    }

    /// Clear all cached data. Resets all blocks to free state.
    public func clear() {
        lock.withLock {
            hashMap.removeAll()
            requestTables.removeAll()
            allocatedBlocks.removeAll()
            freeQueue = FreeBlockQueue()
            for i in 1..<blocks.count {
                blocks[i].reset()
                freeQueue.append(blocks[i])
            }
            stats.freeBlocks = blocks.count - 1
        }
    }

    // MARK: - Allocation

    /// Allocate a single block from the free queue. Returns nil if no blocks are available
    /// (after attempting eviction).
    public func allocateBlock() -> CacheBlock? {
        lock.withLock {
            _allocateBlock()
        }
    }

    /// Internal allocation (must be called under lock).
    private func _allocateBlock() -> CacheBlock? {
        var block = freeQueue.popleft()

        // If the free queue is empty, try to evict a cached block.
        if block == nil {
            block = _evictCachedBlock()
        }

        guard let block else { return nil }

        block.reset()
        block.refCount = 1
        block.touch()
        allocatedBlocks[block.blockId] = block
        stats.freeBlocks = freeQueue.count
        stats.allocatedBlocks = allocatedBlocks.count
        return block
    }

    /// Allocate enough blocks to hold the given number of tokens.
    /// Returns ceil(tokens / blockSize) blocks.
    public func allocateBlocksByTokens(_ tokens: Int) -> [CacheBlock] {
        guard tokens > 0 else { return [] }
        let needed = (tokens + blockSize - 1) / blockSize
        return lock.withLock {
            var result = [CacheBlock]()
            result.reserveCapacity(needed)
            for _ in 0..<needed {
                guard let block = _allocateBlock() else { break }
                result.append(block)
            }
            return result
        }
    }

    // MARK: - Deallocation

    /// Free a block. Decrements refCount; when it reaches 0 the block is returned to the free queue.
    public func freeBlock(_ block: CacheBlock) {
        lock.withLock {
            block.refCount -= 1
            if block.refCount <= 0 {
                block.refCount = 0
                // Remove from allocated tracking.
                allocatedBlocks.removeValue(forKey: block.blockId)
                // Remove any hash mapping for this block.
                if let hash = block.blockHash {
                    _ = hashMap.pop(hash: hash, blockId: block.blockId)
                }
                block.reset()
                freeQueue.append(block)
                stats.freeBlocks = freeQueue.count
                stats.allocatedBlocks = allocatedBlocks.count
            }
        }
    }

    // MARK: - COW Fork

    /// Fork a block for copy-on-write sharing. Increments the block's reference count
    /// and returns the same block (shared, not copied). Tracks the COW copy in stats.
    @discardableResult
    public func forkBlock(_ block: CacheBlock, hash: BlockHash) -> CacheBlock {
        lock.withLock {
            block.refCount += 1
            block.blockHash = hash
            block.touch()
            stats.cowCopies += 1
            return block
        }
    }

    // MARK: - Hash Prefix Cache

    /// Mark a block as cached under the given hash for prefix-sharing lookups.
    public func markCached(block: CacheBlock, hash: BlockHash) {
        lock.withLock {
            block.blockHash = hash
            hashMap.insert(hash: hash, block: block)
        }
    }

    /// Look up a cached block by hash. Updates cache hit/miss stats.
    public func findCachedBlock(hash: BlockHash) -> CacheBlock? {
        lock.withLock {
            if let block = hashMap.getBlock(hash: hash) {
                block.touch()
                stats.cacheHits += 1
                return block
            } else {
                stats.cacheMisses += 1
                return nil
            }
        }
    }

    // MARK: - Block Table Management

    /// Register a block table for a request, associating block IDs with the request.
    public func registerBlockTable(_ requestId: String, blockIds: [Int]) {
        lock.withLock {
            requestTables[requestId] = blockIds
        }
    }

    /// Delete a block table for a request, freeing all associated blocks by decrementing
    /// their reference counts.
    public func deleteBlockTable(_ requestId: String) {
        lock.withLock {
            guard let blockIds = requestTables.removeValue(forKey: requestId) else { return }
            for blockId in blockIds {
                guard blockId > 0, blockId < blocks.count else { continue }
                let block = blocks[blockId]
                block.refCount -= 1
                if block.refCount <= 0 {
                    block.refCount = 0
                    allocatedBlocks.removeValue(forKey: block.blockId)
                    if let hash = block.blockHash {
                        _ = hashMap.pop(hash: hash, blockId: block.blockId)
                    }
                    block.reset()
                    freeQueue.append(block)
                }
            }
            stats.freeBlocks = freeQueue.count
            stats.allocatedBlocks = allocatedBlocks.count
        }
    }

    // MARK: - Eviction

    /// Evict the oldest hash-cached block with refCount <= 1. Returns the evicted block
    /// (already removed from the hash map and reset), or nil if nothing is evictable.
    /// Must be called under lock.
    private func _evictCachedBlock() -> CacheBlock? {
        // Scan all allocated blocks to find the one with the oldest lastAccess
        // that is hash-cached and has refCount <= 1 (not shared).
        var candidate: CacheBlock?
        var oldestDate = Date.distantFuture

        for (_, block) in allocatedBlocks {
            guard block.blockHash != nil, block.refCount <= 1 else { continue }
            if block.lastAccess < oldestDate {
                oldestDate = block.lastAccess
                candidate = block
            }
        }

        guard let victim = candidate else { return nil }

        // Remove from hash map.
        if let hash = victim.blockHash {
            _ = hashMap.pop(hash: hash, blockId: victim.blockId)
        }

        // Remove from allocated tracking.
        allocatedBlocks.removeValue(forKey: victim.blockId)

        // Reset for reuse.
        victim.reset()
        stats.evictions += 1
        stats.allocatedBlocks = allocatedBlocks.count
        // Note: we don't put it in the free queue — we return it directly for allocation.
        return victim
    }
}
