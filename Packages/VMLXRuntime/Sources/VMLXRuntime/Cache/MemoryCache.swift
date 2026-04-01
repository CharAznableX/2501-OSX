import Foundation
import os

// MARK: - Available Memory Helper

/// Estimate available memory on macOS using Mach VM statistics.
/// `os_proc_available_memory()` is iOS-only; this is the cross-platform fallback.
private func _estimateAvailableMemory() -> Int {
    #if os(macOS)
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(
        MemoryLayout<vm_statistics64>.stride / MemoryLayout<integer_t>.stride
    )
    let result = withUnsafeMutablePointer(to: &stats) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            host_statistics64(mach_host_self(), HOST_VM_INFO64, intPtr, &count)
        }
    }
    guard result == KERN_SUCCESS else {
        // Fallback: 50% of physical memory
        return Int(ProcessInfo.processInfo.physicalMemory / 2)
    }
    let pageSize = Int(sysconf(_SC_PAGESIZE))
    let free = Int(stats.free_count) * pageSize
    let inactive = Int(stats.inactive_count) * pageSize
    return free + inactive
    #else
    // iOS / other Apple platforms
    return Int(os_proc_available_memory())
    #endif
}

// MARK: - Configuration

public struct MemoryCacheConfig: Sendable {
    public var maxMemoryMB: Int?        // Explicit MB limit (nil = auto-detect)
    public var maxMemoryPercent: Float   // Fraction of available RAM (default 0.30)
    public var maxEntries: Int           // Hard safety limit (default 1000)
    public var ttlSeconds: TimeInterval  // 0 = disabled

    public init(
        maxMemoryMB: Int? = nil,
        maxMemoryPercent: Float = 0.30,
        maxEntries: Int = 1000,
        ttlSeconds: TimeInterval = 0
    ) {
        self.maxMemoryMB = maxMemoryMB
        self.maxMemoryPercent = maxMemoryPercent
        self.maxEntries = maxEntries
        self.ttlSeconds = ttlSeconds
    }

    /// Compute effective memory limit in bytes.
    public func computeMemoryLimit() -> Int {
        let maxCacheBytes = 32 * 1024 * 1024 * 1024  // 32 GB Metal cap
        let minBytes = 100 * 1024 * 1024  // 100 MB minimum

        if let mb = maxMemoryMB {
            return min(mb * 1024 * 1024, maxCacheBytes)
        }

        let available = _estimateAvailableMemory()
        if available > 0 {
            let limit = Int(Float(available) * maxMemoryPercent)
            return max(min(limit, maxCacheBytes), minBytes)
        }

        // Fallback: assume 4 GB available
        let fallback = Int(Float(4 * 1024 * 1024 * 1024) * maxMemoryPercent)
        return max(min(fallback, maxCacheBytes), minBytes)
    }
}

// MARK: - Cache Entry

private struct CacheEntry {
    let tokens: [Int]
    let cache: HybridCache
    let memoryBytes: Int
    var lastAccessed: CFAbsoluteTime

    mutating func touch() {
        lastAccessed = CFAbsoluteTimeGetCurrent()
    }
}

// MARK: - Memory-Aware Prefix Cache

/// RAM-budget-aware LRU cache with memory pressure adaptation.
/// Checks os_proc_available_memory() every 60s, shrinks budget under pressure.
public final class MemoryCache: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock()
    private let config: MemoryCacheConfig

    // Entries in insertion/access order (most recent at end)
    private var entries: [(key: [Int], entry: CacheEntry)] = []

    private var baseMemoryLimit: Int
    private var effectiveMemoryLimit: Int
    private var currentMemory: Int = 0
    private var lastPressureCheck: CFAbsoluteTime = 0

    // Stats
    public private(set) var hits: Int = 0
    public private(set) var misses: Int = 0
    public private(set) var evictions: Int = 0

    public init(config: MemoryCacheConfig = MemoryCacheConfig()) {
        self.config = config
        let limit = config.computeMemoryLimit()
        self.baseMemoryLimit = limit
        self.effectiveMemoryLimit = limit
    }

    /// Clear all cached entries.
    public func clear() {
        lock.withLock { entries.removeAll() }
    }

    /// Fetch cache for token sequence.
    /// Returns (cache, remainingTokens). nil cache = miss.
    public func fetch(tokens: [Int]) -> (HybridCache?, [Int]) {
        lock.withLock {
            _evictExpired()

            let tokensKey = tokens

            // Exact match
            if let idx = entries.firstIndex(where: { $0.key == tokensKey }) {
                entries[idx].entry.touch()
                let entry = entries.remove(at: idx)
                entries.append(entry)  // Move to end (MRU)
                hits += 1
                return (entry.entry.cache, [])
            }

            // Prefix scan: find best shorter prefix (cached is prefix of request)
            var bestForwardIdx: Int? = nil
            var bestForwardLen: Int = 0

            // Also check for longer match (request is prefix of cached)
            var bestReverseIdx: Int? = nil
            var bestReverseLen: Int = 0

            for (i, entry) in entries.enumerated() {
                let cachedKey = entry.key
                let cachedLen = cachedKey.count
                let requestLen = tokensKey.count

                if cachedLen <= requestLen {
                    // Check if cached is prefix of request
                    if cachedLen > bestForwardLen && Array(tokensKey.prefix(cachedLen)) == cachedKey {
                        bestForwardIdx = i
                        bestForwardLen = cachedLen
                    }
                } else {
                    // Check if request is prefix of cached
                    if cachedLen > bestReverseLen && Array(cachedKey.prefix(requestLen)) == tokensKey {
                        bestReverseIdx = i
                        bestReverseLen = cachedLen
                    }
                }
            }

            // Prefer forward match (shorter prefix)
            if let idx = bestForwardIdx {
                entries[idx].entry.touch()
                let entry = entries.remove(at: idx)
                entries.append(entry)
                hits += 1
                return (entry.entry.cache, Array(tokensKey[bestForwardLen...]))
            }

            // Try reverse match (longer cached, truncate to request length)
            if let idx = bestReverseIdx {
                let cached = entries[idx].entry.cache
                if cached.canTruncate {
                    if let truncated = cached.truncated(to: tokensKey.count) {
                        entries[idx].entry.touch()
                        let entry = entries.remove(at: idx)
                        entries.append(entry)
                        hits += 1
                        return (truncated, [])
                    }
                }
                // Can't truncate (hybrid model) - fall through to miss
            }

            misses += 1
            return (nil, tokens)
        }
    }

    /// Store cache for token sequence. Returns false if too large.
    public func store(tokens: [Int], cache: HybridCache) -> Bool {
        lock.withLock {
            _checkMemoryPressure()
            _evictExpired()

            let tokensKey = tokens

            // Already cached? Update
            if let idx = entries.firstIndex(where: { $0.key == tokensKey }) {
                let oldEntry = entries.remove(at: idx)
                currentMemory -= oldEntry.entry.memoryBytes
            }

            let memoryBytes = cache.estimatedBytes

            // Too large for cache entirely?
            let threshold = Int(Float(effectiveMemoryLimit) * 0.95)
            if memoryBytes > threshold { return false }

            // Evict until there's room
            while (currentMemory + memoryBytes > effectiveMemoryLimit || entries.count >= config.maxEntries) && !entries.isEmpty {
                _evictLRU()
            }

            let entry = CacheEntry(
                tokens: tokensKey,
                cache: cache,
                memoryBytes: memoryBytes,
                lastAccessed: CFAbsoluteTimeGetCurrent()
            )
            entries.append((key: tokensKey, entry: entry))
            currentMemory += memoryBytes
            return true
        }
    }

    public var count: Int {
        lock.withLock { entries.count }
    }

    public var memoryUsed: Int {
        lock.withLock { currentMemory }
    }

    // MARK: - Private

    private func _checkMemoryPressure() {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastPressureCheck > 60 else { return }
        lastPressureCheck = now

        let available = _estimateAvailableMemory()
        guard available > 0 else { return }

        let totalRAM = ProcessInfo.processInfo.physicalMemory
        let availablePercent = Float(available) / Float(totalRAM)

        if availablePercent < 0.20 {
            effectiveMemoryLimit = Int(available) / 2
        } else {
            effectiveMemoryLimit = baseMemoryLimit
        }
    }

    private func _evictLRU() {
        guard !entries.isEmpty else { return }
        let removed = entries.removeFirst()
        currentMemory -= removed.entry.memoryBytes
        evictions += 1
    }

    private func _evictExpired() {
        guard config.ttlSeconds > 0 else { return }
        let now = CFAbsoluteTimeGetCurrent()
        let cutoff = now - config.ttlSeconds

        while let first = entries.first, first.entry.lastAccessed < cutoff {
            let removed = entries.removeFirst()
            currentMemory -= removed.entry.memoryBytes
            evictions += 1
        }
    }
}
