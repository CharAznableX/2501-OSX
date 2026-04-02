import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("DiskCache")
struct DiskCacheTests {

    private func makeTempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func makeHybridCache(tokenCount: Int) -> HybridCache {
        let layers: [LayerCacheEntry] = (0..<2).map { _ in
            .attention(KVCacheLayer(
                keys: MLXArray.zeros([1, 2, tokenCount, 8]),
                values: MLXArray.zeros([1, 2, tokenCount, 8]),
                offset: tokenCount
            ))
        }
        return HybridCache(layers: layers)
    }

    @Test("Token hashing is deterministic")
    func tokenHashing() {
        let h1 = DiskCache.hashTokens([1, 2, 3])
        let h2 = DiskCache.hashTokens([1, 2, 3])
        #expect(h1 == h2)
        #expect(h1.count == 64)  // SHA-256 hex = 64 chars

        let h3 = DiskCache.hashTokens([3, 2, 1])
        #expect(h1 != h3)
    }

    @Test("Store and fetch")
    func storeAndFetch() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let dc = DiskCache(cacheDir: dir, maxSizeGB: 1.0)
        let stored = dc.store(tokens: [1, 2, 3], numTokens: 3, fileSize: 1024)
        #expect(stored == true)
        #expect(dc.stores == 1)

        let result = dc.fetch(tokens: [1, 2, 3])
        #expect(result != nil)
        #expect(result?.numTokens == 3)
        #expect(dc.hits == 1)
    }

    @Test("Miss returns nil")
    func miss() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let dc = DiskCache(cacheDir: dir)
        let result = dc.fetch(tokens: [99, 98, 97])
        #expect(result == nil)
        #expect(dc.misses == 1)
    }

    @Test("Contains check")
    func containsCheck() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let dc = DiskCache(cacheDir: dir)
        #expect(dc.contains(tokens: [1, 2]) == false)
        _ = dc.store(tokens: [1, 2], numTokens: 2)
        #expect(dc.contains(tokens: [1, 2]) == true)
    }

    @Test("Remove deletes entry")
    func removeEntry() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let dc = DiskCache(cacheDir: dir)
        _ = dc.store(tokens: [5, 6, 7], numTokens: 3)
        #expect(dc.entryCount == 1)
        dc.remove(tokens: [5, 6, 7])
        #expect(dc.entryCount == 0)
    }

    @Test("Entry count tracks correctly")
    func entryCount() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let dc = DiskCache(cacheDir: dir)
        _ = dc.store(tokens: [1], numTokens: 1)
        _ = dc.store(tokens: [2], numTokens: 1)
        _ = dc.store(tokens: [3], numTokens: 1)
        #expect(dc.entryCount == 3)
    }

    @Test("Duplicate store updates access, doesn't duplicate")
    func duplicateStore() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let dc = DiskCache(cacheDir: dir)
        _ = dc.store(tokens: [1, 2, 3], numTokens: 3)
        _ = dc.store(tokens: [1, 2, 3], numTokens: 3)
        #expect(dc.entryCount == 1)
    }

    @Test("Metadata round-trips")
    func metadataRoundTrip() {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let dc = DiskCache(cacheDir: dir)
        _ = dc.store(tokens: [10, 20], numTokens: 2, metadata: "__tq_native__=true")
        let result = dc.fetch(tokens: [10, 20])
        #expect(result?.metadata == "__tq_native__=true")
    }

    @Test("clear prevents stale background writes from recreating files")
    func clearPreventsStaleWriteCommit() async {
        let dir = makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tokens = [11, 22, 33]
        let hash = DiskCache.hashTokens(tokens)
        let fileURL = dir.appendingPathComponent("\(hash).safetensors")

        let dc = DiskCache(cacheDir: dir)
        dc.testWriteDelayNanoseconds = 50_000_000
        dc.storeCache(tokens: tokens, cache: makeHybridCache(tokenCount: tokens.count))
        #expect(dc.pendingWriteCount == 1)

        dc.clear()
        await dc.waitForPendingWrites()

        #expect(dc.entryCount == 0)
        #expect(!FileManager.default.fileExists(atPath: fileURL.path))
    }
}
