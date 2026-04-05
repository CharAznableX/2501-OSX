//
//  MLXGenerationEngineTests.swift
//  project2501Tests
//
//  Tests for free functions and helpers in MLXGenerationEngine.swift
//  that can be exercised without Metal / GPU.
//

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import Project2501Core

struct MLXGenerationEngineTests {

    // MARK: effectiveCacheOffset

    /// All layers are MambaCache (ArraysCache subclass) → offset is always 0 for all.
    /// effectiveCacheOffset must fall through to the last branch and return 0.
    @Test func effectiveCacheOffset_allArraysCache_returnsZero() {
        let layers: [any KVCache] = [MambaCache(), MambaCache(), MambaCache()]
        #expect(effectiveCacheOffset(layers) == 0)
    }

    /// Empty cache array → returns 0.
    @Test func effectiveCacheOffset_emptyArray_returnsZero() {
        #expect(effectiveCacheOffset([]) == 0)
    }

    /// First layer is a KVCacheSimple with offset set — should return that offset.
    @Test func effectiveCacheOffset_firstLayerIsKVCacheSimple_returnsItsOffset() {
        let kv = KVCacheSimple()
        // Inject state so offset becomes 4 (key dim(2) == 4).
        kv.state = [MLX.zeros([1, 1, 4, 1]), MLX.zeros([1, 1, 4, 1])]
        #expect(kv.offset == 4)

        let layers: [any KVCache] = [kv]
        #expect(effectiveCacheOffset(layers) == 4)
    }

    /// Hybrid: MambaCache layers come first (offset 0), then a KVCacheSimple with
    /// a real offset.  effectiveCacheOffset must skip Mamba layers and return the
    /// KVCacheSimple's offset.
    @Test func effectiveCacheOffset_hybridCache_skipsArraysCacheReturnsKVOffset() {
        let mamba1 = MambaCache()
        let mamba2 = MambaCache()
        let mamba3 = MambaCache()

        let kv = KVCacheSimple()
        kv.state = [MLX.zeros([1, 1, 8, 1]), MLX.zeros([1, 1, 8, 1])]
        #expect(kv.offset == 8)

        let layers: [any KVCache] = [mamba1, mamba2, mamba3, kv]
        #expect(effectiveCacheOffset(layers) == 8)
    }

    /// If there are multiple KVCacheSimple layers, the first one's offset is returned.
    @Test func effectiveCacheOffset_multipleKVLayers_returnsFirstOffset() {
        let kv1 = KVCacheSimple()
        kv1.state = [MLX.zeros([1, 1, 4, 1]), MLX.zeros([1, 1, 4, 1])]

        let kv2 = KVCacheSimple()
        kv2.state = [MLX.zeros([1, 1, 16, 1]), MLX.zeros([1, 1, 16, 1])]

        let layers: [any KVCache] = [kv1, kv2]
        #expect(effectiveCacheOffset(layers) == 4)
    }

    /// Hybrid with KVCacheSimple that has no state yet (offset 0).
    /// Should return 0 from that layer (not skip it).
    @Test func effectiveCacheOffset_hybridWithFreshKVLayer_returnsZero() {
        let mamba = MambaCache()
        let kv = KVCacheSimple()  // fresh, offset == 0
        let layers: [any KVCache] = [mamba, kv]
        #expect(effectiveCacheOffset(layers) == 0)
    }
}
