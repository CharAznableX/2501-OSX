# Caching Audit

## How It Works

The vmlx-engine supports a multi-layer caching hierarchy, all configured via CLI args built by `VMLXEngineConfig.buildArgs()`.

```
Layer 1: Prefix Cache (L1 — reuse system prompt KV across turns)
  → --enable-prefix-cache, --prefix-cache-size, --cache-memory-percent / --cache-memory-mb
Layer 2: Paged Cache (block-based KV cache management)
  → --use-paged-cache, --paged-cache-block-size, --max-cache-blocks
Layer 3: Disk Cache (persist KV states to disk)
  → --enable-disk-cache, --disk-cache-max-gb
Layer 4: Block Disk Cache (L2 for paged cache)
  → --enable-block-disk-cache, --block-disk-cache-max-gb
Layer 5: KV Cache Quantization (compress KV cache in-memory)
  → --kv-cache-quantization (q4/q8), --kv-cache-group-size
Layer 6: TurboQuant (automatic 3-bit KV compression for JANG models)
  → Auto-detected by engine, not a CLI flag
```

### Configuration → CLI Mapping

**File:** `Services/Inference/VMLXEngineConfig.swift` (lines 22-139)

| ServerConfiguration field | CLI flag | Default | Notes |
|--------------------------|----------|---------|-------|
| `continuousBatching` | `--continuous-batching` | `true` | Required for ALL cache features |
| `maxNumSeqs` | `--max-num-seqs` | `256` | Max concurrent sequences |
| `streamInterval` | `--stream-interval` | `3` | Tokens between SSE events |
| `enablePrefixCache` | `--enable-prefix-cache` / `--disable-prefix-cache` | `true` | |
| `prefixCacheSize` | `--prefix-cache-size` | `100` | Max prefix entries |
| `cacheMemoryPercent` | `--cache-memory-percent` | `0.30` | 30% of RAM |
| `cacheMemoryMB` | `--cache-memory-mb` | `nil` | Fixed MB, overrides percent |
| `cacheTTLMinutes` | `--cache-ttl-minutes` | `0` | 0 = no expiry |
| `usePagedCache` | `--use-paged-cache` | `true` | |
| `pagedCacheBlockSize` | `--paged-cache-block-size` | `64` | Tokens per block |
| `maxCacheBlocks` | `--max-cache-blocks` | `1000` | |
| `enableDiskCache` | `--enable-disk-cache` | `true` | |
| `diskCacheMaxGB` | `--disk-cache-max-gb` | `10.0` | |
| `enableBlockDiskCache` | `--enable-block-disk-cache` | `true` | |
| `blockDiskCacheMaxGB` | `--block-disk-cache-max-gb` | `10.0` | |
| `kvCacheQuantization` | `--kv-cache-quantization` | `"none"` | "none", "q4", "q8" |
| `kvCacheGroupSize` | `--kv-cache-group-size` | `64` | |

**Important:** All cache features require `continuousBatching = true`. The config builder nests cache args inside the `if config.continuousBatching` block (lines 32-79).

### Cache Stats Flow

1. Engine includes `usage` in SSE chunks (when `stream_options.include_usage: true`)
2. `VMLXSSEParser.parse()` extracts `VMLXUsage` with `cachedTokens` from `prompt_tokens_details.cached_tokens` and `cacheDetail`
3. `VMLXService.streamWithTools()` calls `InferenceProgressManager.shared.updateStatsAsync(cached:detail:)` per chunk
4. `FloatingInputCard.inferenceStatsChip` shows green bolt icon + cached token count when `cachedTokens > 0`

### Session ID / Cache Hint

- `session_id` in `ChatCompletionRequest` is passed through to the Python engine for KV cache reuse across conversation turns
- `cache_hint` provides additional cache key context
- Both set in `buildRequestBody()` (VMLXService.swift lines 384-389)

### Prefix Hash

**File:** `Services/Inference/PrefixHash.swift`

- SHA256-based hash of system prompt + tool names
- Used by HTTPHandler to include `prefix_hash` in SSE response headers
- Allows clients to detect when their cache key changes

### KV Quantization UI

**File:** `Views/Settings/ConfigurationView.swift`

- Picker with options: "None (TurboQuant auto)", "4-bit (q4)", "8-bit (q8)"
- "None" means `kvCacheQuantization = "none"` → no `--kv-cache-quantization` flag sent
- TurboQuant auto-detection happens in the engine for JANG-format models (automatic 3-bit KV compression)

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| C1 | **Cache Memory Slider** — `cacheMemoryPercent` default is 0.30 (30%). The settings slider format string was fixed from `"%.0f%%"` to `"%.2f"`. Verify the slider range is 0.0-1.0 and the displayed value makes sense (e.g., "0.30" vs "30%"). |
| C2 | **`cacheMemoryMB` vs `cacheMemoryPercent` priority** — If `cacheMemoryMB` is non-nil, it takes priority (VMLXEngineConfig line 47). But the settings UI allows both to be set. Does the UI clearly indicate which is active? |
| C3 | **Cache features silently disabled when `continuousBatching = false`** — All cache args are inside `if config.continuousBatching` block. If user disables continuous batching, all cache settings are ignored with no warning. |
| C4 | **Paged cache + disk cache interaction during eviction** — When `strictSingleModel` calls `stopAll()`, dirty cache blocks in memory may not be flushed to disk before SIGTERM. Cache corruption possible. |
| C5 | **Cache size growth monitoring** — No UI or logging for current cache size. User can't see if they're hitting the max. Engine may have `/admin/stats` or similar, but Swift doesn't query it. |

### Settings

| # | Issue | Notes |
|---|-------|-------|
| C6 | **`cacheTTLMinutes` default 0** — Means no expiry. For long-running sessions, cache can grow unbounded. Should there be a non-zero default? |
| C7 | **`maxCacheBlocks` default 1000** — At 64 tokens/block, that's 64K tokens max. For large context models (128K+), this may be insufficient. |
| C8 | **Disk cache max 10GB each** — `diskCacheMaxGB` and `blockDiskCacheMaxGB` both default to 10GB. That's 20GB total disk usage. Is this documented? |
| C9 | **KV quantization "none" behavior** — When `kvCacheQuantization == "none"`, no flag is sent. Verify the engine defaults to no quantization (not some default quantization). |

### Verification

| # | Issue | Notes |
|---|-------|-------|
| C10 | **Session ID continuity** — Verify `session_id` is consistently passed from chat session to engine across all turns of a conversation. |
| C11 | **Cache warm on model load** — No pre-warming mechanism. First request after load gets no cache benefit. |
| C12 | **Cache stats accuracy** — Verify `prompt_tokens_details.cached_tokens` is actually populated by the engine (not all models/configs report this). |
| C13 | **TurboQuant auto-detection** — Verify JANG-format models actually trigger automatic 3-bit KV compression without any CLI flag. |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Prefix cache | Working | Enabled by default, CLI args correct |
| Paged cache | Working | Block-based, correct args |
| Disk cache | Working | Both standard and block-level |
| KV quantization | Working | q4/q8 options, "none" for TurboQuant auto |
| Cache stats display | Working | Green bolt + count in stats chip |
| Session ID passthrough | Working | Passed in request body |
| Cache memory slider | **Needs Verify** | Format string fixed, but UX unclear |
| Cache size monitoring | **Missing** | No current-size display |
| Cache warm | **Missing** | No pre-warming mechanism |
