# Performance Audit

## How It Works

### Streaming Pipeline Overhead

The streaming pipeline adds overhead at each layer:

```
Python engine: generates at ~80+ tok/s
  → SSE over HTTP (stream-interval=3: ~27 events/sec)
    → URLSession.bytes.lines (~27 line reads/sec)
      → VMLXSSEParser.parse() per line
        → JSON deserialization per line
          → VMLXService batching (30ms window → ~33 yields/sec max)
            → ChatEngine wrapStreamWithLogging (Task.detached, no overhead)
              → StreamingDeltaProcessor (adaptive flush 50-150ms)
                → ChatTurn.appendContent/appendThinking
                  → NSTableView reconfigureRows (100-250ms sync cadence)
```

### Optimizations Applied (Previous Session)

1. **`bytes.lines` instead of byte-by-byte** — Reduced from ~3000 async suspensions/sec to ~27/sec
2. **Content batching in VMLXService** — 30ms windows reduce MainActor async hops
3. **`CFAbsoluteTimeGetCurrent()` in VMLXService** — Avoids `Date()` syscall overhead
4. **`deltasSinceLastCheck` in StreamingDeltaProcessor** — Only calls `Date()` every 4th delta
5. **Removed per-delta `Date()` in ChatEngine** — Was timing every delta in wrapStreamWithLogging
6. **`stream-interval` default 3** — Engine sends every 3rd token instead of every token
7. **Adaptive sync cadence** — 100ms (short) → 250ms (long) based on total output length

### Key Metrics

| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| Engine TPS | 80+ tok/s | Model-dependent | Measured by engine |
| SSE events/sec | ~27/sec | stream-interval=3 | 80/3 ≈ 27 |
| Async suspensions/sec | <50 | ~27 (bytes.lines) | Was 3000 before fix |
| VMLXService yields/sec | ~33 max | 30ms batch interval | Batched content |
| UI sync/sec | 4-10 | 100-250ms cadence | Adaptive |
| TTFT (warm engine) | <100ms | Model-dependent | Engine TTFB is ~1ms |
| TTFT (cold engine) | 1-120s | Depends on model size | Includes model loading |

### Memory Considerations

- Python engine: 4-22GB depending on model size
- Swift app: relatively lightweight
- No memory budget check before loading models
- `strictSingleModel` policy prevents multiple models (default)
- `manualMultiModel` allows multiple models (risk of OOM)

### MainActor Contention

Sources of MainActor work during streaming:
1. `InferenceProgressManager.updateStatsAsync()` → `Task { @MainActor in ... }` per SSE chunk with usage
2. `StreamingDeltaProcessor` — entirely `@MainActor`, flush + sync at controlled cadence
3. `ChatTurn.appendContent/appendThinking` — mutates `@Published` properties
4. `NSTableView` reconfiguration — triggered by `notifyContentChanged()`
5. `ChatSession.rebuildVisibleBlocks()` — called by onSync callback

All are throttled to manageable cadences, but cumulative load during very fast generation (>100 tok/s) could still cause frame drops.

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| PF1 | **Double buffering** — Engine batches at stream-interval=3, then VMLXService batches at 30ms. At 27 events/sec, the 30ms window captures ~1 event per flush. The Swift batching may be redundant but harmless. |
| PF2 | **MainActor queue depth** — Each SSE chunk with `usage` creates a `Task { @MainActor }` for stats update. At 27 chunks/sec, that's 27 tasks/sec. Combined with StreamingDeltaProcessor flushes. Need to profile queue depth. |
| PF3 | **NSTableView performance** — `reconfigureRows` at 4-10 times/sec during streaming. Each reconfigure measures cell heights. For conversations with 100+ messages, this could be slow. |

### Latency

| # | Issue | Notes |
|---|-------|-------|
| PF4 | **TTFT breakdown** — Need to measure each phase: `resolveModel()`, `ensureEngineRunning()` (gateway check), URLSession connection, first SSE line, first content delta. Identify bottleneck. |
| PF5 | **Cold start latency** — 0-120s for engine launch. No pre-loading option. First message after model switch always pays this cost. |
| PF6 | **`ServerConfigurationStore.load()` on every request** — VMLXService line 142: `await MainActor.run { ServerConfigurationStore.load() ?? .default }`. This reads from disk on every stream start. Should cache. |

### Memory

| # | Issue | Notes |
|---|-------|-------|
| PF7 | **String concatenation in AccumulatedToolCall** — `arguments += delta.arguments` for every chunk. String concatenation in Swift is O(n) worst case. For very large tool args, this is quadratic. |
| PF8 | **ContentBatch/ReasoningBatch strings** — Grow via `+=` in the 30ms window. Usually small, but no cap. |
| PF9 | **`visibleBlocks` array rebuild** — `rebuildVisibleBlocks()` creates a new array on every sync. For long conversations, this allocates frequently. |

### Profiling Needs

| # | Test | Tool |
|---|------|------|
| PF10 | **MainActor utilization** — Profile with Instruments during 80+ tok/s streaming. Measure time spent in MainActor tasks vs idle. | Instruments (Time Profiler) |
| PF11 | **Memory graph** — Profile memory during model load + generation. Check for leaks (especially in async stream wrappers). | Instruments (Leaks) |
| PF12 | **Network overhead** — Measure bytes transferred between Swift and Python per second during streaming. | Instruments (Network) |
| PF13 | **Frame drops** — Profile UI frame rate during high-speed streaming. Target: 60fps (16.6ms per frame). | Instruments (Core Animation) |

---

## Optimization Opportunities

| # | Optimization | Impact | Effort |
|---|-------------|--------|--------|
| O1 | **Cache ServerConfiguration** — Read once, cache for session lifetime. Invalidate on settings save. | Low latency reduction | Easy |
| O2 | **Pre-load engine on model select** — Start engine when user selects model, not when first message sent. | Eliminates cold TTFT | Medium |
| O3 | **Ring buffer for tool args** — Use `Data` or `[UInt8]` instead of String concatenation for large tool arguments. | Memory efficiency | Low priority |
| O4 | **Incremental visibleBlocks** — Only update changed blocks instead of full rebuild. | CPU reduction for long chats | Medium |
| O5 | **Coalesce MainActor tasks** — Batch stats updates with content yields to reduce task creation. | Reduce MainActor contention | Medium |
| O6 | **Loading progress from Python stdout** — Parse engine's loading messages for progress bar. | UX improvement | Medium |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| bytes.lines optimization | Done | 3000 → 27 suspensions/sec |
| Content batching | Done | 30ms windows |
| Adaptive flush tuning | Done | 50-150ms based on output length |
| Stream interval default 3 | Done | Reduces SSE events |
| Delta time-check skipping | Done | Every 4th delta |
| ChatEngine per-delta timing removed | Done | No more Date() per delta |
| MainActor profiling | **Not done** | Needs Instruments measurement |
| Memory profiling | **Not done** | Needs Instruments measurement |
| Frame drop analysis | **Not done** | Needs measurement at 80+ tok/s |
| ServerConfiguration caching | **Not done** | Reads from disk each request |
