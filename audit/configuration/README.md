# Configuration Audit

## How It Works

Configuration is stored in two structs, persisted via JSON stores, and mapped to engine CLI args.

### ServerConfiguration

**File:** `Models/Configuration/ServerConfiguration.swift` (lines 26-341)

28+ fields covering:
- **Server:** port, exposeToNetwork, startAtLogin, hideDockIcon, appearanceMode, numberOfThreads, backlog
- **Generation:** genTopP, maxTokens, allowedOrigins, modelEvictionPolicy
- **Engine:** continuousBatching, maxNumSeqs, streamInterval
- **Prefix cache:** enablePrefixCache, prefixCacheSize, cacheMemoryPercent, cacheMemoryMB, cacheTTLMinutes
- **Paged cache:** usePagedCache, pagedCacheBlockSize, maxCacheBlocks
- **Disk cache:** enableDiskCache, diskCacheMaxGB, enableBlockDiskCache, blockDiskCacheMaxGB
- **KV quantization:** kvCacheQuantization, kvCacheGroupSize
- **Parsers:** toolCallParser, reasoningParser
- **Performance:** enableJIT
- **Idle sleep:** idleSleepMode, idleSleepMinutes, enableSoftSleep, softSleepMinutes, enableDeepSleep, deepSleepMinutes
- **Thinking:** defaultEnableThinking
- **Generation defaults:** defaultTemperature, defaultTopP
- **Speculative:** speculativeModel, numDraftTokens, enablePLD

**Default values** (lines 201-247):
- port: 1337
- continuousBatching: true
- streamInterval: 3
- enablePrefixCache: true, prefixCacheSize: 100
- cacheMemoryPercent: 0.30
- usePagedCache: true, pagedCacheBlockSize: 64, maxCacheBlocks: 1000
- enableDiskCache: true, diskCacheMaxGB: 10.0
- enableBlockDiskCache: true, blockDiskCacheMaxGB: 10.0
- kvCacheQuantization: "none"
- toolCallParser: "auto", reasoningParser: "auto"
- enableJIT: true
- enableSoftSleep: true, softSleepMinutes: 10
- enableDeepSleep: true, deepSleepMinutes: 30
- maxTokens: 32768
- modelEvictionPolicy: .strictSingleModel

### Persistence

**File:** `Models/Configuration/ServerConfigurationStore.swift`

- `save(configuration:)` encodes to JSON, writes to `server.json` in app support
- `load()` → `ServerConfiguration?` — returns nil if file missing (callers use `?? .default`)
- Backward-compatible decoder: every field uses `decodeIfPresent` with fallback to default

### ChatConfiguration

**File:** `Models/Configuration/ChatConfigurationStore.swift`

- Separate store for chat-specific settings (system prompt, chat behavior)
- `systemPrompt` used by `ChatEngine.enrichMessagesWithSystemPrompt()`

### Settings UI

**File:** `Views/Settings/ConfigurationView.swift`

All settings loaded into `@State private var temp*` variables on view appear. Changes are local copies until "Save" is pressed.

Save logic:
1. Builds `ServerConfiguration` from all temp vars
2. Compares with `previousServerCfg` to detect restart-requiring changes
3. Calls `ServerConfigurationStore.save(configuration)`
4. If `serverRestartNeeded`: calls `AppDelegate.shared?.serverController.restartServer()`

**Stats toggle** special case: writes directly to UserDefaults and updates `InferenceProgressManager.shared.showStats` immediately (line 730), no save needed.

### CLI Arg Mapping

**File:** `Services/Inference/VMLXEngineConfig.swift` (lines 22-139)

Static `buildArgs(model:port:config:modelOptions:)`:
- Per-model options (`toolParser`, `reasoningParser`) override global config
- All cache args nested inside `if config.continuousBatching`
- `enable_thinking` intentionally never sent
- Default generation params: `--default-temperature`, `--default-top-p`

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| CF1 | **`maxTokens` not saved** — Known issue from CLAUDE.md: "maxTokens not saved from ConfigurationView (always uses default 32768)". Verify if this is still the case. The default init has `maxTokens: 32768`. |
| CF2 | **Settings that require restart** — `serverRestartNeeded` detection: which fields trigger it? It should cover: parsers, batching, cache settings, sleep mode, JIT, speculative decoding. Verify completeness. |
| CF3 | **Settings changes don't affect running engines** — Only the SwiftNIO server restarts. Running Python engines keep their old CLI args. New settings only apply to newly launched engines. This is by design but could confuse users. |
| CF4 | **Orphaned @State vars** — CLAUDE.md mentions "ConfigurationView has orphaned @State vars for removed parser pickers (dead code, harmless)". Should be cleaned up. |

### Validation

| # | Issue | Notes |
|---|-------|-------|
| CF5 | **Port range** — `isValidPort` checks `(1..<65536)`. But ports < 1024 require root. Should warn user. |
| CF6 | **cacheMemoryPercent range** — Default 0.30. Slider should enforce 0.0-1.0. What happens if user enters 30? |
| CF7 | **maxNumSeqs: 256** — Very high default. Most desktop use cases need 1-4. Could waste memory pre-allocating for 256 concurrent sequences. |
| CF8 | **streamInterval: 3** — Default is every 3rd token. User can set to 1 (every token) which increases SSE overhead. No warning about performance impact. |
| CF9 | **diskCacheMaxGB: 10.0 each** — Two disk cache types, each 10GB. Total 20GB potential. No disk space check. |

### Persistence

| # | Issue | Notes |
|---|-------|-------|
| CF10 | **Backward compatibility** — All fields use `decodeIfPresent` with defaults. If a new field is added but old JSON doesn't have it, it gets the default. This is correct. But verify no field was accidentally made non-optional. |
| CF11 | **`cacheMemoryMB` is truly optional** — `decodeIfPresent(Int.self, forKey: .cacheMemoryMB)` returns nil when missing. This triggers percent-based caching. Verify the UI doesn't accidentally set it to 0. |
| CF12 | **ModelEvictionPolicy default** — `.strictSingleModel` is correct for most users. But the UI should explain what "Flexible (Multi Model)" means clearly. |

### CLI Mapping

| # | Issue | Notes |
|---|-------|-------|
| CF13 | **genTopP vs defaultTopP** — Two paths: `defaultTopP` takes priority, falls back to `genTopP` if non-default (line 134). This dual-path could cause confusion. |
| CF14 | **`enablePLD` flag** — Prompt Lookup Decoding appends `--enable-pld`. Verify the engine actually supports this flag. |
| CF15 | **`defaultEnableThinking`** — Maps to `--default-enable-thinking`. But VMLXService never sends `enable_thinking` per-request. Does the engine-level default interact correctly with per-request behavior? |
| CF16 | **No `--allowed-origins` flag** — `allowedOrigins` is in ServerConfiguration but not mapped in VMLXEngineConfig. The engine's CORS is separate from the SwiftNIO gateway's CORS. Is this intentional? |

---

## Functions Reference

| Function | File | Purpose |
|----------|------|---------|
| `ServerConfiguration.default` | ServerConfiguration.swift:201 | Default values for all 28+ fields |
| `ServerConfiguration.init(from:)` | ServerConfiguration.swift:151 | Backward-compatible decoder |
| `ServerConfigurationStore.save()` | ServerConfigurationStore.swift | JSON persistence |
| `ServerConfigurationStore.load()` | ServerConfigurationStore.swift | Load or nil |
| `VMLXEngineConfig.buildArgs()` | VMLXEngineConfig.swift:22 | Config → CLI args |
| ConfigurationView save logic | ConfigurationView.swift:~1070 | Compare + save + restart |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| 28+ engine settings | Working | All defaults, persistence, CLI mapping |
| Backward-compatible decoder | Working | decodeIfPresent for all fields |
| Settings UI | Working | All panels, save button |
| Engine restart detection | Working | Compares previous vs new config |
| Per-model parser override | Working | ModelOptions takes priority |
| maxTokens persistence | **Bug** | Not saved from ConfigurationView |
| Orphaned @State vars | **Tech debt** | Dead code from removed pickers |
| Validation warnings | **Missing** | No port/memory/disk warnings |
