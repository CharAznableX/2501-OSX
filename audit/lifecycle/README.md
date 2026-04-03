# App Lifecycle Audit

## How It Works

### Startup

**File:** `AppDelegate.swift`

`applicationDidFinishLaunching`:
1. `LaunchGuard.checkOnLaunch()` — crash loop detection
2. Set dock policy (regular or accessory based on `hideDockIcon`)
3. Configure notifications
4. Set up distributed control notifications
5. Create menu bar status item with activity dot
6. Start `ServerController` (SwiftNIO HTTP server on configured port)

### Shutdown

**File:** `AppDelegate.swift`

`applicationWillTerminate`:
1. `VMLXProcessManager.shared.stopAll()` — SIGTERM → SIGKILL all Python subprocesses
2. `MCPServerManager.stopAll()` — clean up MCP stdio transport

### Server Lifecycle

**File:** `Networking/ServerController.swift`

- `@MainActor ObservableObject` owned by `AppDelegate`
- `startServer()` — creates `OsaurusServer` on configured host:port
- `restartServer()` — stop + start (triggered by settings save when restart needed)
- `signalGenerationStart() / signalGenerationEnd()` — tracks `activeRequestCount` for menu bar activity dot

### Session Management

**File:** `Managers/Chat/ChatSessionsManager.swift`

- `@MainActor ObservableObject` singleton
- `[ChatSessionData]` sorted by `updatedAt`
- Operations: `createNew`, `save`, `delete`, `rename`, `updateSession`
- Persistence via `ChatSessionStore` (SQLite)

**File:** `Models/Chat/ChatSessionData.swift`

- `Codable` value type: id (UUID), title, createdAt, updatedAt, selectedModel, turns, agentId
- Custom decoder handles backward compat (`personaId` → `agentId`)

### ChatSession Lifecycle

**File:** `Views/Chat/ChatView.swift`

`ChatSession` (`@MainActor ObservableObject`) owns:
- `turns: [ChatTurn]` — conversation
- `selectedModel: String?` — current model
- `activeModelOptions: [String: ModelOptionValue]` — per-model options
- `visibleBlocks: [ContentBlock]` — flattened blocks for NSTableView
- `isStreaming: Bool` — locks UI during generation

Key lifecycle methods:
- `reset()` — new chat init, loads model options for selected model
- `load(from: ChatSessionData)` — session restore, loads model options
- `send(text:)` — append user turn, start generation
- `finalizeRun()` → `completeRunCleanup()` — save, rebuild blocks, reset streaming

### Engine Process Lifecycle

See `process-management/README.md` for full details. Summary:
- Spawned on first request to a model
- Health-polled for up to 120s
- Monitored for unexpected exit (auto-restart with backoff)
- Idle sleep timers (soft/deep) reset after each request
- SIGTERM → SIGKILL on app quit

### Menu Bar Activity

- Green blinking dot during active generation
- Blue/purple dot for VAD (voice activity detection) active
- `activeRequestCount` tracked by `ServerController`

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| L1 | **`stopAll()` async in sync `applicationWillTerminate`** — `VMLXProcessManager.shared.stopAll()` is async (actor method). The synchronous `applicationWillTerminate` callback may return before processes are killed. Orphaned Python processes will persist until system reboot or manual `kill`. |
| L2 | **Crash loop detection** — `LaunchGuard.checkOnLaunch()` detects crash loops. What threshold? What does it do (disable features, show alert, skip startup)? |
| L3 | **Session auto-save** — Is the current session auto-saved periodically? Or only on explicit save/new-chat/close? Crash during generation could lose the conversation. |

### Startup

| # | Issue | Notes |
|---|-------|-------|
| L4 | **Engine not started on launch** — Engines are lazily launched on first request. User opens app, selects model, sees gray dot. First message triggers 0-120s engine startup. No option for pre-loading. |
| L5 | **Server port conflict** — If port 1337 is already in use (another Osaurus instance, other app), what happens? Error message? Fallback port? |
| L6 | **Model list refresh on startup** — When does `ModelManager.scanLocalModels()` run? On launch? On picker open? Does it detect new models added while app was closed? |

### Shutdown

| # | Issue | Notes |
|---|-------|-------|
| L7 | **Graceful stream termination** — If user quits app during active generation, is the in-flight stream properly cancelled before `stopAll()`? Or does it race? |
| L8 | **Session save on quit** — Is the current unsaved session saved on `applicationWillTerminate`? If the shutdown is forced (SIGKILL), data is lost. |
| L9 | **Disk cache flush** — Python engine's disk cache may have dirty buffers. SIGTERM should trigger cleanup, but if escalated to SIGKILL, cache could be corrupted. |

### Sleep/Wake

| # | Issue | Notes |
|---|-------|-------|
| L10 | **macOS system sleep** — When Mac goes to sleep, do engine processes survive? Do they resume correctly on wake? Does the health check fail during sleep? |
| L11 | **Soft sleep → request** — After soft sleep, engine should auto-wake on next request. But there's no explicit wake call. The engine's HTTP handler should handle this. Verify. |
| L12 | **Deep sleep → request** — After deep sleep, model is unloaded from VRAM. Next request triggers model reload. Does this happen transparently? How long does it take? |

### Memory Pressure

| # | Issue | Notes |
|---|-------|-------|
| L13 | **macOS memory pressure** — When system RAM is low, macOS may kill the Python engine process. The monitor detects this and tries to restart. But restart will also fail if RAM is still low. |
| L14 | **Swap thrashing** — Large models can cause swap. No detection or warning. |
| L15 | **No sleep/wake observer** — No `NSWorkspace.didWakeNotification` handler. Mac sleeps → Python processes suspended → OS may SIGKILL them → wake finds stale gateway entries → next request gets connection refused. Fix: add observer, check health of all registered engines on wake. |
| L16 | **No memory pressure observer** — No `DispatchSource.makeMemoryPressureSource()`. Under pressure, kernel SIGKILLs Python → monitor auto-restarts → also OOMs → infinite restart loop (capped at 3 but still wasteful). Fix: on warning level, soft-sleep engine; on critical, deep-sleep. |
| L17 | **Global orphan scan on launch** — Current orphan detection only checks specific port via `lsof`. Should also scan for any `python3 -m vmlx_engine.cli` processes on startup and kill them (from prior app crash). |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Startup sequence | Working | LaunchGuard, server, menu bar |
| Shutdown cleanup | **Race condition** | async stopAll() in sync callback |
| Session persistence | Working | SQLite, auto-save on chat end |
| Engine lazy launch | Working | Spawned on first request |
| Menu bar activity | Working | Green dot during generation |
| Crash loop detection | Working | LaunchGuard |
| Pre-loading engines | **Missing** | No eager start option |
| Memory pressure handling | **Missing** | No detection or warning |
| System sleep handling | **Needs verify** | Untested behavior |
