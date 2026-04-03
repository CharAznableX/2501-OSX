# Process Management Audit

## How It Works

Each local model runs as a separate Python subprocess managed by `VMLXProcessManager` (actor singleton).

```
VMLXProcessManager (actor)
  ├── processes: [String: Process]           — running subprocesses keyed by model name
  ├── launching: Set<String>                 — prevents duplicate concurrent launches
  ├── idleTimers: [String: Task<Void,Never>] — soft/deep sleep timers per model
  ├── lastStderrLines: [String: LastLine]    — crash diagnostics per model
  ├── restartCounts: [String: Int]           — backoff tracking
  └── monitors: [String: Task<Void,Never>]   — process.waitUntilExit() watchers
```

### Launch Sequence

**File:** `Services/Inference/VMLXProcessManager.swift` (lines 46-151)

**Function:** `launchEngine(model:modelPath:config:modelOptions:) async throws -> Int`

1. Check `VMLXGateway.shared.port(for: model)` — return early if already running
2. Check `launching.contains(model)` — if another launch in progress, poll gateway every 2s for 120s
3. `launching.insert(model)` (guard against duplicates)
4. `findFreePort()` — bind socket to port 0, read OS-assigned port, close socket
5. `VMLXEngineConfig.buildArgs()` — build CLI args from config + per-model options
6. `bundledPythonPath()` — resolve Python binary (4 search locations)
7. `killOrphanedEngine(port:)` — `lsof -ti :<port>`, SIGTERM any occupant, wait 1s
8. Create `Process` with clean environment, attach stdout/stderr pipes
9. `process.run()` — spawn the subprocess
10. `waitForHealth(port:timeout:process:model:)` — poll `/health` every 2s, 120s timeout
11. Register `VMLXInstance` with `VMLXGateway` (dual-key: name + path)
12. `startMonitor()` — watch for unexpected exit

### Environment Isolation

**Lines 88-107:**

```swift
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["PYTHONNOUSERSITE"] = "1"
env["PYTHONPATH"] = ""
env["PYTHONHOME"] = libDir  // bundled python root
env["HOME"] = ProcessInfo.processInfo.environment["HOME"]
env["PATH"] = "/usr/bin:/bin"
env["DYLD_FRAMEWORK_PATH"] = ...
env["TMPDIR"] = ...
env["METAL_DEVICE_WRAPPER_TYPE"] = ...
env["HF_TOKEN"] = ... // optional, for gated models
```

- `-s` flag suppresses user site-packages
- `PYTHONPATH=""` prevents external module paths
- Only minimal system env inherited (dyld, Metal, tmpdir)

### Health Checking

**Function:** `waitForHealth(port:timeout:process:model:)` (lines 274-300)

- Polls `GET http://127.0.0.1:<port>/health` every 2s
- Checks `process.isRunning` each iteration — surfaces crash immediately as `VMLXError.engineCrashed(model:stderr:)` with last stderr line
- 120s total timeout, then `VMLXError.engineStartTimeout`
- No periodic health monitor between requests (crash detected only on next request)

### Crash Restart

**Function:** `startMonitor()` (lines 332-382)

- Creates a `Task` that calls `process.waitUntilExit()`
- On non-zero exit: increments restart count, exponential backoff (2s, 4s, 6s)
- Max 3 restarts (`maxRestarts`), then gives up with OOM warning log
- On successful restart: resets count to 0
- Unregisters from gateway, cleans up process/timers

### Idle Sleep

**Function:** `resetIdleTimer(for:config:)` (lines 233-259)

Two independent timers per model:
- **Soft sleep** (`enableSoftSleep`, `softSleepMinutes`): `POST /admin/soft-sleep` — clears GPU caches, model stays loaded
- **Deep sleep** (`enableDeepSleep`, `deepSleepMinutes`): `POST /admin/deep-sleep` — unloads model from VRAM

Timer keys: `model + ".soft"` and `model + ".deep"` in `idleTimers` dict.

**Timer reset:** Called after each completed request in `VMLXService.streamWithTools()` (line 263). Not called before stream — prevents sleep during active generation.

### Shutdown

**Function:** `stopEngine(model:)` (lines 156-186)

1. Find process key (exact match, then last-path-component match)
2. Cancel idle timer and monitor task
3. `process.terminate()` (SIGTERM)
4. Wait 1.5s
5. If still running: `kill(pid, SIGKILL)`
6. Unregister from gateway

**`stopAll()`** (lines 189-194): Iterates all process keys, calls `stopEngine()` for each.

### Port Allocation

**Function:** `findFreePort()` (lines 395-422)

1. `socket(AF_INET, SOCK_STREAM, 0)`
2. `bind()` to `INADDR_LOOPBACK` port 0
3. `getsockname()` to read assigned port
4. `close(socket)` — releases the port

### Orphan Detection

**Function:** `killOrphanedEngine(port:)` (lines 303-328)

- Runs `/usr/sbin/lsof -ti :<port>` to find PIDs
- SIGTERM each PID
- Waits 1s for cleanup

### Python Binary Resolution

**Function:** `bundledPythonPath()` (lines 428-456)

Search order:
1. App bundle: `Resources/bundled-python/python/bin/python3`
2. Dev mode project: `Resources/bundled-python/python/bin/python3` (relative to #filePath)
3. Dev mode vmlx: `~/mlx/vllm-mlx/panel/bundled-python/python/bin/python3`
4. System fallback: `/usr/bin/python3`

---

## What Needs Checking

### Critical

| # | Issue | File | Line |
|---|-------|------|------|
| P1 | **No periodic health monitor** — Between requests, if the engine crashes, it's only detected on the next request. User sees a stale green dot in the model picker. Need a periodic (5-10s) health poll like the Electron app does. | VMLXProcessManager.swift | — |
| P2 | **App quit cleanup race** — `applicationWillTerminate` calls `VMLXProcessManager.shared.stopAll()` but this is `async` on an actor. The synchronous `applicationWillTerminate` may return before processes are killed, leaving orphaned Python processes. | AppDelegate.swift | — |
| P3 | **Port TOCTOU race** — Between `findFreePort()` closing the socket and Python binding, another process could grab the port. `killOrphanedEngine()` runs before spawn but after port selection. | VMLXProcessManager.swift | 73-80 |
| P4 | **Idle sleep → wake detection** — After deep sleep, engine is still "running" (process alive, gateway has port). Next `ensureEngineRunning()` finds the port and returns it. But the engine needs to reload the model. Does the engine auto-wake on next request? | VMLXService.swift | 429-434 |
| P5 | **Monitor task leak on manual unload** — `stopEngine()` cancels monitors (line 169). But if `stopEngine()` is called with a model path while monitors are keyed by model name, the monitor might not be found/cancelled. | VMLXProcessManager.swift | 158-170 |

### Process Lifecycle

| # | Issue | File | Line |
|---|-------|------|------|
| P6 | **Crash restart passes no modelOptions** — `launchEngine()` in the restart path (line 369) doesn't pass `modelOptions`. The relaunched engine won't have per-model parser settings. | VMLXProcessManager.swift | 369-373 |
| P7 | **Restart count never resets on normal stop** — `restartCounts` is only reset on successful restart (line 375) or `stopEngine()` (line 180). If a model crashes 3 times and the user manually unloads+reloads, the count is reset. But if they just send another message, `launchEngine()` starts fresh (count not checked there). | VMLXProcessManager.swift | 358-379 |
| P8 | **LastLine captures only single line** — `LastLine` actor stores one string. OOM errors often span multiple lines. Should capture last N lines (e.g., 5) for better diagnostics. | VMLXProcessManager.swift | 19-23 |
| P9 | **stderr/stdout pipe Tasks never cancelled** — `Task.detached` for stdout/stderr reading (lines 117-127) are fire-and-forget. When the process exits, the pipe closes and the loop ends naturally. But if the process hangs, these tasks live forever. | VMLXProcessManager.swift | 117-127 |
| P10 | **`swapModel()` doesn't pass modelOptions** — `swapModel(from:to:modelPath:config:)` calls `launchEngine()` without `modelOptions`. | VMLXProcessManager.swift | 199-207 |

### Eviction

| # | Issue | File | Line |
|---|-------|------|------|
| P11 | **strictSingleModel race** — `stopAll()` is called before `launchEngine()`. But `stopAll()` iterates all keys and calls `stopEngine()` sequentially. If stop is slow (1.5s SIGTERM + 1.5s SIGKILL per model), the new engine launch is delayed. | VMLXService.swift | 449-451 |
| P12 | **manualMultiModel memory pressure** — No memory budget check before launching a new model. Two large models (e.g., 2x Gemma 4 27B) could OOM the system. | VMLXProcessManager.swift | — |
| P13 | **Idle timer dead code** — `cancelIdleTimer(for: model)` cancels bare `model` key, but `resetIdleTimer` only sets `model + ".soft"` and `model + ".deep"`. The bare key cancel is a no-op. Harmless but confusing. | VMLXProcessManager.swift | 262-263 |
| P14 | **PYTHONHOME correctness** — `bundleDir` = `bin/`, `libDir` = `python/`. `PYTHONHOME` should point to the root containing `lib/python3.12/`. Verify the path math is correct for all 4 search locations. | VMLXProcessManager.swift | 93-96 |

---

## Functions Reference

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `launchEngine()` | VMLXProcessManager.swift | 46 | Spawn Python subprocess |
| `stopEngine()` | VMLXProcessManager.swift | 156 | SIGTERM → SIGKILL cleanup |
| `stopAll()` | VMLXProcessManager.swift | 189 | Stop all engines |
| `swapModel()` | VMLXProcessManager.swift | 199 | Stop old, launch new |
| `softSleep()` | VMLXProcessManager.swift | 212 | POST /admin/soft-sleep |
| `deepSleep()` | VMLXProcessManager.swift | 222 | POST /admin/deep-sleep |
| `resetIdleTimer()` | VMLXProcessManager.swift | 233 | Arm soft/deep sleep timers |
| `cancelIdleTimer()` | VMLXProcessManager.swift | 261 | Cancel all timers for model |
| `waitForHealth()` | VMLXProcessManager.swift | 274 | Poll /health with crash detection |
| `killOrphanedEngine()` | VMLXProcessManager.swift | 303 | lsof + SIGTERM orphans |
| `startMonitor()` | VMLXProcessManager.swift | 332 | Watch for unexpected exit |
| `findFreePort()` | VMLXProcessManager.swift | 395 | Bind socket port 0 |
| `bundledPythonPath()` | VMLXProcessManager.swift | 428 | Resolve Python binary |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Process spawning | Working | Clean env, bundled Python, all CLI args |
| Health polling | Working | 2s intervals, 120s timeout, crash detection |
| Crash restart | Working | 3 retries, exponential backoff |
| Idle sleep | Working | Separate soft/deep timers |
| Shutdown | Working | SIGTERM → SIGKILL escalation |
| Orphan detection | Working | lsof-based, runs before each launch |
| Duplicate launch guard | Working | `launching` Set |
| Periodic health monitor | **Missing** | No between-request health checks |
| Memory budget check | **Missing** | No pre-launch RAM verification |
| Loading progress UI | **Missing** | No visual feedback during 0-120s launch |
