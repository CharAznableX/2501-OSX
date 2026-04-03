# Process Management Rule Book

Rules governing the Python subprocess lifecycle: spawning, health checks, monitoring, idle sleep, crash restart, shutdown.

---

## Process Spawning

### RULE PM-LAUNCH-1: Duplicate Launch Guard
Concurrent launches for the same model MUST be prevented. `launching` Set tracks in-progress launches. Second caller polls gateway every 2s for up to 120s.

**Verified:** `VMLXProcessManager.swift:59-71`.

### RULE PM-LAUNCH-2: Port Allocation
Ports MUST be allocated via `socket(AF_INET, SOCK_STREAM, 0)` → `bind()` to `INADDR_LOOPBACK` port 0 → `getsockname()` → `close()`. OS assigns a random free port.

**TOCTOU Risk (P3):** Between `close(socket)` and Python binding, another process could grab the port. Mitigated by `killOrphanedEngine()` running after port selection. Practical risk is low (ephemeral port range is 49152-65535, ~16K ports).

**Verified:** `VMLXProcessManager.swift:395-422`.

### RULE PM-LAUNCH-3: Environment Isolation
Subprocess MUST have a clean environment with these exact variables:
- `PYTHONDONTWRITEBYTECODE=1` — no `.pyc` files
- `PYTHONNOUSERSITE=1` — no user site-packages
- `PYTHONPATH=""` — no external module paths
- `PYTHONHOME=<bundled python root>` — points to `python/` directory containing `lib/python3.12/`
- `HOME` — inherited from parent
- `PATH=/usr/bin:/bin` — minimal
- `DYLD_FRAMEWORK_PATH` — inherited (Metal framework)
- `TMPDIR` — inherited
- `METAL_DEVICE_WRAPPER_TYPE` — inherited (GPU access)
- `HF_TOKEN` — inherited if present (for gated models)
- `-s` flag on python3 — suppresses user site-packages

**P14 Note:** `PYTHONHOME` is set to `libDir` = `(bin/).deletingLastPathComponent` = `python/`. This is correct — `PYTHONHOME` should point to the root containing `lib/python3.12/`. Verified for search locations 1-3 where path ends in `.../python/bin/python3`.

**Verified:** `VMLXProcessManager.swift:88-107`.

### RULE PM-LAUNCH-4: CLI Arguments
Arguments are built by `VMLXEngineConfig.buildArgs()` which maps `ServerConfiguration` + per-model `ModelOptions` to CLI flags. The command is: `python3 -s -m vmlx_engine.cli serve <model> --port <port> [28+ flags]`.

**Verified:** `VMLXProcessManager.swift:74,85`.

### RULE PM-LAUNCH-5: Orphan Cleanup
Before spawning on a port, `killOrphanedEngine(port:)` MUST run `lsof -ti :<port>` to find any existing process, SIGTERM it, and wait 1s.

**Verified:** `VMLXProcessManager.swift:80,303-328`.

### RULE PM-LAUNCH-6: Gateway Registration
After health check passes, the engine MUST be registered with `VMLXGateway` as a `VMLXInstance` containing: modelName, modelPath, port, processIdentifier, startedAt.

**Verified:** `VMLXProcessManager.swift:137-144`.

---

## Health Checking

### RULE PM-HEALTH-1: Startup Health Poll
After `process.run()`, MUST poll `GET http://127.0.0.1:<port>/health` every 2s for up to 120s. Returns when HTTP 200 received.

**Verified:** `VMLXProcessManager.swift:274-300`.

### RULE PM-HEALTH-2: Crash Detection During Startup
Each poll iteration MUST check `process.isRunning`. If process has exited, MUST immediately throw `VMLXError.engineCrashed(model:stderr:)` with the last stderr line — NOT wait for the full 120s timeout.

**Verified:** `VMLXProcessManager.swift:280-283,295-298`.

### RULE PM-HEALTH-3: No Periodic Health Monitor (P1 — MISSING)
Between requests, there is NO periodic health check. If the engine crashes between requests, it's only detected on the next request. User sees a stale green dot in the model picker.

**Fix needed:** Add a 5-10s periodic health poll Task per running engine. On failure, unregister from gateway and update UI.

### RULE PM-HEALTH-4: Engine Health Endpoint
Python engine's `GET /health` (`server.py:1285`) returns 200 with JSON containing: `model_loaded`, `standby_state`, MCP info. Health returns 200 even during soft/deep sleep (process is alive).

**Compatibility note:** Swift only checks `statusCode == 200`, doesn't read the JSON body. Could use `model_loaded` and `standby_state` for better UI status.

---

## Engine Compatibility: Sleep/Wake

### RULE PM-SLEEP-1: Soft Sleep
`POST /admin/soft-sleep` clears GPU caches, reduces Metal cache limit. Model stays loaded in memory. Engine enters `_standby_state = "soft_sleep"`.

**Verified:** `server.py:1418-1457`.

### RULE PM-SLEEP-2: Deep Sleep
`POST /admin/deep-sleep` unloads model from VRAM entirely. Process stays alive, port stays allocated. Engine enters `_standby_state = "deep_sleep"`.

**Verified:** `server.py:1460-1516`.

### RULE PM-SLEEP-3: JIT (Just-In-Time) Auto-Wake (P4 — RESOLVED)
When engine is in any sleep state and an inference request arrives, middleware intercepts and calls `admin_wake()` automatically. For deep sleep, waits up to **300 seconds** for model reload (large JANG models can take 30-60s to mmap load).

**P4 answer:** YES, the engine auto-wakes on next request. Swift does NOT need to send an explicit wake. `ensureEngineRunning()` finding a gateway port for a sleeping engine is correct — the engine handles wake transparently.

**Verified:** `server.py:512-549` — middleware checks `_standby_state`, acquires `_wake_lock`, calls `admin_wake()`.

### RULE PM-SLEEP-4: Wake Lock
Wake uses an `asyncio.Lock()` to prevent multiple concurrent requests from triggering parallel wakes. Second request waits for first to complete, then re-checks `_standby_state`.

**Verified:** `server.py:514-520`.

### RULE PM-SLEEP-5: Swift Idle Timer Keys
Timer keys MUST be `model + ".soft"` and `model + ".deep"`. `cancelIdleTimer()` cancels bare `model` key (no-op — P13 dead code) AND both `.soft` and `.deep` keys.

**P13 note:** Line 262 `idleTimers[model]?.cancel()` is dead code since no timer is ever stored under bare `model` key. Harmless but should be cleaned up.

**Verified:** `VMLXProcessManager.swift:239,251,261-267`.

---

## Crash Restart

### RULE PM-RESTART-1: Monitor Task
Each engine gets a monitor `Task` that calls `process.waitUntilExit()`. On non-zero exit status, auto-restart is triggered.

**Verified:** `VMLXProcessManager.swift:332-381`.

### RULE PM-RESTART-2: Exponential Backoff
Restart delay is `min(count * 2.0, 10.0)` seconds: 2s, 4s, 6s, 8s, 10s. Maximum 3 restarts before giving up.

**Verified:** `VMLXProcessManager.swift:364`.

### RULE PM-RESTART-3: Restart Count Reset
On successful restart: count resets to 0. On manual `stopEngine()`: count is removed from dict.

**Verified:** `VMLXProcessManager.swift:375,180`.

### RULE PM-RESTART-4: Missing ModelOptions on Restart (P6 — BUG)
`launchEngine()` in the restart path (line 369) does NOT pass `modelOptions`. The relaunched engine won't have per-model parser settings (tool parser, reasoning parser). Engine falls back to `auto` for both, which may or may not match what the user configured.

**Impact:** Medium — `auto` detection usually works, but user's explicit parser choice is lost.

**Fix:** Store `modelOptions` when starting the monitor. Pass it on restart.

### RULE PM-RESTART-5: Cleanup Before Restart
On crash, monitor unregisters from gateway and cleans up process/timers before attempting restart.

**Verified:** `VMLXProcessManager.swift:354-355`.

---

## Shutdown

### RULE PM-STOP-1: SIGTERM → SIGKILL Escalation
`stopEngine()` sends SIGTERM first, waits 1.5s, then SIGKILL if still running.

**Verified:** `VMLXProcessManager.swift:174-178`.

### RULE PM-STOP-2: Fuzzy Key Matching
`stopEngine(model:)` first tries exact match on `processes` dict, then matches by last path component (lowercased). This handles calls with either model name or full path.

**Verified:** `VMLXProcessManager.swift:158-165`.

### RULE PM-STOP-3: Cleanup on Stop
`stopEngine()` cancels idle timers, cancels monitor task, removes process from dict, removes restart count, unregisters from gateway.

**Verified:** `VMLXProcessManager.swift:168-184`.

### RULE PM-STOP-4: App Quit (P2 — RACE CONDITION)
`applicationShouldTerminate()` returns `.terminateLater`, runs `stopAll()` in async Task, then calls `NSApp.reply(toApplicationShouldTerminate: true)`. This is the correct pattern — but if the app is force-quit (Cmd+Q during hang, SIGKILL), processes are orphaned.

**Mitigation:** `killOrphanedEngine(port:)` cleans up orphans on next launch. But orphans consume memory until then.

**P17 note:** Global orphan scan on launch would catch all stale `python3 -m vmlx_engine.cli` processes, not just the specific port.

---

## Stderr Capture

### RULE PM-STDERR-1: Last Line Tracking
`LastLine` actor stores the most recent stderr line per model. Updated by `Task.detached` reading `stderrPipe.fileHandleForReading.bytes.lines`.

**P8 note:** Only captures single line. OOM errors often span multiple lines. Should capture last 5-10 lines for better diagnostics.

**Verified:** `VMLXProcessManager.swift:19-23,122-127`.

### RULE PM-STDERR-2: Pipe Tasks Not Cancelled (P9)
Stdout/stderr pipe reader Tasks are `Task.detached` and never cancelled. They end naturally when the pipe closes on process exit. If a process hangs indefinitely, these tasks leak.

**Risk:** Low — process hangs are rare, and tasks are lightweight.

---

## Python Binary Resolution

### RULE PM-PYTHON-1: Search Order
1. App bundle: `Bundle.main.resourcePath/bundled-python/python/bin/python3`
2. Dev mode project: `#filePath → ../../Resources/bundled-python/python/bin/python3`
3. Dev mode vmlx: `~/mlx/vllm-mlx/panel/bundled-python/python/bin/python3`
4. System fallback: `/usr/bin/python3` (likely won't have vmlx_engine)

**Verified:** `VMLXProcessManager.swift:428-456`.

---

## Known Issues

| ID | Rule Violated | Severity | Status |
|----|---------------|----------|--------|
| P1 | PM-HEALTH-3 | **HIGH** | No periodic health monitor — stale UI for crashed engines |
| P2 | PM-STOP-4 | Medium | App quit race — async stopAll may not complete before force-quit |
| P3 | PM-LAUNCH-2 | Low | Port TOCTOU race — mitigated by orphan cleanup |
| P4 | PM-SLEEP-3 | **RESOLVED** | Engine auto-wakes on request via JIT middleware |
| P5 | PM-STOP-2 | Low | Monitor key mismatch — fuzzy matching handles both name and path |
| P6 | PM-RESTART-4 | Medium | Crash restart doesn't pass modelOptions — parsers fall back to auto |
| P8 | PM-STDERR-1 | Low | Only captures single stderr line — OOM spans multiple lines |
| P9 | PM-STDERR-2 | Low | Pipe reader tasks never cancelled — end naturally on pipe close |
| P10 | PM-RESTART-4 | Medium | `swapModel()` doesn't pass modelOptions either |
| P13 | PM-SLEEP-5 | None | Dead code — bare model key cancel is no-op |
| P14 | PM-LAUNCH-3 | **VERIFIED OK** | PYTHONHOME correctly points to python/ root |

---

## Invariants

1. **One process per model** — `launching` Set + gateway check prevents duplicates.
2. **Gateway always in sync** — Every `process.run()` success → gateway register. Every stop/crash → gateway unregister.
3. **Cleanup always runs** — stopEngine cancels timers + monitors. Crash monitor cleans up + optional restart.
4. **Port always freed** — Process exit releases port. Orphan cleanup catches stale binds.
5. **SIGTERM before SIGKILL** — Always give engine 1.5s grace period.
6. **Engine auto-wakes** — Sleep states are transparent to Swift; JIT middleware handles wake.
