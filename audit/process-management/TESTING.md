# Process Management Testing Suite

Test cases for all process management audit items (P1-P14) and engine compatibility.

---

## Unit Tests: Port Allocation

### T-PORT-1: findFreePort returns valid port
```
Call findFreePort()
Expect: Port in range 1024-65535
Expect: Port NOT in use (bind test succeeds)
```

### T-PORT-2: findFreePort uniqueness
```
Call findFreePort() 100 times
Expect: All ports unique (OS assigns different ephemeral ports)
```

### T-PORT-3: Port TOCTOU race (P3)
```
Setup: findFreePort() returns port X, immediately bind another socket to port X
Expect: Python engine fails to bind, health check fails, VMLXError thrown
Mitigation: killOrphanedEngine runs lsof and kills occupant
```

---

## Unit Tests: Python Path Resolution

### T-PATH-1: App bundle path
```
Setup: Bundle.main.resourcePath has bundled-python/python/bin/python3
Expect: Returns app bundle path (location 1)
```

### T-PATH-2: Dev mode project path
```
Setup: No app bundle, but Resources/bundled-python/ exists relative to #filePath
Expect: Returns dev project path (location 2)
```

### T-PATH-3: Dev mode vmlx path
```
Setup: No app bundle, no project Resources, but ~/mlx/vllm-mlx/panel/bundled-python/ exists
Expect: Returns vmlx bundled path (location 3)
```

### T-PATH-4: System fallback
```
Setup: None of the bundled paths exist
Expect: Returns /usr/bin/python3 (location 4)
```

### T-PATH-5: PYTHONHOME correctness (P14)
```
For each search location:
  Given python path = .../python/bin/python3
  PYTHONHOME = pythonPath.deletingLastPathComponent.deletingLastPathComponent = .../python/
  Expect: PYTHONHOME/lib/python3.12/ exists (contains stdlib)
```

---

## Integration Tests: Launch Sequence

### T-LAUNCH-1: Normal launch
```
Setup: Model path exists, no engine running
Call: launchEngine(model: "test", modelPath: "/path/to/model", config: .default)
Expect: Process spawned, health check passes, gateway registered, port returned
Verify: Gateway.port(for: "test") returns the port
```

### T-LAUNCH-2: Already running — returns existing port
```
Setup: Engine running on port 54321
Call: launchEngine(model: "test", ...)
Expect: Returns 54321 immediately (no new process spawned)
```

### T-LAUNCH-3: Duplicate concurrent launch guard
```
Setup: Two concurrent launchEngine() calls for same model
Expect: First call spawns. Second call enters polling loop, eventually returns same port.
Verify: Only one process spawned
```

### T-LAUNCH-4: Engine crash during startup
```
Setup: Model path is invalid (engine exits immediately with non-zero status)
Call: launchEngine(model: "bad", modelPath: "/nonexistent", ...)
Expect: VMLXError.engineCrashed with last stderr line
Timing: Should fail quickly (not wait full 120s)
```

### T-LAUNCH-5: Engine startup timeout
```
Setup: Engine starts but /health never returns 200 within 120s
Expect: VMLXError.engineStartTimeout after 120s
```

### T-LAUNCH-6: Environment isolation verified
```
Call: launchEngine, then read /proc/<pid>/environ or check engine logs
Expect: PYTHONPATH is empty, PYTHONNOUSERSITE=1, PYTHONDONTWRITEBYTECODE=1
Expect: No user site-packages loaded
```

### T-LAUNCH-7: Orphan killed before launch
```
Setup: Stale process bound to the port assigned by findFreePort
Call: launchEngine (which calls killOrphanedEngine)
Expect: Stale process SIGTERMed, new engine starts successfully
```

---

## Integration Tests: Shutdown

### T-STOP-1: Normal stop — SIGTERM succeeds
```
Setup: Running engine
Call: stopEngine(model: "test")
Expect: SIGTERM sent, process exits within 1.5s, gateway unregistered, timers cancelled
```

### T-STOP-2: SIGTERM ignored — SIGKILL escalation
```
Setup: Running engine that traps SIGTERM
Call: stopEngine(model: "test")
Expect: SIGTERM sent, 1.5s wait, SIGKILL sent, process killed
```

### T-STOP-3: Fuzzy key matching (P5)
```
Setup: Engine registered under key "gemma-2-9b" 
Call: stopEngine(model: "/path/to/gemma-2-9b")
Expect: Matches by last path component, stops correctly
```

### T-STOP-4: stopAll stops all engines
```
Setup: 3 engines running
Call: stopAll()
Expect: All 3 stopped, all gateway entries removed
```

### T-STOP-5: App quit cleanup (P2)
```
Setup: 2 engines running, trigger applicationShouldTerminate
Expect: Both engines SIGTERMed before app exits
Verify: No orphaned python3 processes after app quit
Edge case: Force-quit during stopAll() — orphans expected, cleaned on next launch
```

---

## Integration Tests: Crash Restart

### T-RESTART-1: Auto-restart on crash
```
Setup: Running engine, kill the python process externally
Expect: Monitor detects exit, waits 2s (first restart), relaunches
Verify: Gateway re-registered with new port, engine responds to /health
```

### T-RESTART-2: Exponential backoff
```
Setup: Engine crashes 3 times in succession
Expect: Delays are 2s, 4s, 6s between restarts
After 3rd crash: gives up, logs OOM warning
```

### T-RESTART-3: Restart count reset on success
```
Setup: Engine crashes once, restarts successfully
Expect: restartCounts[model] == 0 (reset after successful restart)
Next crash: backoff starts at 2s again (not 4s)
```

### T-RESTART-4: Missing modelOptions on restart (P6 — BUG)
```
Setup: Launch engine with modelOptions["reasoningParser"] = "qwen3"
Then: Kill the engine process
Expect (current — BUG): Restarted engine has no --reasoning-parser qwen3 flag
Expect (fixed): Restarted engine passes same modelOptions
```

### T-RESTART-5: swapModel missing modelOptions (P10)
```
Setup: User configured per-model parsers for model B
Call: swapModel(from: "A", to: "B", ...)
Expect (current — BUG): Model B launched without modelOptions
Expect (fixed): swapModel passes modelOptions parameter
```

---

## Integration Tests: Idle Sleep

### T-SLEEP-1: Soft sleep timer fires
```
Setup: Launch engine, set enableSoftSleep=true, softSleepMinutes=1
Wait: 65 seconds (no requests)
Expect: POST /admin/soft-sleep sent to engine
Verify: Engine enters soft_sleep state
```

### T-SLEEP-2: Deep sleep timer fires
```
Setup: Launch engine, set enableDeepSleep=true, deepSleepMinutes=2
Wait: 125 seconds (no requests)
Expect: POST /admin/deep-sleep sent to engine
Verify: Engine enters deep_sleep state
```

### T-SLEEP-3: Timer reset on request
```
Setup: Soft sleep timer at 60s. At t=50s, send a request.
Expect: Timer resets. Soft sleep fires at t=110s (50+60), NOT t=60s.
```

### T-SLEEP-4: Cancel timer on stop
```
Setup: Soft sleep timer running
Call: stopEngine(model: "test")
Expect: Timer cancelled (no POST sent after engine stops)
```

### T-SLEEP-5: Dead code cancel (P13)
```
Verify: cancelIdleTimer() cancels model, model+".soft", model+".deep"
Note: model key cancel is no-op (no timer stored under bare key)
Harmless but should be cleaned up.
```

---

## Engine Compatibility: Sleep/Wake

### T-WAKE-1: JIT auto-wake from soft sleep (P4)
```
Setup: Engine in soft_sleep state (POST /admin/soft-sleep)
Send: POST /v1/chat/completions
Expect: Engine auto-wakes (JIT middleware), processes request, returns response
Timing: Near-instant (soft sleep only cleared caches, model still loaded)
```

### T-WAKE-2: JIT auto-wake from deep sleep (P4)
```
Setup: Engine in deep_sleep state (POST /admin/deep-sleep)
Send: POST /v1/chat/completions
Expect: Engine auto-wakes, reloads model, processes request
Timing: 5-60s depending on model size (large JANG MoE models take 30-60s for mmap load)
Verify: Engine waits up to 300s for model reload before returning 503
```

### T-WAKE-3: Concurrent requests during wake
```
Setup: Engine in deep_sleep. Two requests arrive simultaneously
Expect: First request triggers wake, second waits on _wake_lock
After wake: Both requests proceed
Verify: Only one admin_wake() call (lock prevents double wake)
```

### T-WAKE-4: Health check during sleep
```
Setup: Engine in deep_sleep state
Call: GET /health
Expect: Returns 200 (process alive). JSON contains standby_state: "deep_sleep", model_loaded: false
Note: Swift only checks status code, doesn't read body. Engine appears "healthy" even when sleeping.
Gap: Swift could use model_loaded/standby_state for better UI status (green dot vs sleeping indicator)
```

---

## Edge Cases

### T-EDGE-1: Multiple models running (P12 memory)
```
Setup: Launch 2 large models (e.g., 2x 27B Gemma)
Expect (current): Both launch, system may OOM. No pre-launch memory check.
Desired: Check available memory before launch, warn user if insufficient.
```

### T-EDGE-2: Model crash with active stream
```
Setup: Streaming response, kill engine mid-stream
Expect: VMLXService bytes.lines throws, error caught, partial response preserved
Monitor: Detects crash, auto-restart with backoff
```

### T-EDGE-3: Global orphan scan on launch (P17)
```
Setup: Prior app crash left orphaned python3 -m vmlx_engine.cli processes
Launch: New app instance starts
Current: Only checks specific port via lsof
Desired: Scan for all vmlx_engine.cli processes and kill them
```

### T-EDGE-4: System sleep (macOS) (L10, L15)
```
Setup: Engine running, Mac goes to sleep
On wake: Engine process may be suspended or killed by OS
Current: No NSWorkspace.didWakeNotification handler
Desired: On wake, health-check all registered engines, unregister dead ones
```

### T-EDGE-5: Memory pressure (L16)
```
Setup: Engine running, system RAM low
Current: No DispatchSource.makeMemoryPressureSource() handler
If kernel kills Python: Monitor auto-restarts, but may OOM again (capped at 3 retries)
Desired: On memory warning → soft-sleep. On critical → deep-sleep.
```

---

## Stderr Diagnostics

### T-STDERR-1: Single line capture
```
Setup: Engine crashes with "RuntimeError: CUDA OOM"
Expect: VMLXError.engineCrashed(model: "test", stderr: "RuntimeError: CUDA OOM")
```

### T-STDERR-2: Multi-line OOM error (P8)
```
Setup: Engine crashes with 5-line traceback ending in OOM
Current: Only last line captured (e.g., "MemoryError")
Desired: Last 5-10 lines captured for full context
```

### T-STDERR-3: stderr pipe cleanup
```
Setup: Engine exits normally
Expect: Pipe closes, Task.detached reading loop ends naturally
Verify: No task leak
```

---

## Test Infrastructure Notes

- **Process tests:** Need actual Python engine binary or mock subprocess
- **Health tests:** Can use a lightweight HTTP server returning 200/503
- **Sleep tests:** Need actual engine with model loaded (or mock /admin endpoints)
- **Timing tests:** Use `CFAbsoluteTimeGetCurrent()` for precise measurements
- **Orphan tests:** Spawn a dummy process on a port, verify it's killed
- **Memory tests:** Use `os_proc_available_memory()` to simulate pressure
