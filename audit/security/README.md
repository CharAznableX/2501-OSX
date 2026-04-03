# Security Audit

## How It Works

### Network Exposure

**SwiftNIO gateway** (port 1337):
- Default: `127.0.0.1` (localhost only)
- Configurable: `exposeToNetwork` → `0.0.0.0` (LAN)
- Auth: loopback connections skip auth when `trustLoopback = true`
- LAN: requires `Bearer osk-v1` token

**Python engine** (random port):
- Always binds to `127.0.0.1` (hardcoded in VMLXEngineConfig line 27)
- No authentication between Swift gateway and Python engine
- No TLS

### Authentication

**File:** `Networking/HTTPHandler.swift` (lines 141-192)

- Loopback check: `remoteAddress` is `127.0.0.1` or `::1`
- Token validation: `APIKeyValidator` checks `Authorization: Bearer osk-v1-*`
- Public paths: `/`, `/health` exempt from auth
- Plugin routes: handle own auth

### Process Isolation

**File:** `Services/Inference/VMLXProcessManager.swift` (lines 88-107)

Python subprocess runs with:
- `PYTHONNOUSERSITE=1` — no user site-packages
- `PYTHONPATH=""` — no external module paths
- `-s` flag — suppresses user site-packages
- `PYTHONHOME` set to bundled Python root
- Clean env: only HOME, PATH, TMPDIR, DYLD, Metal vars inherited
- `PYTHONDONTWRITEBYTECODE=1` — no .pyc files

### HF_TOKEN Handling

**Line 104:**
```swift
if let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"] ?? ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"] {
    env["HF_TOKEN"] = hfToken
}
```

- Passed to child process for gated model access (Llama 3, Gemma, etc.)
- Visible in `ps aux` output (process environment)

### CORS

**File:** `Models/Configuration/ServerConfiguration.swift`

- `allowedOrigins: [String]` — empty = disabled, `"*"` = any origin
- Headers set on every response via `stateRef.value.corsHeaders`
- OPTIONS preflight handled in HTTPHandler

### API Key Generation

- `osk-v1` prefix for Osaurus API keys
- Generated/validated by `APIKeyValidator`
- Storage mechanism: needs investigation

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| SEC1 | **No auth between Swift ↔ Python** — Any process on localhost can hit the Python engine's random port. The port is discoverable via `lsof`. For a desktop app this is acceptable, but if `exposeToNetwork` is true, the Python port is still localhost-only (safe). |
| SEC2 | **HF_TOKEN in `ps aux`** — Process environment is visible to other users on the system. For single-user macOS this is low risk. But if the app runs on a shared server (unlikely), tokens are exposed. |
| SEC3 | **CORS `*` + loopback trust** — If `allowedOrigins: ["*"]` and `trustLoopback: true`, any website visited in the user's browser can make fetch requests to `localhost:1337` and get responses. This is a CSRF-like attack vector. |
| SEC4 | **No request body size limit** — HTTPHandler accumulates the entire request body in memory. A malicious client could send a multi-GB body to exhaust memory. |

### Network

| # | Issue | Notes |
|---|-------|-------|
| SEC5 | **LAN exposure without TLS** — When `exposeToNetwork = true`, API tokens travel in plaintext over the network. Should warn user or require TLS. |
| SEC6 | **Random port predictability** — Python engine uses OS-assigned random ports. On macOS, ephemeral ports are in the 49152-65535 range. An attacker scanning this range could find the engine. |
| SEC7 | **Plugin route auth** — Plugins handle their own auth. A malicious plugin could expose unauthenticated endpoints. |

### Input Validation

| # | Issue | Notes |
|---|-------|-------|
| SEC8 | **Message content injection** — User messages are passed through to the Python engine via JSON. No sanitization. The engine processes them as text. Low risk for command injection since messages are JSON-encoded. |
| SEC9 | **Tool arguments injection** — Tool call arguments from the LLM are JSON strings. If a tool executes these as commands (e.g., sandbox shell), proper escaping is needed. |
| SEC10 | **Model path traversal** — Model paths come from filesystem scanning. But the model picker also accepts user input (search). Verify no path traversal in model resolution. |

### Data Privacy

| # | Issue | Notes |
|---|-------|-------|
| SEC11 | **Conversation history persistence** — Sessions stored in SQLite. Not encrypted. Anyone with file access can read conversations. |
| SEC12 | **InsightsService logs** — Inference logs (ring buffer of 500) include model, token counts, duration. Stored in memory only (not persisted). |
| SEC13 | **SystemPrompt in requests** — System prompt is injected into every request. If logging captures full requests, system prompts are logged. |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Localhost-only engine binding | Working | Hardcoded 127.0.0.1 |
| API auth (Bearer token) | Working | osk-v1 tokens |
| Loopback trust | Working | Standard for desktop apps |
| Process env isolation | Working | Clean env, no user packages |
| CORS configuration | Working | Configurable origins |
| HF_TOKEN passing | Working | Needed for gated models |
| Request body size limit | **Missing** | No limit |
| LAN TLS | **Missing** | Plaintext over network |
| CORS wildcard warning | **Missing** | No warning about risks |
| Conversation encryption | **Missing** | SQLite unencrypted |
