# Engine Sync Audit

## How It Works

The bundled vmlx-engine source lives in `Resources/vmlx_engine/` — a stripped copy of the upstream Python engine from the separate vmlx-engine repo.

### Bundled Source

**Location:** `Resources/vmlx_engine/`

82 Python files. Excluded from upstream:
- `audio/` — audio processing (not needed for text/vision inference)
- `mcp/` — MCP server functionality (handled by Swift MCPServerManager)
- `image_gen.py` — image generation
- `gradio*.py` — Gradio UI (UI is Swift)
- `benchmark.py` — benchmarking tools
- `commands/` — CLI commands beyond `serve`

**Patches applied by `scripts/bundle-python.sh`:**
- `cli.py` — try/except for image_gen and commands imports (graceful missing module handling)

### Bundle Script

**File:** `scripts/bundle-python.sh`

1. Downloads Python 3.12 (python-build-standalone from Astral)
2. Installs dependencies: mlx, mlx-lm, mlx-vlm, transformers, fastapi, uvicorn, etc.
3. Applies patches
4. Cleans up: removes tests, docs, unnecessary files
5. Output: `Resources/bundled-python/python/` (~400-640MB, gitignored)

### Branch Context

- **Our branch:** `feature/osaurus-vmlx-py` — replaces mlx-swift with Python subprocess
- **Main branch:** fully contained in our branch (we branched from tip of v0.15.18)
- **`feature/vmlx`:** old Swift-native VMLXRuntime attempt — **NOT relevant**, we are replacing this approach
- **Upstream vmlx-engine:** separate Python repo (`~/mlx/vllm-mlx/` in dev)

### Interface Contract

The Swift ↔ Python interface has two surfaces:

**1. CLI args** (launch time):
```
python3 -s -m vmlx_engine.cli serve <model> --port <port> [28+ flags]
```
Mapped by `VMLXEngineConfig.buildArgs()`. Any upstream CLI changes must be reflected here.

**2. HTTP API** (runtime):
- `POST /v1/chat/completions` — OpenAI-compatible, SSE streaming
- `GET /health` — health check (200 = ready)
- `POST /admin/soft-sleep` — clear GPU caches
- `POST /admin/deep-sleep` — unload model from VRAM

SSE format parsed by `VMLXSSEParser`: content, reasoning_content, tool_calls, usage, finish_reason, [DONE].

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| ES1 | **Version tracking** — No upstream commit hash recorded in the bundled source. When the engine is updated, there's no way to know which version is bundled. Should add a `VERSION` or `__version__` file. |
| ES2 | **CLI arg compatibility** — If upstream renames/removes a flag, `VMLXEngineConfig.buildArgs()` passes invalid args and engine fails to start with a cryptic Python error. Need validation or version check. |
| ES3 | **New parsers in upstream** — If upstream adds tool/reasoning parsers, the Swift picker UI doesn't show them. Need mechanism to discover available parsers (e.g., engine `/capabilities` endpoint). |
| ES4 | **Model-specific fixes** — Upstream engine gets fixes for new model architectures (Gemma 4 decoder, NemotronH hybrid SSM, Mistral 4 MLA, etc.). Bundled source must be kept in sync. |

### Sync Process

| # | Issue | Notes |
|---|-------|-------|
| ES5 | **Manual sync** — Updating bundled source is manual: copy files, re-apply patches, rebuild. Should have a script or documented process. |
| ES6 | **Patch maintenance** — Patches in bundle script must be re-applied on each sync. If upstream changes patched files, patches may fail silently. |
| ES7 | **Dependency versions** — Bundle script pins deps via pip. Upstream may require newer versions. Need to track. |
| ES8 | **Python version** — Bundle uses 3.12. If upstream requires 3.13+ features, bundle breaks. |

### Feature Parity

| # | Issue | Notes |
|---|-------|-------|
| ES9 | **Admin endpoints** — Are there other admin endpoints beyond soft-sleep/deep-sleep? (e.g., `/admin/stats`, `/admin/cache-clear`, `/admin/reload`) that Swift could leverage. |
| ES10 | **Engine capabilities** — No mechanism for Swift to query engine version, available parsers, supported features. Would help with forward compatibility. |
| ES11 | **audio/ exclusion** — If engine later integrates audio inference, bundled source needs updating. |

### Build

| # | Issue | Notes |
|---|-------|-------|
| ES12 | **Bundle size** — 400-640MB for bundled Python + deps. Ships with app. Could strip more aggressively. |
| ES13 | **CI integration** — Is `bundle-python.sh` run in CI? Or only manually before release? |
| ES14 | **Reproducibility** — pip versions, mirrors could vary between builds. |

### Sync Strategy Options

| Option | Approach | Pro | Con |
|--------|----------|-----|-----|
| A (current) | Manual copy | Simple | Drift accumulates, easy to forget |
| B | Symlink `Resources/vmlx_engine/` → upstream | Always in sync | Requires upstream repo on build machine |
| C | Build script `sync-engine.sh` | Automated, strips excluded modules | Needs manual trigger |
| D | Git submodule | Version-pinned, easy update | Submodule complexity |

**Recommendation:** Option C for production, Option A acceptable during active development. Add `SYNC_VERSION` file with upstream git hash.

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Bundled engine source | Working | 82 Python files, stripped |
| Bundle script | Working | Downloads Python + deps |
| CLI arg interface | Working | 28+ flags mapped |
| HTTP API interface | Working | chat/completions, health, admin |
| Main branch parity | Complete | All main features in our branch |
| Version tracking | **Missing** | No commit hash recorded |
| Automated sync | **Missing** | Manual process |
| Engine capabilities query | **Missing** | No discovery endpoint |
