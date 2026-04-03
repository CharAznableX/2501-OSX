# Main Branch Delta

## Branch Status

**Our branch:** `feature/osaurus-vmlx-py` — 9 commits ahead of main
**Merge base:** `28e4144d` (tip of main, v0.15.18)
**Main is fully contained** — no missing features or commits to port.

### Our Commits (on top of main)

```
4ef86db4 Fix parser restart, stop abort, streaming batch, Gemma 4 config
64b6288d Load saved per-model parser options on startup and session restore
1637380d Reduce streaming overhead for smoother 80+ tok/s rendering
7cd3a4dc Fix duplicate engine launches via dual-key gateway registration
825aae1d Compact inference stats chip for narrow windows
ab5454c4 Optimize streaming for high token rates (80+ tok/s)
651157ad Fix thinking toggle display, enable_thinking API passthrough, VLM content
a6dbe2f1 Add thinking toggle for all local models
6c95416b Replace mlx-swift-lm with vmlx Python engine backend
```

### Other Branches (NOT relevant)

- `feature/vmlx` — 167 commits, old Swift-native VMLXRuntime approach. **We are replacing this.** Not to be merged or considered.
- `feature/ollama-compatible-endpoints` — 1 commit (Interop API endpoints). Minor, can merge later if needed.
- `ritave/bonjour-on-server` — 6 commits (Bonjour agents run tools on server). Bonjour discovery already integrated into main.
- `feature/relay-sites` — 1 commit (static website sharing via relay). Unrelated to engine.

### What Main Had Before We Branched

All features from v0.15.18 are in our branch:
- Bonjour broadcast/discovery (#683)
- Clipboard monitoring from other apps
- Autonomous execution mode / auto plugin creation (#765)
- UI freeze fix for async skill I/O (#763)
- Linting with lefthook (#761)
- Embedding/skills crash fixes (#760)
- Chat view performance stress testing
- Plugin agent concurrency improvements
- Metal race condition guards
- FluidAudio package update
- Text selectable in message cells
- Hover buttons in message cells

### What We Changed From Main

We replaced the mlx-swift-lm inference backend with Python vmlx-engine subprocess architecture:
- Deleted 14 MLX Swift files (see CLAUDE.md)
- Created 6 new VMLX Swift files + bundled Python engine source
- Modified 20+ existing files for VMLXService integration
- Removed mlx-swift/mlx-swift-lm Package.swift deps (kept Hub for downloads)
