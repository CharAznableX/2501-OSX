# UI/UX Audit

## How It Works

### Chat UI Architecture

```
ChatView (SwiftUI)
  ├── ChatSessionSidebar (left)
  ├── messageThread() → MessageThreadView → MessageTableRepresentable (NSTableView)
  │     ├── NativeMessageCellView (per message)
  │     │     ├── NativeMarkdownView (content rendering)
  │     │     └── NativeThinkingView (collapsible thinking box)
  │     └── ScrollAnchorManager (auto-scroll)
  └── FloatingInputCard (bottom, always visible)
        ├── modelSelectorChip (left — model name + green/gray dot)
        ├── thinkingToggleChip (checkbox)
        ├── parserConfigChip (wrench icon)
        ├── modelOptionsSelectorChip (sliders icon)
        ├── sandboxToggleChip
        ├── clipboardToggleChip
        ├── inferenceStatsChip (right — TPS, TTFT, cache, tokens)
        └── contextIndicatorChip (right — token count)
```

### Model Selector Chip

**File:** `Views/Chat/FloatingInputCard.swift` (lines 1139-1216)

- Green dot: engine running (`VMLXGateway.shared.port() != nil`)
- Gray dot: engine not loaded
- Displays model `displayName`, VLM eye icon, parameter count badge
- Right-click: "Unload Model" context menu (only when running)
- Model picker popover: `ModelPickerView` with grouping, search, VLM badge

**Engine status polling:**
- `checkEngineStatus()` (line 1120): async check of `VMLXGateway.shared.port(for:)`
- Called on: `.onAppear`, `.onChange(of: selectedModel)`, `.onChange(of: isStreaming)` when streaming ends

### Inference Stats Chip

**File:** `Views/Chat/FloatingInputCard.swift` (lines 986-1041)

Visible when `inferenceStats.showStats && (isGenerating || completionTokens > 0)`.

Layout:
- `X t/s` — tokens per second (accent color during gen, secondary when done)
- `Xms` or `X.Xs` — time to first token
- Lightning bolt + count — cached tokens (green)
- `prompt→completion` — token counts

Style: monospaced font size 10, capsule background, `lineLimit(1)`, `fixedSize(horizontal: true)`

### Thinking Toggle Chip

**File:** `Views/Chat/FloatingInputCard.swift` (lines 1220-1260)

- Shows checkbox (square/checkmark) + "Thinking" label
- Only visible when `ModelProfileRegistry.profile(for: model)?.thinkingOption` exists
- When nil (never toggled): shows as OFF
- Saves to `ModelOptionsStore` under `"disableThinking"` key
- Does NOT restart engine — UI display toggle only

### Parser Config Chip

**File:** `Views/Chat/FloatingInputCard.swift` (lines 1282-1376)

Wrench icon chip → popover with:
- Tool Call Parser picker (16 options)
- Reasoning Parser picker (7 options)
- "Saved per model" note
- Change triggers engine restart

### Settings View

**File:** `Views/Settings/ConfigurationView.swift`

Sections:
- Server (port, network exposure, login, dock icon, appearance)
- Local Inference (engine settings, cache, parsers, power, performance)
- Stats Display toggle
- Model Eviction Policy

All changes buffered in `@State` vars, committed on "Save".

### Model Detail View

**File:** `Views/Model/ModelDetailView.swift`

- Model info: name, parameter count, quantization, VLM badge
- Parser Configuration card with same tool/reasoning pickers
- Download controls with progress
- Loads per-model options on appear

### Model Cache Inspector

**File:** `Views/Model/ModelCacheInspectorView.swift`

- Lists running engines from `VMLXGateway.shared.allInstances()`
- Shows model name, port, uptime
- Stop individual engines or clear all

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| U1 | **No loading indicator during cold start** — User sends message, engine needs 0-120s to start. No spinner, progress bar, or status text. The UI appears frozen. |
| U2 | **Error messages in chat** — When engine crashes, OOM, or times out: does the user see a clear error in the chat? Or does the spinner just hang? |
| U3 | **Stats chip at narrow widths** — `fixedSize(horizontal: true)` prevents truncation but could overflow the container at very narrow window widths. Previous session noted this. |
| U4 | **Engine status dot accuracy** — The green/gray dot is checked via `VMLXGateway.shared.port()`. But if engine crashed between requests, the gateway still has the port registered (no periodic health check). Dot shows green for a dead engine. |

### Streaming Display

| # | Issue | Notes |
|---|-------|-------|
| U5 | **Scroll anchoring during high-speed streaming** — At 80+ tok/s with UI sync at 100-250ms, the table reconfigures 4-10 times/sec. Does auto-scroll keep up smoothly? |
| U6 | **Thinking box during streaming** — Content updates live as thinking tokens arrive. Does the expand/collapse state persist correctly during rapid updates? |
| U7 | **Model picker locked during streaming** — `isStreaming = true` should disable the model picker. Verify user can't accidentally switch models mid-generation. |

### Session Management

| # | Issue | Notes |
|---|-------|-------|
| U8 | **Session restore** — On app restart, `ChatSession.load(from:)` restores selectedModel + conversation. Does it also load per-model options correctly? (Fixed in previous session via explicit loadOptions call.) |
| U9 | **New chat with same model** — User starts new chat, same model already loaded. `ensureEngineRunning()` finds existing port in gateway. No reload needed. Verify this path works correctly. |
| U10 | **Session title** — Auto-generated from first message? Or manual? |

### Menu Bar

| # | Issue | Notes |
|---|-------|-------|
| U11 | **Activity dot** — Green blinking during generation. `ServerController.signalGenerationStart()/End()`. Verify timing matches actual generation, not the full request lifecycle. |
| U12 | **Menu bar items** — What's in the status item menu? Model status? Quick settings? |

### Accessibility

| # | Issue | Notes |
|---|-------|-------|
| U13 | **VoiceOver** — NSTableView-based message list. Are cells accessible? |
| U14 | **Keyboard navigation** — Can user send messages, switch models, toggle thinking via keyboard? |
| U15 | **Dark/Light mode** — Theme system. Does it respect system appearance setting? |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Model selector chip | Working | Green/gray dot, context menu |
| Inference stats chip | Working | TPS, TTFT, cache, tokens |
| Thinking toggle | Working | Per-model, OFF by default |
| Parser config chip | Working | Tool + reasoning pickers |
| Model picker popover | Working | Grouping, search, VLM badge |
| Settings view | Working | All panels, save logic |
| Model detail view | Working | Parser config, download controls |
| Cache inspector | Working | Running engine list |
| Loading progress | **Missing** | No visual feedback during engine start |
| Error display in chat | **Needs verify** | May show generic error |
| Stats at narrow widths | **Needs verify** | May overflow |
| Engine status accuracy | **Bug** | Green dot for crashed engine |
