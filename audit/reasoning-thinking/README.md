# Reasoning / Thinking Audit

## How It Works

Reasoning (thinking) support spans 4 layers: engine parser, SSE transport, Swift processing, and UI display.

### Layer 1: Engine Reasoning Parser

The Python engine's `--reasoning-parser` extracts model-specific thinking tokens into `delta.reasoning_content`:

| Parser | Models | Think Tokens |
|--------|--------|-------------|
| `auto` | Auto-detect from config.json | — |
| `qwen3` | Qwen 3/3.5 | `<think>...</think>` |
| `deepseek_r1` | DeepSeek R1 | `<think>...</think>` |
| `mistral` | Mistral reasoning models | Model-specific |
| `gemma4` | Gemma 4 | Architectural thinking (always on) |
| `openai_gptoss` | GPT-OSS / GLM | `<|thinking|>...<|/thinking|>` |

Parser configured per-model via `ModelOptionsStore` (keys: `"reasoningParser"`).

### Layer 2: SSE Transport

**File:** `Services/Inference/VMLXSSEParser.swift` (lines 97-100)

Engine sends `reasoning_content` alongside `content` in delta:
```json
{"choices":[{"delta":{"reasoning_content":"Let me think...","content":null}}]}
```

`VMLXSSEParser.parse()` extracts both `chunk.content` and `chunk.reasoningContent`.

### Layer 3: VMLXService Tag Wrapping

**File:** `Services/Inference/VMLXService.swift` (lines 114-119, 176-232)

**`showThinking` flag** (line 114-119):
```swift
let showThinking: Bool = {
    if let val = parameters.modelOptions["disableThinking"]?.boolValue {
        return !val  // disableThinking=false means show thinking
    }
    return false  // Default: don't show thinking bubbles
}()
```

- Only accumulates `reasoningContent` when `showThinking == true` (line 176)
- Wraps accumulated reasoning in `<think>...</think>` tags before yielding to stream
- `hasEmittedThinkOpen` tracks tag state for proper open/close
- On stream end: closes any unclosed `<think>` tag (line 259)
- On error: closes any unclosed `<think>` tag (line 267-268)

**History stripping** (lines 465-479):
- `stripThinkingBlocks()` removes `<think>...</think>` and `[THINK]...[/THINK]` from prior assistant messages
- Prevents history contamination (model re-reading its own reasoning)
- Called in `buildRequestBody()` for assistant role messages

### Layer 4: StreamingDeltaProcessor

**File:** `Utils/StreamingDeltaProcessor.swift` (lines 244-278)

State machine parses `<think>` and `</think>` tags:
- `isInsideThinking: Bool` tracks current state
- `pendingTagBuffer: String` holds partial tags across chunk boundaries
- Partial prefix arrays: `["<think", "<thin", "<thi", "<th", "<t", "<"]`
- Routes content to `turn.appendContent()` or `turn.appendThinking()`

### Layer 5: UI Display

**File:** `Views/Chat/NativeThinkingView.swift`

- Collapsible `NSView` with chevron disclosure triangle
- `NativeMarkdownView` child renders thinking content
- Expand/collapse state in `ExpandedBlocksStore` (survives cell reuse)
- 90-degree chevron rotation animation via `CAMediaTimingFunction`

### Thinking Toggle

**File:** `Views/Chat/FloatingInputCard.swift` (lines 1220-1260)

- Checkbox chip: "Thinking" with square/checkmark icon
- When `nil` (never toggled): shows as OFF (auto-detect mode)
- Toggle saves to `ModelOptionsStore.shared.saveOptions()` under `"disableThinking"` key
- Does NOT restart engine — thinking is a UI display toggle only

**Visibility:** Only shown when `ModelProfileRegistry.profile(for: model)?.thinkingOption` is non-nil.

**File:** `Models/Configuration/ModelOptions.swift`

- `LocalMLXThinkingProfile` is a catch-all profile for all local models
- Returns `thinkingOption` for any non-remote model

### enable_thinking API Field

**File:** `Models/API/OpenAIAPI.swift`

- `ChatCompletionRequest.enable_thinking: Bool?` — optional field
- In `ChatEngine.streamChat()` (line 70): merged into `modelOptions["disableThinking"]`
- In `VMLXService.buildRequestBody()` (line 391-394): explicitly NOT sent to engine
- Comment explains: "Sending enable_thinking:true breaks Gemma 4 2-bit (all output goes to thinking channel)"

### Reasoning Parser Picker

**File:** `Views/Chat/FloatingInputCard.swift` (lines 1349-1367)

Options:
- Auto-detect → `"auto"`
- None → `"none"`
- Qwen 3 → `"qwen3"`
- DeepSeek R1 → `"deepseek_r1"`
- Mistral → `"mistral"`
- Gemma 4 → `"gemma4"`
- GPT-OSS / GLM → `"openai_gptoss"`

Change triggers engine restart via `setParserAndRestart()`.

---

## What Needs Checking

### Critical

| # | Issue | Notes |
|---|-------|-------|
| R1 | **Reasoning content in external API** — `reasoning_content` from engine is wrapped in `<think>` tags by VMLXService. When HTTPHandler streams to external clients, they get `<think>` tags in `delta.content` instead of `delta.reasoning_content`. **Known bug.** External clients (Cursor, Claude Desktop) see raw think tags. |
| R2 | **Gemma 4 thinking is architectural** — Gemma 4's thinking is always on (part of the model architecture). Setting `enable_thinking: false` breaks output. VMLXService correctly never sends this flag. But does the thinking toggle UI correctly show this model's behavior? |
| R3 | **showThinking defaults to false** — When user never touches the toggle, `disableThinking` is nil, so `showThinking = false`. This means reasoning content is silently discarded. Is this the right default? Users might expect to see thinking. |
| R4 | **Reasoning parser picker missing some parsers** — The engine may support more reasoning parsers than the 7 listed in the UI picker. If new parsers are added to the engine, the UI won't show them. |

### Edge Cases

| # | Issue | Notes |
|---|-------|-------|
| R5 | **Mixed reasoning + content in one batch** — VMLXService batches at 30ms. If reasoning ends and content begins within one window, the yield order is: reasoning batch → `</think>` → content batch. StreamingDeltaProcessor handles this via tag parsing, but verify no content leaks into thinking or vice versa. |
| R6 | **Incomplete `<think>` tag at stream end** — If the engine abruptly stops (crash/timeout) mid-reasoning, `finalize()` in StreamingDeltaProcessor drains remaining buffer. If `isInsideThinking = true`, everything goes to appendThinking. But the `</think>` close tag is never emitted by VMLXService in error case — wait, it IS emitted at line 267-268. |
| R7 | **History stripping regex** — `stripThinkingBlocks()` uses while-loop range-finding (not regex). Handles multiline. But what about nested `<think>` tags? (Model outputs `<think>inner<think>nested</think>outer</think>`) — the inner `</think>` would close the first match. |
| R8 | **`[THINK]...[/THINK]` format** — Mistral uses `[THINK]` markers. Stripped in history but not parsed by StreamingDeltaProcessor. If the engine sends `[THINK]` in content instead of `reasoning_content`, it won't be routed to thinking box. |
| R9 | **Thinking toggle + parser interaction** — If user sets `disableThinking = true` (thinking OFF) but reasoning parser is active, engine still separates reasoning into `reasoning_content`. VMLXService just discards it (`showThinking = false`). The tokens are still generated but invisible. Wasted compute? |
| R10 | **Auto-detect fallback** — When parser is "auto" and engine can't detect the model's reasoning format, what happens? Does it fall back to no reasoning? Or error? |
| R14 | **stripThinkingBlocks missing Gemma 4 format** — `stripThinkingBlocks()` handles `<think>` and `[THINK]` but NOT Gemma 4 `<\|channel>thought...<channel\|>` or DeepSeek R1 markers. Prior messages with these formats re-sent to engine unstripped → confuses generation. |
| R15 | **Parser auto-detection mapping** — Qwen 3.5 uses `gemma4` reasoning parser (not `qwen3`). Step 3.5 also uses `gemma4`. This is correct per `model_config_registry.py` but potentially confusing. |

### Per-Model Options

| # | Issue | Notes |
|---|-------|-------|
| R11 | **Options not loading on startup** — Fixed in previous session: explicit `ModelOptionsStore.shared.loadOptions()` calls in `ChatSession.reset()` and `load(from:)`. Verify this is working correctly. |
| R12 | **Parser options in ModelDetailView** — `ModelDetailView` has its own parser picker that saves under `model.id`. Verify it uses the same key format as `FloatingInputCard.setParserAndRestart()`. |
| R13 | **3-key fallback in ensureEngineRunning** — Options loaded by trying `requestedModel`, then `resolved.path`, then `resolved.name`. If user saved under the picker ID (filesystem path) but engine launches with display name, the fallback should find it. Verify. |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Reasoning content parsing | Working | VMLXSSEParser extracts reasoning_content |
| Think tag wrapping | Working | VMLXService wraps in `<think>` tags |
| StreamingDeltaProcessor parsing | Working | State machine with partial tag buffering |
| Thinking box UI | Working | NativeThinkingView, collapsible |
| Thinking toggle | Working | Per-model, persisted via ModelOptionsStore |
| History stripping | Working | Both `<think>` and `[THINK]` formats |
| enable_thinking passthrough | Working | NOT sent to engine (correct) |
| Reasoning parser picker | Working | 7 options + auto + none |
| External API reasoning_content | **Bug** | Leaked as `<think>` tags in content |
| Default thinking visibility | **Design question** | Defaults to OFF (hidden) |
