# Tool Calling Audit

## How It Works

Tool calling flows through two paths: Chat UI (internal) and HTTP API (external clients).

### Tool Registration

**File:** `Tools/ToolRegistry.swift`

- `@MainActor` singleton holding `[String: OsaurusTool]` keyed by name
- Categories: `builtInToolNames`, `sandboxToolNames`, `mcpToolNames`, `pluginToolNames`
- `ToolEntry` struct: name, description, enabled flag, JSON schema, token count estimate

### Tool Call Flow (Chat UI)

```
ChatSession.send()
  → ChatEngine.streamChat(request with tools)
    → VMLXService.streamWithTools(tools: [...])
      → buildRequestBody() serializes tools as OpenAI format
      → POST to Python engine with tools array
      → Engine generates with tool_call_parser
      → SSE chunks: tool_calls deltas (index, id, name, arguments)
      → VMLXSSEParser extracts VMLXToolCallDelta
      → AccumulatedToolCall struct accumulates across chunks
      → finish_reason == "tool_calls":
        → StreamingToolHint.encode(name) + encodeArgs(args) yielded
        → ServiceToolInvocation thrown (first tool only!)
    → ChatEngine catches ServiceToolInvocation
      → HTTPHandler converts to OpenAI tool_calls response
```

### Tool Call SSE Protocol

The engine sends tool calls in two chunk types:

**Chunk 1 (data):** Contains `id`, `function.name`, `function.arguments` (partial)
```json
{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc123","function":{"name":"search","arguments":"{\"q"}}]}}]}
```

**Chunk 2+ (continuations):** Contains `index` + more `arguments` (no `id`, no `name`)
```json
{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"uery\":\"hello\"}"}}]}}]}
```

**Finish chunk:** `finish_reason: "tool_calls"`

### AccumulatedToolCall

**File:** `Services/Inference/VMLXService.swift` (lines 286-290)

```swift
private struct AccumulatedToolCall {
    let id: String
    let name: String
    var arguments: String  // concatenated across chunks
}
```

Accumulation logic (lines 195-207):
- If `delta.index < accumulatedToolCalls.count`: append arguments to existing
- Else: create new `AccumulatedToolCall` with id, name, initial arguments

### ServiceToolInvocation

**File:** `Services/Inference/ModelService.swift`

An `Error` subclass that carries tool call data through the `AsyncThrowingStream<String, Error>` error channel:
- `toolName: String`
- `jsonArguments: String`
- `toolCallId: String`
- `geminiThoughtSignature: String?` (for Gemini thought-with-tool pattern)

### StreamingToolHint

**File:** `Services/Inference/ModelService.swift`

Uses Unicode sentinel `\u{FFFE}` (non-character, can never appear in LLM output) as prefix:
- `encode(name)` → `"\u{FFFE}TOOL:\(name)"`
- `encodeArgs(args)` → `"\u{FFFE}ARGS:\(args)"`
- `isSentinel(delta)` → checks for `\u{FFFE}` prefix

### Tool Parsers (Engine-Side)

The Python engine supports 15+ tool call parsers, configured via `--tool-call-parser`:

| Parser | Models | Tag |
|--------|--------|-----|
| `auto` | Auto-detect from config.json | — |
| `qwen` | Qwen 2.5/3 | `qwen` |
| `llama` | Llama 3.x | `llama` |
| `mistral` | Mistral, Codestral | `mistral` |
| `hermes` | NousResearch Hermes | `hermes` |
| `deepseek` | DeepSeek V2/V3 | `deepseek` |
| `nemotron` | NVIDIA Nemotron | `nemotron` |
| `minimax` | MiniMax M2.5 | `minimax` |
| `kimi` | Moonshot Kimi | `kimi` |
| `granite` | IBM Granite | `granite` |
| `functionary` | MeetKai Functionary | `functionary` |
| `glm47` | GLM-4.7 | `glm47` |
| `step3p5` | Step 3.5 | `step3p5` |
| `xlam` | Salesforce xLAM | `xlam` |
| `gemma4` | Google Gemma 4 | `gemma4` |

### Parser Picker UI

**File:** `Views/Chat/FloatingInputCard.swift` (lines 1319-1345)

Wrench icon chip → popover with two pickers:
- Tool Call Parser: auto, none, qwen, llama, mistral, hermes, deepseek, nemotron, minimax, kimi, granite, functionary, glm47, step3p5, xlam, gemma4
- Reasoning Parser: auto, none, qwen3, deepseek_r1, mistral, gemma4, openai_gptoss

**Save:** `setParserAndRestart()` (line 1303) saves to `ModelOptionsStore` and calls `VMLXProcessManager.shared.stopEngine()` — engine restarts with new parser on next message.

### Per-Model Parser Options

**File:** `Services/ModelOptionsStore.swift`

- Stored in UserDefaults under `"model_options_" + modelId`
- Keys: `"toolParser"`, `"reasoningParser"`, `"disableThinking"`
- Loaded in `VMLXService.ensureEngineRunning()` with 3-key fallback (requestedModel → resolved.path → resolved.name)
- Passed to `VMLXEngineConfig.buildArgs(modelOptions:)` which overrides global config

### Non-Streaming Tool Calls

**File:** `Services/Chat/ChatEngine.swift` (lines 302-396)

`completeChat()` catches `ServiceToolInvocation` and converts to OpenAI-style response:
- Generates `call_<24char>` call ID
- Creates `ToolCall` with `id`, `type: "function"`, `function.name`, `function.arguments`
- Returns `ChatCompletionResponse` with `finish_reason: "tool_calls"`

### MCP Tool Integration

**File:** `Services/MCP/MCPServerManager.swift`

- MCP stdio server registers enabled tools from `ToolRegistry`
- `ListTools` handler converts to MCP Tool format
- `CallTool` handler validates args via `SchemaValidator`, dispatches to `ToolRegistry.shared.callTool()`

**HTTP MCP endpoints** (HTTPHandler):
- `GET /mcp/tools` — list enabled tools
- `POST /mcp/call` — call tool by name

---

## What Needs Checking

### Critical

| # | Issue | File | Line |
|---|-------|------|------|
| T1 | **Parallel tool calls — only first dispatched** — `ServiceToolInvocation` thrown for `accumulatedToolCalls[0]` only. Remaining tools emitted as `StreamingToolHint` sentinels but never executed. **Known limitation.** Need to design multi-tool dispatch. | VMLXService.swift | 240-241 |
| T2 | **Tool call during thinking** — If engine returns tool_calls while reasoning_content is being emitted, the `hasEmittedThinkOpen` may be true. The tool call finish branch (line 234) checks this condition BEFORE the `return` at 246, but the error catch at 267 handles the close tag. Verify no unclosed `<think>` tags when tool calls interrupt reasoning. | VMLXService.swift | 234-247 |
| T3 | **Parser picker tag values must match engine exactly** — Picker tags like `"qwen"`, `"llama"`, etc. are passed directly to `--tool-call-parser`. If the engine CLI expects different strings (e.g., `"qwen2"` vs `"qwen"`), tool calling silently fails. | FloatingInputCard.swift | 1329-1344 |
| T4 | **Tool choice `.function()` binding** — `ToolChoiceOption.function(let fn)` binds a struct. `fn.function.name` extracts the name (VMLXService line 379). Verify this matches the FunctionName struct's property. | VMLXService.swift | 379 |

### Edge Cases

| # | Issue | Notes |
|---|-------|-------|
| T5 | **Very large tool arguments** — Arguments accumulate via string concatenation with no size limit. A tool returning 100KB+ JSON could cause memory pressure. |
| T6 | **Empty tool arguments** — Engine may send `"arguments": ""` or `"arguments": "{}"`. Verify downstream handles both. |
| T7 | **Tool call with no id** — Continuation chunks have `id: ""`. The `AccumulatedToolCall.id` is set from the first chunk. If the first chunk somehow has no id, the call ID will be empty. |
| T8 | **Multiple tools same name** — If engine returns two tool calls with the same function name but different arguments, both accumulate correctly (indexed by position, not name). But dispatch only handles index 0. |
| T9 | **MCP tool schema validation** — `SchemaValidator.validate` before dispatch. What happens if the LLM generates invalid args? Is the error propagated back to the model for retry? |
| T10 | **Tool enabled/disabled persistence** — Can users toggle individual tools? Does this persist across app restarts? |

### Parser Configuration

| # | Issue | Notes |
|---|-------|-------|
| T11 | **Parser auto-detection accuracy** — `"auto"` relies on `model_config_registry` in the Python engine. Does it correctly detect all 15+ model families? What happens for unknown models? |
| T12 | **Parser change requires engine restart** — `setParserAndRestart()` stops the engine. But the user might not send another message immediately. The engine stays stopped until next message. Is this UX clear? |
| T13 | **Global vs per-model parser** — VMLXEngineConfig uses per-model options first, then falls back to global `config.toolCallParser`. If per-model is "auto" and global is "qwen", which wins? Per-model "auto" is non-empty, so it wins. Is this correct? |

---

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| Tool call SSE parsing | Working | Two-chunk protocol, argument accumulation |
| Single tool dispatch | Working | First tool via ServiceToolInvocation |
| Parallel tool dispatch | **Limitation** | Only first tool executed |
| Parser picker UI | Working | 15+ tool parsers, 7 reasoning parsers |
| Per-model parser persistence | Working | ModelOptionsStore, 3-key fallback |
| Parser → engine restart | Working | stopEngine() on change |
| MCP tool listing | Working | /mcp/tools endpoint |
| MCP tool calling | Working | /mcp/call with schema validation |
| StreamingToolHint sentinels | Working | Unicode sentinel encoding |
| Non-streaming tool calls | Working | ChatEngine.completeChat() |
