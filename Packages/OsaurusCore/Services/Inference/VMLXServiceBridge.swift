//
//  VMLXServiceBridge.swift
//  osaurus
//
//  Bridges VMLXRuntime's VMLXService into Osaurus's ToolCapableService protocol.
//  Handles type mapping between Osaurus ChatMessage/Tool/GenerationParameters and
//  VMLXRuntime's VMLXChatMessage/VMLXToolDefinition/SamplingParams.
//

import Foundation
import VMLXRuntime

private func _vmlxLog(_ msg: String) {
    let line = "[\(Date())] \(msg)\n"
    let path = "/tmp/vmlx_debug.log"
    if !FileManager.default.fileExists(atPath: path) {
        FileManager.default.createFile(atPath: path, contents: nil)
    }
    if let fh = FileHandle(forWritingAtPath: path) {
        fh.seekToEndOfFile()
        fh.write(line.data(using: .utf8)!)
        fh.closeFile()
    }
}

// MARK: - Bridge Actor

/// Adapts VMLXService (from VMLXRuntime) to Osaurus's ToolCapableService protocol,
/// enabling it to participate in Osaurus's ModelServiceRouter alongside MLXService
/// and FoundationModelService.
actor VMLXServiceBridge: ToolCapableService {

    nonisolated let id: String = "vmlx"

    /// Shared singleton for model management (unload, status checks).
    static let shared = VMLXServiceBridge()

    private let service: VMLXService

    /// Global topP default from ServerConfiguration, updated via applyRuntimeConfig().
    /// Used as fallback when no per-request topP override is set.
    private var globalTopP: Float = 0.9

    /// Global parser overrides from Server Settings → Local Inference → Parsers.
    /// Per-model overrides (from ModelOptionsStore) take priority over these.
    private var globalToolParser: String?
    private var globalReasoningParser: String?

    /// Config-based formats from the loaded model's config.json (via ModelFamilyConfig).
    /// Updated after each model load. Used by the UI streaming middleware to match
    /// the engine's auto-detection instead of relying on model name substring matching.
    private var configReasoningFormat: String?
    private var configToolFormat: String?
    private var configThinkInTemplate: Bool = false

    init(service: VMLXService = .shared) {
        self.service = service
    }

    // MARK: - ModelService

    nonisolated func isAvailable() -> Bool {
        service.isAvailable()
    }

    nonisolated func handles(requestedModel: String?) -> Bool {
        service.handles(requestedModel: requestedModel)
    }

    /// Apply user-configured runtime settings from Osaurus's ConfigurationView
    /// to VMLXRuntime's scheduler. Called after model load so UI settings
    /// (KV cache bits, max context, prefill step, etc.) take effect.
    ///
    /// Settings flow:
    ///   ConfigurationView (UI sliders/steppers)
    ///     -> ServerConfiguration (UserDefaults)
    ///       -> RuntimeConfig.snapshot()
    ///         -> VMLXService.applyUserConfig()
    ///           -> VMLXRuntimeActor.applyUserConfig()
    ///             -> SchedulerConfig fields
    ///
    /// What each setting controls in VMLXRuntime:
    ///   topP          -> default nucleus sampling threshold (SamplingParams)
    ///   kvBits        -> KV cache quantization: "q2"/"q4"/"q8" or "none"
    ///   kvGroup       -> quantization group size (default 64)
    ///   maxKV         -> maxNumBatchedTokens (caps context window)
    ///   prefillStep   -> prefillStepSize (tokens per prefill chunk)
    private func applyRuntimeConfig() async {
        let cfg = await RuntimeConfig.snapshot()
        let serverCfg = await ServerController.sharedConfiguration()

        // Store global settings for use in toSamplingParams() fallback
        self.globalTopP = cfg.topP
        self.globalToolParser = serverCfg?.toolParserOverride
        self.globalReasoningParser = serverCfg?.reasoningParserOverride

        let cacheRebuilt = await service.applyUserConfig(
            kvBits: cfg.kvBits,
            kvGroupSize: cfg.kvGroup,
            maxContextLength: cfg.maxKV,
            prefillStepSize: cfg.prefillStep,
            enableDiskCache: serverCfg?.enableDiskCache ?? false,
            enableTurboQuant: serverCfg?.enableTurboQuant ?? false,
            cacheMemoryPercent: serverCfg?.cacheMemoryPercent
        )
        if cacheRebuilt {
            _vmlxLog("[Bridge] Cache settings changed — multi-turn cache cleared")
            await MainActor.run {
                NotificationCenter.default.post(
                    name: .init("VMLXCacheRebuilt"),
                    object: nil,
                    userInfo: ["message": "Cache settings changed — conversation cache cleared"]
                )
            }
        }
    }

    /// Ensure the requested model is loaded before inference.
    /// Auto-loads by resolving model name to directory path via ModelManager.
    /// Track which model is currently loaded so we can detect switches.
    private var currentLoadedModel: String?

    private func ensureModelLoaded(requestedModel: String?) async throws {
        let modelName = requestedModel ?? ""
        let isLoaded = await service.isModelLoaded
        _vmlxLog("[Bridge] ensureModelLoaded: requested='\(modelName)' currentLoaded='\(currentLoadedModel ?? "nil")' isLoaded=\(isLoaded)")
        guard !modelName.isEmpty else {
            throw NSError(domain: "VMLXServiceBridge", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "No model specified"])
        }

        // Check if the CORRECT model is already loaded on the shared runtime.
        // Note: each ChatEngine creates a new VMLXServiceBridge, but the runtime
        // (VMLXRuntimeActor.shared) is a singleton that persists across instances.
        let modelLoaded = await service.isModelLoaded
        if modelLoaded {
            // Check the runtime's model name, not our instance variable
            let runtimeModelName = await service.currentModelName
            if let runtimeName = runtimeModelName,
               modelName.lowercased() == runtimeName.lowercased() {
                _vmlxLog("[Bridge] Model already loaded on runtime, skipping: \(modelName) (runtime: \(runtimeName))")
                currentLoadedModel = modelName
                return
            }
            _vmlxLog("[Bridge] UNLOADING current model to load different one: \(modelName) (runtime: \(runtimeModelName ?? "nil"))")
            await service.unloadModel()
        } else {
            _vmlxLog("[Bridge] No model loaded, loading fresh: \(modelName)")
        }

        // Load the requested model
        if let found = ModelManager.findInstalledModel(named: modelName) {
            // Use the resolved path from MLXModel.localDirectory which handles
            // both standard org/repo models and VMLX-detected models (JANG, HF cache)
            // that live outside the effective models directory.
            let resolved = found.path.resolvingSymlinksInPath()

            // Check if this model type needs MLXService instead of VMLX
            if _isMLXServiceOnlyModel(at: resolved) {
                throw NSError(domain: "VMLXServiceBridge", code: 2,
                             userInfo: [NSLocalizedDescriptionKey: "Model requires MLXService (unsupported architecture for VMLXRuntime)"])
            }

            // VL models load as text-only — vision features won't work but text
            // inference is fine. The model's sanitize() strips vision tower weights.

            do {
                try await service.loadModel(from: resolved)
            } catch {
                _vmlxLog("[Bridge] loadModel FAILED for \(modelName) at \(resolved.path): \(error)")
                throw error
            }
        } else {
            try await service.loadModel(name: modelName)
        }

        currentLoadedModel = modelName

        // Capture the loaded model's config.json-based format detection.
        // This is the source of truth for reasoning/tool parsers (not model name matching).
        if let familyConfig = await service.loadedFamilyConfig {
            let rf = familyConfig.reasoningFormat
            configReasoningFormat = rf == .none ? nil : rf.rawValue
            let tf = familyConfig.toolCallFormat
            configToolFormat = tf == .none ? nil : tf.rawValue
            configThinkInTemplate = familyConfig.thinkInTemplate
            // Update static snapshots for sync access from UI
            Self._lastReasoningFormat = configReasoningFormat
            Self._lastThinkInTemplate = configThinkInTemplate
            _vmlxLog("[Bridge] Model config: reasoning=\(configReasoningFormat ?? "none") tool=\(configToolFormat ?? "none") thinkInTemplate=\(configThinkInTemplate)")
        }

        // Apply user's inference settings
        await applyRuntimeConfig()
    }

    func generateOneShot(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        requestedModel: String?
    ) async throws -> String {
        try await ensureModelLoaded(requestedModel: requestedModel)
        await applyRuntimeConfig()  // Refresh settings on every request
        let vmlxMessages = messages.map { $0.toVMLX() }
        let params = parameters.toSamplingParams(globalTopP: self.globalTopP, globalToolParser: self.globalToolParser, globalReasoningParser: self.globalReasoningParser)
        return try await service.generateOneShot(
            messages: vmlxMessages,
            params: params,
            requestedModel: requestedModel
        )
    }

    func streamDeltas(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        requestedModel: String?,
        stopSequences: [String]
    ) async throws -> AsyncThrowingStream<String, Error> {
        try await ensureModelLoaded(requestedModel: requestedModel)
        await applyRuntimeConfig()
        let vmlxMessages = messages.map { $0.toVMLX() }
        var params = parameters.toSamplingParams(globalTopP: self.globalTopP, globalToolParser: self.globalToolParser, globalReasoningParser: self.globalReasoningParser)
        params.stop = stopSequences

        return try await service.streamDeltas(
            messages: vmlxMessages,
            params: params,
            requestedModel: requestedModel,
            stopSequences: stopSequences
        )
    }

    // MARK: - ToolCapableService

    func respondWithTools(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        stopSequences: [String],
        tools: [Tool],
        toolChoice: ToolChoiceOption?,
        requestedModel: String?
    ) async throws -> String {
        try await ensureModelLoaded(requestedModel: requestedModel)
        await applyRuntimeConfig()
        let vmlxMessages = messages.map { $0.toVMLX() }
        var params = parameters.toSamplingParams(globalTopP: self.globalTopP, globalToolParser: self.globalToolParser, globalReasoningParser: self.globalReasoningParser)
        params.stop = stopSequences
        let vmlxTools = tools.map { $0.toVMLX() }
        let vmlxChoice = toolChoice?.toVMLXString()
        return try await service.respondWithTools(
            messages: vmlxMessages,
            params: params,
            stopSequences: stopSequences,
            tools: vmlxTools,
            toolChoice: vmlxChoice,
            requestedModel: requestedModel
        )
    }

    func streamWithTools(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        stopSequences: [String],
        tools: [Tool],
        toolChoice: ToolChoiceOption?,
        requestedModel: String?
    ) async throws -> AsyncThrowingStream<String, Error> {
        try await ensureModelLoaded(requestedModel: requestedModel)
        await applyRuntimeConfig()
        let vmlxMessages = messages.map { $0.toVMLX() }
        var params = parameters.toSamplingParams(globalTopP: self.globalTopP, globalToolParser: self.globalToolParser, globalReasoningParser: self.globalReasoningParser)
        params.stop = stopSequences
        let vmlxTools = tools.map { $0.toVMLX() }
        let vmlxChoice = toolChoice?.toVMLXString()
        return try await service.streamWithTools(
            messages: vmlxMessages,
            params: params,
            stopSequences: stopSequences,
            tools: vmlxTools,
            toolChoice: vmlxChoice,
            requestedModel: requestedModel
        )
    }

    // MARK: - Model Management Passthrough

    func loadModel(name: String) async throws {
        try await service.loadModel(name: name)
        currentLoadedModel = name
    }

    func unloadModel() async {
        await service.unloadModel()
        currentLoadedModel = nil
    }

    var isModelLoaded: Bool {
        get async { await service.isModelLoaded }
    }

    /// Name of the currently loaded VMLX model (nil if none).
    var loadedModelName: String? {
        get async { await service.currentModelName }
    }

    // MARK: - Static Model Discovery

    /// Check if a model at the given path needs MLXService (not VMLXRuntime).
    /// Reads config.json to check model_type against VMLXModelRegistry.mlxServiceOnlyTypes.
    /// Check if a model is a vision/multimodal model that needs MLXService.
    /// Checks config.json for vision_config, image_token_id, or preprocessor_config.json.
    private func _isVisionModel(at path: URL) -> Bool {
        // Check config.json model_type — only VL model types need vision pipeline.
        // preprocessor_config.json is NOT reliable: JANG text-only conversions often
        // copy it from the VL base model, causing false positives.
        let configURL = path.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        // Check model_type for known VL types
        if let modelType = json["model_type"] as? String {
            let vlTypes: Set<String> = [
                "llava", "llava_next", "llava_onevision",
                "paligemma",
                "qwen2_vl", "qwen2_5_vl", "qwen3_5_vl",
                "phi3_v",
                "pixtral", "mllama",
                "idefics2", "idefics3",
                "internvl_chat",
            ]
            // NOTE: gemma3, gemma4, phi4mm NOT in vlTypes — they use the same
            // model_type for both VL and text-only. Let them load as text;
            // vision features just won't work until VL pipeline is implemented.
            // VL types need vision pipeline. Text subtypes (gemma3_text, qwen3_5_moe) do NOT.
            if vlTypes.contains(modelType) {
                return true
            }
        }
        // Also check if vision_config has actual layer definitions (not just null)
        if let visionConfig = json["vision_config"] as? [String: Any],
           visionConfig["num_hidden_layers"] != nil {
            // Has real vision encoder — but still check model_type isn't text-only
            if let modelType = json["model_type"] as? String,
               modelType.hasSuffix("_text") || modelType.hasSuffix("_moe") {
                return false  // Text-only variant with inherited vision config
            }
            return true
        }
        return false
    }

    private func _isMLXServiceOnlyModel(at path: URL) -> Bool {
        let configURL = path.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = json["model_type"] as? String else {
            return false
        }
        return VMLXModelRegistry.mlxServiceOnlyTypes.contains(modelType)
    }

    /// Return available VMLX model names by scanning well-known directories.
    /// Called from ChatEngine's installedModelsProvider to merge with MLXService models.
    nonisolated static func getAvailableModels() -> [String] {
        ModelDetector.scanAvailableModels().map(\.name)
    }

    /// Return available VMLX models with their actual filesystem paths.
    /// Used by ModelManager.scanLocalModels() to create MLXModel with correct rootDirectory.
    nonisolated static func getAvailableModelsWithPaths() -> [(name: String, path: URL)] {
        ModelDetector.scanAvailableModels().map { ($0.name, $0.modelPath) }
    }

    /// Last-known config formats from the loaded model (synchronous access).
    /// Updated after each model load. Safe for UI reads — worst case is one
    /// message with stale format before the next load updates it.
    nonisolated(unsafe) private static var _lastReasoningFormat: String?
    nonisolated(unsafe) private static var _lastThinkInTemplate: Bool = false

    /// Query the loaded model's config.json-based reasoning format (async).
    /// Returns the ReasoningFormat rawValue (e.g. "qwen3", "mistral", "gptoss") or nil.
    static func getConfigReasoningFormat() async -> String? {
        await shared.configReasoningFormat
    }

    /// Query whether the loaded model's chat template natively injects <think> tags (async).
    static func getConfigThinkInTemplate() async -> Bool {
        await shared.configThinkInTemplate
    }

    /// Synchronous snapshot of the loaded model's reasoning format.
    /// Use from non-async contexts (e.g. WorkSession delegate).
    nonisolated static var lastKnownReasoningFormat: String? { _lastReasoningFormat }
    nonisolated static var lastKnownThinkInTemplate: Bool { _lastThinkInTemplate }

    /// Force-unload the model from the shared VMLXService singleton.
    /// Use this from UI buttons — it unloads from the actual runtime
    /// regardless of which bridge instance loaded it.
    static func forceUnload() async {
        _vmlxLog("[Bridge] forceUnload() called!")
        await VMLXService.shared.unloadModel()
        await VMLXServiceBridge.shared.resetLoadedModel()
    }

    /// Reset loaded model tracking (called after force unload).
    func resetLoadedModel() {
        currentLoadedModel = nil
    }
}

// MARK: - ChatMessage → VMLXChatMessage

extension ChatMessage {
    /// Convert Osaurus ChatMessage to VMLXRuntime's VMLXChatMessage.
    func toVMLX() -> VMLXChatMessage {
        // Map content parts (multimodal)
        let vmlxParts: [VMLXContentPart]? = contentParts?.map { part in
            switch part {
            case .text(let text):
                return .text(text)
            case .imageUrl(let url, let detail):
                return .imageURL(url: url, detail: detail)
            }
        }

        // Map tool calls
        let vmlxToolCalls: [VMLXToolCall]? = tool_calls?.map { tc in
            VMLXToolCall(
                id: tc.id,
                name: tc.function.name,
                arguments: tc.function.arguments
            )
        }

        return VMLXChatMessage(
            role: role,
            content: content,
            contentParts: vmlxParts,
            toolCalls: vmlxToolCalls,
            toolCallId: tool_call_id
        )
    }
}

// MARK: - VMLXChatMessage → ChatMessage

extension VMLXChatMessage {
    /// Convert VMLXRuntime's VMLXChatMessage back to Osaurus ChatMessage.
    func toOsaurus() -> ChatMessage {
        // Map tool calls back
        let osToolCalls: [ToolCall]? = toolCalls?.map { tc in
            ToolCall(
                id: tc.id,
                type: tc.type,
                function: ToolCallFunction(
                    name: tc.function.name,
                    arguments: tc.function.arguments
                )
            )
        }

        return ChatMessage(
            role: role,
            content: content,
            tool_calls: osToolCalls,
            tool_call_id: toolCallId
        )
    }
}

// MARK: - GenerationParameters → SamplingParams

extension GenerationParameters {
    /// Convert Osaurus GenerationParameters to VMLXRuntime's SamplingParams.
    /// topP: uses per-request override if set, otherwise falls back to globalTopP.
    /// Parser overrides: per-model (from modelOptions) → global (from ServerConfiguration) → auto.
    func toSamplingParams(globalTopP: Float = 0.9, globalToolParser: String? = nil, globalReasoningParser: String? = nil) -> SamplingParams {
        let perModelToolParser = modelOptions["toolParser"]?.stringValue
        let perModelReasoningParser = modelOptions["reasoningParser"]?.stringValue

        let effectiveToolParser = LocalParserOptions.resolveToolOverride(
            perModel: perModelToolParser,
            global: globalToolParser
        )
        let effectiveReasoningParser = LocalParserOptions.resolveReasoningOverride(
            perModel: perModelReasoningParser,
            global: globalReasoningParser
        )

        return SamplingParams(
            maxTokens: maxTokens,
            temperature: temperature ?? 0.7,
            topP: topPOverride ?? globalTopP,
            repetitionPenalty: repetitionPenalty ?? 1.1,
            enableThinking: !isThinkingDisabled,
            reasoningEffort: reasoningEffort,
            toolParserOverride: effectiveToolParser,
            reasoningParserOverride: effectiveReasoningParser
        )
    }

    /// Extract reasoning effort from model options ("low", "medium", "high").
    var reasoningEffort: String? {
        if let val = modelOptions["reasoningEffort"] {
            switch val {
            case .string(let s): return s
            default: return nil
            }
        }
        return nil
    }

    /// Whether thinking is explicitly disabled via modelOptions.
    private var isThinkingDisabled: Bool {
        if let val = modelOptions["disableThinking"] {
            switch val {
            case .bool(let b): return b
            case .string(let s): return s.lowercased() == "true"
            default: return false
            }
        }
        return false
    }

    /// Whether thinking/reasoning mode is enabled via modelOptions.
    /// The UI sets "disableThinking" (inverted) via ModelProfileRegistry.
    var isThinkingEnabled: Bool {
        if let val = modelOptions["disableThinking"] {
            // Inverted: disableThinking=false means thinking IS enabled
            switch val {
            case .bool(let b): return !b
            case .string(let s): return s.lowercased() == "false"
            default: return false
            }
        }
        // If no explicit option, default to false (no thinking)
        return false
    }
}

// MARK: - Tool → VMLXToolDefinition

extension Tool {
    /// Convert Osaurus Tool to VMLXRuntime's VMLXToolDefinition.
    func toVMLX() -> VMLXToolDefinition {
        VMLXToolDefinition(
            name: function.name,
            description: function.description
        )
    }
}

// MARK: - ToolChoiceOption → String

extension ToolChoiceOption {
    /// Convert Osaurus ToolChoiceOption to VMLXRuntime's string-based tool choice.
    func toVMLXString() -> String {
        switch self {
        case .auto:
            return "auto"
        case .none:
            return "none"
        case .function(let fn):
            return fn.function.name
        }
    }
}
