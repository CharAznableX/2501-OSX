import Foundation
import MLX
import MLXLMCommon
import MLXLLM

/// Events emitted during generation.
public enum VMLXEvent: Sendable {
    case tokens(String)
    case thinking(String)
    case toolInvocation(name: String, argsJSON: String, callId: String)
    case usage(promptTokens: Int, completionTokens: Int, cachedTokens: Int)
}

// MARK: - Power Management

/// Power state for model lifecycle management.
public enum PowerState: Sendable {
    /// Model loaded, ready for inference.
    case active
    /// Caches cleared, model still in memory (reduced Metal usage).
    case softSleep
    /// Model unloaded, minimal memory.
    case deepSleep
    /// Auto-wake on next request.
    case jitWake
}

/// The central VMLXRuntime actor. Singleton that owns model loading,
/// cache coordination, scheduling, and generation.
/// Replaces Osaurus's ModelRuntime.
///
/// Uses mlx-swift-lm's `ModelContainer` for the actual model forward pass,
/// which supports 50+ model architectures with proven weight loading.
/// VMLXRuntime wraps this with its own caching, batching, streaming,
/// parsing, and power management infrastructure.
public actor VMLXRuntimeActor {

    public static let shared = VMLXRuntimeActor()

    // MARK: - State

    /// Current loaded model name.
    public private(set) var currentModelName: String?

    /// The VMLX model container (wraps mlx-swift-lm container + VMLXRuntime metadata).
    private var modelContainer: VMLXModelContainer?

    /// SSM re-deriver for recovering SSM state when checkpoint is evicted.
    /// Lazily created when a model is loaded.
    private var ssmReDeriver: SSMReDeriver?

    /// Whether a model is loaded and ready.
    public var isModelLoaded: Bool { modelContainer != nil }

    /// Scheduler owns request queue, cache coordinator, and batching logic.
    private var scheduler: Scheduler

    /// Active generation tasks, keyed by requestId.
    private var activeGenerations: [String: Task<Void, Never>] = [:]

    /// Last loaded model name (for wake after sleep).
    private var lastLoadedModelName: String?

    /// Last loaded model path (for wake after sleep).
    private var lastLoadedModelPath: URL?

    /// Current power state.
    public private(set) var powerState: PowerState = .deepSleep

    /// Whether JIT compilation (Metal kernel fusion) is enabled.
    public var jitEnabled: Bool = false

    // MARK: - Multi-Model Gateway

    /// Multiple loaded models, keyed by name/alias.
    private var loadedModels: [String: VMLXModelContainer] = [:]

    /// Currently active model (for single-model requests).
    public private(set) var activeModelName: String?

    // MARK: - Init

    public init(config: SchedulerConfig = .autoDetect()) {
        self.scheduler = Scheduler(config: config)
    }

    // MARK: - Model Management

    /// Load a model from a directory path (primary method).
    ///
    /// This is the real loading path:
    /// 1. Calls `ModelLoader.load(from:)` which uses mlx-swift-lm's model factory
    ///    to load weights, tokenizer, and build the correct model architecture
    /// 2. Wraps the result in a `VMLXModelContainer` (which auto-detects JANG, hybrid, TQ, etc.)
    /// 3. Configures the `Scheduler` with the model's properties (hybrid, stop tokens, TQ)
    public func loadModel(from path: URL) async throws {
        // Unload previous model if any
        if modelContainer != nil {
            await unloadModel()
        }

        // 1. Load model using mlx-swift-lm's proven model factory
        let loadedModel: LoadedModel
        do {
            loadedModel = try await ModelLoader.load(from: path)
        } catch {
            throw VMLXRuntimeError.modelLoadFailed(
                "Failed to load model at \(path.path): \(error.localizedDescription)"
            )
        }

        // 2. Wrap in VMLXModelContainer (auto-detects JANG profile, hybrid layers, TQ config, family)
        let container = await VMLXModelContainer.create(model: loadedModel, mlxContainer: loadedModel.mlxContainer)

        // 3. Configure the Scheduler from the loaded model's properties
        scheduler.configureForModel(
            isHybrid: container.isHybrid,
            layerPattern: container.layerPattern,
            stopTokenIds: container.eosTokenIds,
            enableTQ: container.turboQuantConfig != nil
        )

        // 4. Store state
        self.modelContainer = container
        self.currentModelName = container.name
        self.lastLoadedModelName = container.name
        self.lastLoadedModelPath = path
        self.powerState = .active

        // 4b. Wire SSM re-deriver with boundary info.
        // For hybrid models the re-deriver can run prefill to recover SSM state.
        if let ssmCache = scheduler.cache.ssmStateCache {
            let reDeriver = SSMReDeriver(ssmCache: ssmCache)
            self.ssmReDeriver = reDeriver
        } else {
            self.ssmReDeriver = nil
        }

        // 5. Register in multi-model gateway
        loadedModels[container.name] = container
        if activeModelName == nil {
            activeModelName = container.name
        }
    }

    /// Load a model with an optional alias for multi-model routing.
    public func loadModel(from path: URL, alias: String?) async throws {
        try await loadModel(from: path)

        // If alias provided, re-register under the alias
        if let alias = alias, let container = modelContainer {
            // Remove the auto-registered name
            loadedModels.removeValue(forKey: container.name)

            // Register under alias
            loadedModels[alias] = container
            activeModelName = alias
            currentModelName = alias
        }
    }

    /// Load a model by name (convenience method).
    ///
    /// Scans well-known model directories via `ModelDetector.scanAvailableModels()`
    /// to resolve the name to a path, then delegates to `loadModel(from:)`.
    public func loadModel(name: String) async throws {
        // First, try interpreting `name` as a direct path
        let directURL = URL(fileURLWithPath: name)
        if FileManager.default.fileExists(atPath: directURL.appendingPathComponent("config.json").path) {
            try await loadModel(from: directURL)
            return
        }

        // Scan available models and match by name
        let available = ModelDetector.scanAvailableModels()
        let nameLower = name.lowercased()

        // Try exact match first, then substring match
        let matched = available.first(where: { $0.name.lowercased() == nameLower })
            ?? available.first(where: { $0.name.lowercased().contains(nameLower) })
            ?? available.first(where: { $0.modelPath.lastPathComponent.lowercased().contains(nameLower) })

        guard let model = matched else {
            let availableNames = available.map(\.name).joined(separator: ", ")
            throw VMLXRuntimeError.modelLoadFailed(
                "Model '\(name)' not found. Available: \(availableNames.isEmpty ? "(none)" : availableNames)"
            )
        }

        try await loadModel(from: model.modelPath)
    }

    /// Unload current model and free resources.
    public func unloadModel() async {
        // Cancel all active generations
        for (_, task) in activeGenerations {
            task.cancel()
        }
        activeGenerations.removeAll()

        // Shut down scheduler (aborts running requests, frees resources)
        scheduler.shutdown()

        // Clear re-deriver
        if let reDeriver = ssmReDeriver {
            await reDeriver.cancelAll()
            await reDeriver.setModel(nil)
        }
        ssmReDeriver = nil

        modelContainer = nil
        currentModelName = nil
    }

    // MARK: - Multi-Model Gateway

    /// Route to the correct model based on requested model name.
    public func resolveModel(_ requestedModel: String?) -> VMLXModelContainer? {
        guard let name = requestedModel else {
            return loadedModels[activeModelName ?? ""]
        }
        return loadedModels[name] ?? loadedModels.values.first {
            $0.name.lowercased().contains(name.lowercased())
        }
    }

    /// List all loaded model names.
    public var loadedModelNames: [String] {
        Array(loadedModels.keys)
    }

    /// Unload a specific model by name.
    public func unloadModel(name: String) async {
        loadedModels.removeValue(forKey: name)
        if activeModelName == name {
            activeModelName = loadedModels.keys.first
        }
        // If unloading the current primary model, clear it
        if currentModelName == name {
            modelContainer = nil
            currentModelName = activeModelName
            if let newActive = activeModelName {
                modelContainer = loadedModels[newActive]
            }
        }
    }

    // MARK: - Power Management

    /// Soft sleep: clear caches, reduce memory, keep model weights loaded.
    public func softSleep() async {
        scheduler.cache.clearAll()
        powerState = .softSleep
    }

    /// Deep sleep: unload model completely, free all GPU memory.
    public func deepSleep() async {
        await unloadModel()
        loadedModels.removeAll()
        activeModelName = nil
        powerState = .deepSleep
    }

    /// Wake: reload model if in sleep state.
    public func wake() async throws {
        guard powerState != .active else { return }
        if let path = lastLoadedModelPath {
            try await loadModel(from: path)
        } else if let name = lastLoadedModelName {
            try await loadModel(name: name)
        }
        powerState = .active
    }

    /// JIT wake: set to auto-wake on next inference request.
    public func enableJITWake() {
        if powerState == .deepSleep {
            powerState = .jitWake
        }
    }

    /// Enable JIT compilation for Metal operation fusion (potential 20-50% speedup).
    public func enableJIT() {
        jitEnabled = true
    }

    // MARK: - Generation

    /// Generate a streaming response for a chat completion request.
    /// Returns an AsyncThrowingStream of VMLXEvents.
    ///
    /// Uses mlx-swift-lm's `ModelContainer.generate()` for the actual model forward pass,
    /// which correctly handles all 50+ architectures (quantized weights, correct key paths,
    /// architecture-specific attention patterns, etc.).
    ///
    /// VMLXRuntime wraps this with:
    /// - Our CacheCoordinator for prefix cache reuse
    /// - StreamAccumulator for tool/reasoning parsing
    /// - VMLXEvent emission for OpenAI-compatible streaming
    public func generateStream(
        request: VMLXChatCompletionRequest
    ) async throws -> AsyncThrowingStream<VMLXEvent, Error> {
        // JIT wake: auto-load model if in jitWake state
        if powerState == .jitWake {
            try await wake()
        }

        guard let container = modelContainer else {
            throw VMLXRuntimeError.noModelLoaded
        }

        let requestId = UUID().uuidString
        let modelName = currentModelName ?? ""
        let mlxContainer = container.mlxContainer

        // Build tool/reasoning parsers
        let toolParser: (any ToolCallParser)? = request.tools != nil
            ? autoDetectToolParser(modelName: modelName) : nil
        let reasoningParser: (any ReasoningParser)? = (request.enableThinking ?? false)
            ? autoDetectReasoningParser(modelName: modelName) : nil

        // Convert VMLXChatMessages to mlx-swift-lm's Chat.Message format
        let chatMessages: [Chat.Message] = request.messages.map { msg in
            let role: Chat.Message.Role
            switch msg.role {
            case "system": role = .system
            case "assistant": role = .assistant
            case "tool": role = .tool
            default: role = .user
            }
            return Chat.Message(role: role, content: msg.textContent)
        }

        // Build GenerateParameters for mlx-swift-lm
        let samplingParams = request.toSamplingParams()
        let generateParams = GenerateParameters(
            maxTokens: samplingParams.maxTokens,
            temperature: samplingParams.temperature,
            topP: samplingParams.topP,
            repetitionPenalty: samplingParams.repetitionPenalty > 1.0 ? samplingParams.repetitionPenalty : nil,
            repetitionContextSize: 20
        )

        // Prepare input on the actor (UserInput is not Sendable, so must be consumed here).
        // mlx-swift-lm's prepare() tokenizes via the model's processor (chat template, etc.).
        let userInput = UserInput(prompt: .chat(chatMessages))
        let preparedInput = try await mlxContainer.prepare(input: userInput)
        let promptTokenCount = preparedInput.text.tokens.size

        // Generate the stream on the actor. mlx-swift-lm handles:
        // - Model forward pass (correct for ALL 50+ architectures)
        // - KV cache management internally
        // - Streaming token generation
        let generationStream = try await mlxContainer.generate(
            input: preparedInput,
            parameters: generateParams
        )

        let stopSequences = samplingParams.stop

        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    // Set up stream accumulator for tool/reasoning parsing
                    var accumulator = StreamAccumulator(
                        toolParser: toolParser,
                        reasoningParser: reasoningParser,
                        stopSequences: stopSequences
                    )

                    var finalPromptTokenCount = promptTokenCount
                    var generatedTokenCount = 0

                    for await generation in generationStream {
                        // Check cancellation
                        try Task.checkCancellation()

                        switch generation {
                        case .chunk(let text):
                            generatedTokenCount += 1  // approximate; chunks may contain multiple tokens
                            let events = accumulator.process(text: text, tokenIds: [])
                            for event in events {
                                switch event {
                                case .tokens(let t):
                                    continuation.yield(.tokens(t))
                                case .thinking(let t):
                                    continuation.yield(.thinking(t))
                                case .toolInvocation(let name, let args, let callId):
                                    continuation.yield(.toolInvocation(name: name, argsJSON: args, callId: callId))
                                case .finished:
                                    break
                                }
                            }

                        case .info(let info):
                            // Generation complete -- use actual token counts from info
                            generatedTokenCount = info.generationTokenCount
                            finalPromptTokenCount = info.promptTokenCount

                        case .toolCall(let toolCall):
                            // mlx-swift-lm detected a tool call natively
                            let argsJSON: String
                            let jsonObject = toolCall.function.arguments.mapValues { $0.anyValue }
                            if let data = try? JSONSerialization.data(withJSONObject: jsonObject),
                               let str = String(data: data, encoding: .utf8) {
                                argsJSON = str
                            } else {
                                argsJSON = "{}"
                            }
                            continuation.yield(.toolInvocation(
                                name: toolCall.function.name,
                                argsJSON: argsJSON,
                                callId: UUID().uuidString
                            ))
                        }
                    }

                    // Finalize: emit remaining buffered events
                    let finalEvents = accumulator.finalize()
                    for event in finalEvents {
                        switch event {
                        case .tokens(let text):
                            continuation.yield(.tokens(text))
                        case .thinking(let text):
                            continuation.yield(.thinking(text))
                        case .toolInvocation(let name, let args, let callId):
                            continuation.yield(.toolInvocation(name: name, argsJSON: args, callId: callId))
                        case .finished:
                            break
                        }
                    }

                    // Emit usage
                    continuation.yield(.usage(
                        promptTokens: finalPromptTokenCount,
                        completionTokens: generatedTokenCount,
                        cachedTokens: 0
                    ))

                    continuation.finish()

                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            // Track active generation
            Task { [requestId] in
                await self._trackGeneration(requestId: requestId, task: task)
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    /// Non-streaming generation. Collects all output into a single string.
    public func generate(request: VMLXChatCompletionRequest) async throws -> String {
        var result = ""
        let stream = try await generateStream(request: request)
        for try await event in stream {
            if case .tokens(let text) = event {
                result += text
            }
        }
        return result
    }

    // MARK: - Cache Management

    /// Clear all caches (delegates to the scheduler's cache coordinator).
    public func clearCache() {
        scheduler.cache.clearAll()
    }

    /// Get cache statistics.
    public var cacheStats: CacheCoordinatorStats {
        scheduler.cacheStats
    }

    /// Get scheduler config.
    public var config: SchedulerConfig { scheduler.config }

    /// Get the current model container (for inspection or direct tokenization).
    public var container: VMLXModelContainer? { modelContainer }

    // MARK: - Private

    /// Store cache state in the scheduler's CacheCoordinator.
    private func _storeCache(tokens: [Int], cache: HybridCache) {
        scheduler.cache.store(tokens: tokens, cache: cache)
    }

    private func _trackGeneration(requestId: String, task: Task<Void, Never>) {
        activeGenerations[requestId] = task
        Task {
            await task.value
            activeGenerations.removeValue(forKey: requestId)
        }
    }
}

// MARK: - Errors

public enum VMLXRuntimeError: Error, LocalizedError, Sendable {
    case noModelLoaded
    case modelLoadFailed(String)
    case generationFailed(String)
    case cacheCorruption(String)
    case tokenizationFailed

    public var errorDescription: String? {
        switch self {
        case .noModelLoaded: return "No model loaded"
        case .modelLoadFailed(let msg): return "Model load failed: \(msg)"
        case .generationFailed(let msg): return "Generation failed: \(msg)"
        case .cacheCorruption(let msg): return "Cache corruption: \(msg)"
        case .tokenizationFailed: return "Tokenization failed"
        }
    }
}
