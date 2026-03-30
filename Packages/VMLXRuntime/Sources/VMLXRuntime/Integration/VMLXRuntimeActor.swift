import Foundation
import MLX

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
public actor VMLXRuntimeActor {

    public static let shared = VMLXRuntimeActor()

    // MARK: - State

    /// Current loaded model name.
    public private(set) var currentModelName: String?

    /// The loaded model container (weights + tokenizer + config + runtime config).
    private var modelContainer: ModelContainer?

    /// The transformer forward pass wrapper (created after model load).
    private var forwardPass: TransformerModelForwardPass?

    /// SSM re-deriver for recovering SSM state when checkpoint is evicted.
    /// Lazily created when a model is loaded, wired with the model's forward pass.
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
    private var loadedModels: [String: ModelContainer] = [:]

    /// Forward pass wrappers for each loaded model.
    private var loadedForwardPasses: [String: TransformerModelForwardPass] = [:]

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
    /// 1. Calls `ModelLoader.load(from:)` to read safetensors weights, tokenizer, and config
    /// 2. Wraps the result in a `ModelContainer` (which auto-detects JANG, hybrid, TQ, etc.)
    /// 3. Creates `TransformerModel` from config and loads weights
    /// 4. Configures the `Scheduler` with the model's properties (hybrid, stop tokens, TQ)
    public func loadModel(from path: URL) async throws {
        // Unload previous model if any
        if modelContainer != nil {
            await unloadModel()
        }

        // 1. Load weights, tokenizer, and config from disk
        let loadedModel: LoadedModel
        do {
            loadedModel = try await ModelLoader.load(from: path)
        } catch {
            throw VMLXRuntimeError.modelLoadFailed(
                "Failed to load model at \(path.path): \(error.localizedDescription)"
            )
        }

        // 2. Wrap in ModelContainer (auto-detects JANG profile, hybrid layers, TQ config, family)
        let container = ModelContainer(model: loadedModel)

        // 3. Create TransformerModel and load weights
        let transformer = TransformerModel.from(loaded: loadedModel)
        let fwdPass = TransformerModelForwardPass(model: transformer)

        // 4. Configure the Scheduler from the loaded model's properties
        scheduler.configureForModel(
            isHybrid: container.isHybrid,
            layerPattern: container.layerPattern,
            stopTokenIds: container.eosTokenIds,
            enableTQ: container.turboQuantConfig != nil
        )

        // 5. Store state
        self.modelContainer = container
        self.forwardPass = fwdPass
        self.currentModelName = container.name
        self.lastLoadedModelName = container.name
        self.lastLoadedModelPath = path
        self.powerState = .active

        // 5b. Wire SSM re-deriver with the model's forward pass.
        // For hybrid models the re-deriver can run prefill to recover SSM state.
        // For pure-attention models the re-deriver still records boundary info.
        if let ssmCache = scheduler.cache.ssmStateCache {
            let reDeriver = SSMReDeriver(ssmCache: ssmCache, model: fwdPass)
            self.ssmReDeriver = reDeriver
        } else {
            self.ssmReDeriver = nil
        }

        // 6. Register in multi-model gateway
        loadedModels[container.name] = container
        loadedForwardPasses[container.name] = fwdPass
        if activeModelName == nil {
            activeModelName = container.name
        }
    }

    /// Load a model with an optional alias for multi-model routing.
    public func loadModel(from path: URL, alias: String?) async throws {
        try await loadModel(from: path)

        // If alias provided, re-register under the alias
        if let alias = alias, let container = modelContainer, let fwdPass = forwardPass {
            // Remove the auto-registered name
            loadedModels.removeValue(forKey: container.name)
            loadedForwardPasses.removeValue(forKey: container.name)

            // Register under alias
            loadedModels[alias] = container
            loadedForwardPasses[alias] = fwdPass
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

        // Clear forward pass and re-deriver
        forwardPass = nil
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
    public func resolveModel(_ requestedModel: String?) -> ModelContainer? {
        guard let name = requestedModel else {
            return loadedModels[activeModelName ?? ""]
        }
        return loadedModels[name] ?? loadedModels.values.first {
            $0.name.lowercased().contains(name.lowercased())
        }
    }

    /// Resolve the forward pass for a requested model name.
    private func resolveForwardPass(_ requestedModel: String?) -> TransformerModelForwardPass? {
        guard let name = requestedModel else {
            return loadedForwardPasses[activeModelName ?? ""]
        }
        if let fwdPass = loadedForwardPasses[name] {
            return fwdPass
        }
        // Fuzzy match
        for (key, fwdPass) in loadedForwardPasses {
            if key.lowercased().contains(name.lowercased()) {
                return fwdPass
            }
        }
        return nil
    }

    /// List all loaded model names.
    public var loadedModelNames: [String] {
        Array(loadedModels.keys)
    }

    /// Unload a specific model by name.
    public func unloadModel(name: String) async {
        loadedModels.removeValue(forKey: name)
        loadedForwardPasses.removeValue(forKey: name)
        if activeModelName == name {
            activeModelName = loadedModels.keys.first
        }
        // If unloading the current primary model, clear it
        if currentModelName == name {
            modelContainer = nil
            forwardPass = nil
            currentModelName = activeModelName
            if let newActive = activeModelName {
                modelContainer = loadedModels[newActive]
                forwardPass = loadedForwardPasses[newActive]
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
        loadedForwardPasses.removeAll()
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

        guard let fwdPass = forwardPass else {
            throw VMLXRuntimeError.generationFailed("TransformerModelForwardPass not initialized")
        }

        let requestId = UUID().uuidString
        let samplingParams = request.toSamplingParams()
        let modelName = currentModelName ?? ""
        let stopTokenIds = container.eosTokenIds

        // Tokenize messages using the loaded model's chat template
        let promptTokenIds: [Int]
        do {
            promptTokenIds = try container.applyChatTemplate(messages: request.messages)
        } catch {
            throw VMLXRuntimeError.tokenizationFailed
        }

        // For hybrid thinking models, compute how many tokens the generation prompt
        // (e.g., <think>) adds so SSM checkpointing knows the stable boundary
        let genPromptLen: Int
        if request.enableThinking == true, container.isHybrid {
            genPromptLen = container.computeGenPromptLen(messages: request.messages)
        } else {
            genPromptLen = 0
        }

        // Cache lookup with real token IDs
        let cacheResult = scheduler.cache.fetch(tokens: promptTokenIds)

        // Build tool/reasoning parsers
        let toolParser: (any ToolCallParser)? = request.tools != nil
            ? autoDetectToolParser(modelName: modelName) : nil
        let reasoningParser: (any ReasoningParser)? = (request.enableThinking ?? false)
            ? autoDetectReasoningParser(modelName: modelName) : nil

        return AsyncThrowingStream { continuation in
            let task = Task { [cacheResult, genPromptLen, fwdPass] in
                // genPromptLen used by forward pass to checkpoint SSM state
                _ = genPromptLen
                do {
                    var inferenceRequest = InferenceRequest(
                        requestId: requestId,
                        promptTokenIds: promptTokenIds,
                        samplingParams: samplingParams,
                        enableThinking: request.enableThinking ?? false,
                        reasoningEffort: request.reasoningEffort ?? "medium",
                        isMultimodal: request.isMultimodal
                    )

                    // Apply cache result and populate model KV caches from CacheCoordinator
                    switch cacheResult {
                    case .hit(let cache, let remaining, _):
                        fwdPass.loadCache(cache)
                        inferenceRequest.promptCache = cache
                        inferenceRequest.remainingTokenIds = remaining
                        inferenceRequest.cachedTokens = promptTokenIds.count - remaining.count

                    case .partialHit(let attentionCache, let remaining, _):
                        // Hybrid model: have KV but not SSM
                        fwdPass.loadCache(attentionCache)
                        inferenceRequest.promptCache = attentionCache
                        inferenceRequest.remainingTokenIds = remaining

                    case .miss:
                        fwdPass.resetCaches()
                        inferenceRequest.remainingTokenIds = promptTokenIds
                    }

                    // Set up stream accumulator
                    var accumulator = StreamAccumulator(
                        toolParser: toolParser,
                        reasoningParser: reasoningParser,
                        stopSequences: samplingParams.stop
                    )

                    // ---------------------------------------------------------------
                    // GENERATION LOOP
                    // ---------------------------------------------------------------

                    let uncachedTokens = inferenceRequest.remainingTokenIds ?? promptTokenIds
                    // ModelForwardPass protocol requires [MLXArray] but TransformerModelForwardPass
                    // manages structured KVCache objects internally and ignores this parameter.
                    var cacheArrays: [MLXArray] = []
                    var generatedIds: [Int] = []
                    let maxTokens = samplingParams.maxTokens

                    // Phase 1: Prefill — run uncached tokens through the model
                    let inputArray = MLXArray(uncachedTokens.map { Int32($0) })

                    // Build causal mask for prefill
                    let prefillMask: MLXArray?
                    if uncachedTokens.count > 1 {
                        let offset = promptTokenIds.count - uncachedTokens.count
                        prefillMask = TransformerModel.createCausalMask(
                            seqLen: uncachedTokens.count,
                            offset: offset,
                            dtype: .float16
                        )
                    } else {
                        prefillMask = nil
                    }

                    let prefillLogits = try await fwdPass.prefill(
                        inputIds: inputArray,
                        cache: &cacheArrays,
                        mask: prefillMask
                    )

                    // Phase 2: Sample first token from prefill logits
                    let firstToken = Sampler.sample(
                        logits: prefillLogits,
                        params: samplingParams
                    )
                    generatedIds.append(firstToken)

                    // Emit first token (unless it's a stop token)
                    var stopped = false
                    if stopTokenIds.contains(firstToken) {
                        stopped = true
                    } else {
                        let text = container.decode([firstToken])
                        let events = accumulator.process(text: text, tokenIds: [firstToken])
                        for event in events {
                            switch event {
                            case .tokens(let t):
                                continuation.yield(.tokens(t))
                            case .thinking(let t):
                                continuation.yield(.thinking(t))
                            case .toolInvocation(let name, let args, let callId):
                                continuation.yield(.toolInvocation(name: name, argsJSON: args, callId: callId))
                            case .finished:
                                stopped = true
                            }
                        }
                    }

                    // Phase 3: Decode loop
                    if !stopped {
                        var lastToken = firstToken
                        for _ in 1..<maxTokens {
                            // Check cancellation
                            try Task.checkCancellation()

                            // Decode next token
                            let logits = try await fwdPass.decode(
                                tokenId: lastToken,
                                cache: &cacheArrays
                            )

                            // Sample
                            let tokenId = Sampler.sample(
                                logits: logits,
                                params: samplingParams,
                                previousTokens: generatedIds
                            )
                            generatedIds.append(tokenId)

                            // Check stop token
                            if stopTokenIds.contains(tokenId) {
                                break
                            }

                            // Decode and emit
                            let text = container.decode([tokenId])
                            let events = accumulator.process(text: text, tokenIds: [tokenId])
                            for event in events {
                                switch event {
                                case .tokens(let t):
                                    continuation.yield(.tokens(t))
                                case .thinking(let t):
                                    continuation.yield(.thinking(t))
                                case .toolInvocation(let name, let args, let callId):
                                    continuation.yield(.toolInvocation(name: name, argsJSON: args, callId: callId))
                                case .finished:
                                    stopped = true
                                }
                            }

                            if stopped { break }
                            lastToken = tokenId
                        }
                    }

                    // Phase 4: Export model KV cache and store in CacheCoordinator
                    // for future prefix cache reuse across requests.
                    let allTokens = promptTokenIds + generatedIds
                    let finalCache = fwdPass.exportCache()
                    finalCache.materialized()
                    await self._storeCache(tokens: allTokens, cache: finalCache)

                    // Finalize and emit remaining events
                    let events = accumulator.finalize()
                    for event in events {
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
                        promptTokens: promptTokenIds.count,
                        completionTokens: generatedIds.count,
                        cachedTokens: inferenceRequest.cachedTokens
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
    public var container: ModelContainer? { modelContainer }

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
