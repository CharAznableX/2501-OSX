import Foundation
import MLX

/// Events emitted during generation.
public enum VMLXEvent: Sendable {
    case tokens(String)
    case thinking(String)
    case toolInvocation(name: String, argsJSON: String, callId: String)
    case usage(promptTokens: Int, completionTokens: Int, cachedTokens: Int)
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

    /// Whether a model is loaded and ready.
    public var isModelLoaded: Bool { modelContainer != nil }

    /// Scheduler owns request queue, cache coordinator, and batching logic.
    private var scheduler: Scheduler

    /// Active generation tasks, keyed by requestId.
    private var activeGenerations: [String: Task<Void, Never>] = [:]

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
    /// 3. Configures the `Scheduler` with the model's properties (hybrid, stop tokens, TQ)
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

        modelContainer = nil
        currentModelName = nil
    }

    // MARK: - Generation

    /// Generate a streaming response for a chat completion request.
    /// Returns an AsyncThrowingStream of VMLXEvents.
    public func generateStream(
        request: VMLXChatCompletionRequest
    ) throws -> AsyncThrowingStream<VMLXEvent, Error> {
        guard let container = modelContainer else {
            throw VMLXRuntimeError.noModelLoaded
        }

        let requestId = UUID().uuidString
        let samplingParams = request.toSamplingParams()
        let modelName = currentModelName ?? ""

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
            let task = Task { [cacheResult, genPromptLen] in
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

                    // Apply cache result
                    switch cacheResult {
                    case .hit(let cache, let remaining, _):
                        inferenceRequest.promptCache = cache
                        inferenceRequest.remainingTokenIds = remaining
                        inferenceRequest.cachedTokens = promptTokenIds.count - remaining.count

                    case .partialHit(let attentionCache, let remaining, _):
                        // Hybrid model: have KV but not SSM
                        // TODO: Trigger SSM re-derive or fall back to full prefill
                        inferenceRequest.promptCache = attentionCache
                        inferenceRequest.remainingTokenIds = remaining

                    case .miss:
                        inferenceRequest.remainingTokenIds = promptTokenIds
                    }

                    // Set up stream accumulator
                    var accumulator = StreamAccumulator(
                        toolParser: toolParser,
                        reasoningParser: reasoningParser,
                        stopSequences: samplingParams.stop
                    )

                    // ---------------------------------------------------------------
                    // GENERATION LOOP STUB
                    //
                    // This is where ModelForwardPass will plug in. The steps are:
                    //
                    // 1. Prefill: Run uncached tokens through the model forward pass
                    //    - let prefillTokens = inferenceRequest.remainingTokenIds
                    //    - let (logits, newKVCache) = ModelForwardPass.prefill(
                    //          tokens: prefillTokens,
                    //          cache: inferenceRequest.promptCache,
                    //          container: container
                    //      )
                    //    - For hybrid models with genPromptLen > 0, checkpoint SSM
                    //      state at (promptTokenIds.count - genPromptLen)
                    //
                    // 2. Decode loop:
                    //    while generatedCount < samplingParams.maxTokens {
                    //        let tokenId = sampler.sample(logits, params: samplingParams)
                    //        if container.eosTokenIds.contains(tokenId) { break }
                    //        let events = accumulator.process(tokenId: tokenId,
                    //            text: container.decode([tokenId]))
                    //        for event in events { continuation.yield(event) }
                    //        let (nextLogits, _) = ModelForwardPass.step(
                    //            token: tokenId, cache: newKVCache, container: container)
                    //        logits = nextLogits
                    //        generatedCount += 1
                    //    }
                    //
                    // 3. Store cache state:
                    //    scheduler.cache.store(tokens: promptTokenIds + generated,
                    //                         cache: newKVCache)
                    // ---------------------------------------------------------------

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
                        completionTokens: accumulator.generatedTokenIds.count,
                        cachedTokens: inferenceRequest.cachedTokens
                    ))

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
        let stream = try generateStream(request: request)
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
