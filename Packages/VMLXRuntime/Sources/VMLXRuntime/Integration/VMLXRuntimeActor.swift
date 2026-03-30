import Foundation
import MLX
import MLXRandom
import MLXNN

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
///
/// Uses native model implementations (Qwen3.5, etc.) for the forward pass.
/// No external model library dependency -- only mlx-swift for the computation backend.
public actor VMLXRuntimeActor {

    public static let shared = VMLXRuntimeActor()

    // MARK: - State

    /// Current loaded model name.
    public private(set) var currentModelName: String?

    /// The VMLX model container (native model + tokenizer + metadata).
    private var modelContainer: VMLXModelContainer?

    /// SSM re-deriver for recovering SSM state when checkpoint is evicted.
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
    /// 1. Calls `ModelLoader.load(from:)` which uses native model registry
    ///    to load weights, tokenizer, and build the correct model architecture
    /// 2. Wraps the result in a `VMLXModelContainer`
    /// 3. Configures the `Scheduler` with the model's properties
    public func loadModel(from path: URL) async throws {
        if modelContainer != nil {
            await unloadModel()
        }

        // 1. Load model using native model registry
        let loadedModel: LoadedModel
        do {
            loadedModel = try await ModelLoader.load(from: path)
        } catch {
            throw VMLXRuntimeError.modelLoadFailed(
                "Failed to load model at \(path.path): \(error.localizedDescription)"
            )
        }

        // 2. Wrap in VMLXModelContainer
        let container = VMLXModelContainer.create(model: loadedModel)

        // 3. Configure the Scheduler
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

        // 4b. Wire SSM re-deriver
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

        if let alias = alias, let container = modelContainer {
            loadedModels.removeValue(forKey: container.name)
            loadedModels[alias] = container
            activeModelName = alias
            currentModelName = alias
        }
    }

    /// Load a model by name (convenience method).
    public func loadModel(name: String) async throws {
        let directURL = URL(fileURLWithPath: name)
        if FileManager.default.fileExists(atPath: directURL.appendingPathComponent("config.json").path) {
            try await loadModel(from: directURL)
            return
        }

        let available = ModelDetector.scanAvailableModels()
        let nameLower = name.lowercased()

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
        for (_, task) in activeGenerations {
            task.cancel()
        }
        activeGenerations.removeAll()
        scheduler.shutdown()

        if let reDeriver = ssmReDeriver {
            await reDeriver.cancelAll()
            await reDeriver.setModel(nil)
        }
        ssmReDeriver = nil

        modelContainer = nil
        currentModelName = nil
    }

    // MARK: - Multi-Model Gateway

    public func resolveModel(_ requestedModel: String?) -> VMLXModelContainer? {
        guard let name = requestedModel else {
            return loadedModels[activeModelName ?? ""]
        }
        return loadedModels[name] ?? loadedModels.values.first {
            $0.name.lowercased().contains(name.lowercased())
        }
    }

    public var loadedModelNames: [String] {
        Array(loadedModels.keys)
    }

    public func unloadModel(name: String) async {
        loadedModels.removeValue(forKey: name)
        if activeModelName == name {
            activeModelName = loadedModels.keys.first
        }
        if currentModelName == name {
            modelContainer = nil
            currentModelName = activeModelName
            if let newActive = activeModelName {
                modelContainer = loadedModels[newActive]
            }
        }
    }

    // MARK: - Power Management

    public func softSleep() async {
        scheduler.cache.clearAll()
        powerState = .softSleep
    }

    public func deepSleep() async {
        await unloadModel()
        loadedModels.removeAll()
        activeModelName = nil
        powerState = .deepSleep
    }

    public func wake() async throws {
        guard powerState != .active else { return }
        if let path = lastLoadedModelPath {
            try await loadModel(from: path)
        } else if let name = lastLoadedModelName {
            try await loadModel(name: name)
        }
        powerState = .active
    }

    public func enableJITWake() {
        if powerState == .deepSleep {
            powerState = .jitWake
        }
    }

    public func enableJIT() {
        jitEnabled = true
    }

    // MARK: - Generation

    /// Generate a streaming response for a chat completion request.
    ///
    /// Uses native VMLXRuntime models for the forward pass.
    /// Implements autoregressive token-by-token generation with:
    /// - Chat template tokenization via swift-transformers
    /// - Greedy/sampling decoding
    /// - StreamAccumulator for tool/reasoning parsing
    /// - VMLXEvent emission for OpenAI-compatible streaming
    public func generateStream(
        request: VMLXChatCompletionRequest
    ) async throws -> AsyncThrowingStream<VMLXEvent, Error> {
        if powerState == .jitWake {
            try await wake()
        }

        guard let container = modelContainer else {
            throw VMLXRuntimeError.noModelLoaded
        }

        let requestId = UUID().uuidString
        let modelName = currentModelName ?? ""

        // Build tool/reasoning parsers
        let toolParser: (any ToolCallParser)? = request.tools != nil
            ? autoDetectToolParser(modelName: modelName) : nil
        let reasoningParser: (any ReasoningParser)? = (request.enableThinking ?? false)
            ? autoDetectReasoningParser(modelName: modelName) : nil

        // Tokenize via chat template
        let samplingParams = request.toSamplingParams()
        let tokens: [Int]
        do {
            tokens = try container.applyChatTemplate(
                messages: request.messages,
                addGenerationPrompt: true
            )
        } catch {
            throw VMLXRuntimeError.tokenizationFailed
        }

        let promptTokenCount = tokens.count
        let maxTokens = samplingParams.maxTokens
        let temperature = samplingParams.temperature
        let topP = samplingParams.topP
        _ = samplingParams.repetitionPenalty  // TODO: implement repetition penalty
        let stopSequences = samplingParams.stop
        let eosTokenIds = container.eosTokenIds
        let stopTokenIds = Set(samplingParams.stopTokenIds)

        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    var accumulator = StreamAccumulator(
                        toolParser: toolParser,
                        reasoningParser: reasoningParser,
                        stopSequences: stopSequences
                    )

                    // Create cache and run generation
                    let cache = container.newCache()
                    var inputTokens = MLXArray(tokens)
                    var generatedTokenCount = 0

                    for _ in 0 ..< maxTokens {
                        try Task.checkCancellation()

                        // Forward pass
                        let logits = container.forward(
                            inputTokens.expandedDimensions(axis: 0),
                            cache: cache
                        )

                        // Sample next token from last position's logits
                        var nextLogits = logits[0, -1]

                        // Apply repetition penalty
                        // (simplified -- full implementation would track context window)

                        // Temperature and sampling
                        let nextToken: Int
                        if temperature == 0 {
                            // Greedy
                            nextToken = nextLogits.argMax().item(Int.self)
                        } else {
                            // Temperature scaling
                            nextLogits = nextLogits / temperature

                            // Top-p (nucleus) sampling
                            if topP < 1.0 {
                                let sortedVals = sorted(nextLogits, axis: -1)
                                let sortedIndices = argSort(nextLogits, axis: -1)
                                let cumProbs = cumsum(softmax(sortedVals, axis: -1), axis: -1)
                                let mask = cumProbs .< (1.0 - topP)
                                nextLogits[sortedIndices] = MLX.where(mask, Float(-1e9), sortedVals)
                            }

                            // Sample from distribution
                            let probs = MLX.softmax(nextLogits, axis: -1)
                            let sampled = MLXRandom.categorical(probs.expandedDimensions(axis: 0))
                            nextToken = sampled.item(Int.self)
                        }

                        generatedTokenCount += 1

                        // Check for EOS
                        if eosTokenIds.contains(nextToken) || stopTokenIds.contains(nextToken) {
                            break
                        }

                        // Decode token to text
                        let text = container.decode([nextToken])

                        // Process through accumulator
                        let events = accumulator.process(text: text, tokenIds: [nextToken])
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

                        // Set up next input
                        inputTokens = MLXArray([Int32(nextToken)])
                    }

                    // Finalize
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
                        promptTokens: promptTokenCount,
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

            Task { @MainActor [requestId] in
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

    public func clearCache() {
        scheduler.cache.clearAll()
    }

    public var cacheStats: CacheCoordinatorStats {
        scheduler.cacheStats
    }

    public var config: SchedulerConfig { scheduler.config }

    public var container: VMLXModelContainer? { modelContainer }

    // MARK: - Private

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
