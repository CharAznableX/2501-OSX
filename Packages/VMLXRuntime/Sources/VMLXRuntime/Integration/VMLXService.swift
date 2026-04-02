import Foundation

/// Protocol matching Osaurus's ModelService interface.
/// VMLXService conforms to this; the OsaurusCore integration layer bridges to the actual protocol.
public protocol VMLXModelService: Sendable {
    var serviceId: String { get }
    func isAvailable() -> Bool
    func handles(requestedModel: String?) -> Bool

    func generateOneShot(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        requestedModel: String?
    ) async throws -> String

    func streamDeltas(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        requestedModel: String?,
        stopSequences: [String]
    ) async throws -> AsyncThrowingStream<String, Error>
}

/// Protocol for tool-capable services (extends VMLXModelService).
public protocol VMLXToolCapableService: VMLXModelService {
    func respondWithTools(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        stopSequences: [String],
        tools: [VMLXToolDefinition],
        toolChoice: String?,
        requestedModel: String?
    ) async throws -> String

    func streamWithTools(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        stopSequences: [String],
        tools: [VMLXToolDefinition],
        toolChoice: String?,
        requestedModel: String?
    ) async throws -> AsyncThrowingStream<String, Error>
}

/// The main VMLXService — drop-in replacement for Osaurus's MLXService.
/// Delegates to VMLXRuntimeActor for all inference operations.
public actor VMLXService: VMLXToolCapableService {

    public static let shared = VMLXService()

    public nonisolated var serviceId: String { "vmlx" }

    private let runtime: VMLXRuntimeActor

    public init(runtime: VMLXRuntimeActor = .shared) {
        self.runtime = runtime
    }

    private nonisolated func bridgeEventStream(
        _ eventStream: AsyncThrowingStream<VMLXEvent, Error>
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let producerTask = Task {
                var isInsideThinking = false

                func closeThinkingIfNeeded() {
                    guard isInsideThinking else { return }
                    continuation.yield("</think>")
                    isInsideThinking = false
                }

                do {
                    for try await event in eventStream {
                        switch event {
                        case .tokens(let text):
                            closeThinkingIfNeeded()
                            if !text.isEmpty {
                                continuation.yield(text)
                            }
                        case .thinking(let text):
                            guard !text.isEmpty else { continue }
                            if !isInsideThinking {
                                continuation.yield("<think>")
                                isInsideThinking = true
                            }
                            continuation.yield(text)
                        case .toolInvocation(let name, let args, _):
                            closeThinkingIfNeeded()
                            continuation.yield("\u{FFFE}tool:" + name)
                            continuation.yield("\u{FFFE}args:" + args)
                        case .usage(let prompt, let completion, let cached,
                                    let ttft, let ppTPS, let decTPS, let detail, let cacheBytes):
                            closeThinkingIfNeeded()
                            let statsJSON = "{\"p\":\(prompt),\"c\":\(completion),\"k\":\(cached),"
                                + "\"ttft\":\(String(format:"%.3f",ttft)),"
                                + "\"pp\":\(String(format:"%.1f",ppTPS)),"
                                + "\"tg\":\(String(format:"%.1f",decTPS)),"
                                + "\"cb\":\(cacheBytes),"
                                + "\"d\":\"\(detail ?? "miss")\"}"
                            continuation.yield("\u{FFFE}stats:" + statsJSON)
                        }
                    }
                    closeThinkingIfNeeded()
                    continuation.finish()
                } catch {
                    if isInsideThinking {
                        continuation.yield("</think>")
                    }
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                producerTask.cancel()
            }
        }
    }

    // MARK: - VMLXModelService

    public nonisolated func isAvailable() -> Bool {
        // Available if the runtime exists (model may or may not be loaded)
        true
    }

    public nonisolated func handles(requestedModel: String?) -> Bool {
        guard let model = requestedModel else { return true }
        let lower = model.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if lower.isEmpty || lower == "local" || lower == "default" || lower == "vmlx" {
            return true
        }
        // Reject obvious remote API models (provider/ prefix)
        let remoteProviders = ["openai/", "anthropic/", "google/", "venice-ai/",
                               "groq/", "together/", "fireworks/", "perplexity/",
                               "deepinfra/", "anyscale/"]
        if remoteProviders.contains(where: { lower.hasPrefix($0) }) {
            return false
        }
        // Accept everything else — loader determines if architecture is supported
        return true
    }

    public func generateOneShot(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        requestedModel: String?
    ) async throws -> String {
        let request = VMLXChatCompletionRequest(
            messages: messages,
            model: requestedModel,
            temperature: params.temperature,
            maxTokens: params.maxTokens,
            topP: params.topP,
            repetitionPenalty: params.repetitionPenalty,
            stop: params.stop,
            stream: false,
            enableThinking: params.enableThinking,
            toolParserOverride: params.toolParserOverride,
            reasoningParserOverride: params.reasoningParserOverride
        )
        return try await runtime.generate(request: request)
    }

    public func streamDeltas(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        requestedModel: String?,
        stopSequences: [String]
    ) async throws -> AsyncThrowingStream<String, Error> {
        let request = VMLXChatCompletionRequest(
            messages: messages,
            model: requestedModel,
            temperature: params.temperature,
            maxTokens: params.maxTokens,
            topP: params.topP,
            repetitionPenalty: params.repetitionPenalty,
            stop: stopSequences,
            stream: true,
            enableThinking: params.enableThinking,
            reasoningEffort: params.reasoningEffort,
            toolParserOverride: params.toolParserOverride,
            reasoningParserOverride: params.reasoningParserOverride
        )

        let eventStream = try await runtime.generateStream(request: request)
        return bridgeEventStream(eventStream)
    }

    // MARK: - VMLXToolCapableService

    public func respondWithTools(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        stopSequences: [String],
        tools: [VMLXToolDefinition],
        toolChoice: String?,
        requestedModel: String?
    ) async throws -> String {
        let request = VMLXChatCompletionRequest(
            messages: messages,
            model: requestedModel,
            temperature: params.temperature,
            maxTokens: params.maxTokens,
            topP: params.topP,
            repetitionPenalty: params.repetitionPenalty,
            stop: stopSequences,
            stream: false,
            tools: tools,
            toolChoice: toolChoice,
            enableThinking: params.enableThinking,
            toolParserOverride: params.toolParserOverride,
            reasoningParserOverride: params.reasoningParserOverride
        )
        return try await runtime.generate(request: request)
    }

    public func streamWithTools(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        stopSequences: [String],
        tools: [VMLXToolDefinition],
        toolChoice: String?,
        requestedModel: String?
    ) async throws -> AsyncThrowingStream<String, Error> {
        let request = VMLXChatCompletionRequest(
            messages: messages,
            model: requestedModel,
            temperature: params.temperature,
            maxTokens: params.maxTokens,
            topP: params.topP,
            repetitionPenalty: params.repetitionPenalty,
            stop: stopSequences,
            stream: true,
            tools: tools,
            toolChoice: toolChoice,
            enableThinking: params.enableThinking,
            toolParserOverride: params.toolParserOverride,
            reasoningParserOverride: params.reasoningParserOverride
        )

        let eventStream = try await runtime.generateStream(request: request)
        return bridgeEventStream(eventStream)
    }

    // MARK: - Runtime Configuration Passthrough

    /// Forward user-facing runtime settings to VMLXRuntimeActor.
    public func applyUserConfig(
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        maxContextLength: Int? = nil,
        prefillStepSize: Int? = nil,
        enableDiskCache: Bool = false,
        diskCacheDir: URL? = nil,
        enableTurboQuant: Bool = false,
        cacheMemoryPercent: Float? = nil,
        usePagedCache: Bool? = nil
    ) async {
        await runtime.applyUserConfig(
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            maxContextLength: maxContextLength,
            prefillStepSize: prefillStepSize,
            enableDiskCache: enableDiskCache,
            diskCacheDir: diskCacheDir,
            enableTurboQuant: enableTurboQuant,
            cacheMemoryPercent: cacheMemoryPercent,
            usePagedCache: usePagedCache
        )
    }

    // MARK: - Model Management Passthrough

    /// Load a model from a directory path.
    public func loadModel(from path: URL) async throws {
        try await runtime.loadModel(from: path)
    }

    /// Load a model by name (scans well-known directories to resolve).
    public func loadModel(name: String) async throws {
        try await runtime.loadModel(name: name)
    }

    public func unloadModel() async {
        await runtime.unloadModel()
    }

    public var currentModelName: String? {
        get async { await runtime.currentModelName }
    }

    public var isModelLoaded: Bool {
        get async { await runtime.isModelLoaded }
    }
}
