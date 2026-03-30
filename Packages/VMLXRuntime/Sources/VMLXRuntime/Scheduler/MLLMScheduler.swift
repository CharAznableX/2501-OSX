import Foundation
import MLX

/// Configuration for the MLLM (Multimodal LLM) scheduler.
public struct MLLMSchedulerConfig: Sendable {
    /// Base scheduler config.
    public let schedulerConfig: SchedulerConfig

    /// Maximum images per request.
    public let maxImagesPerRequest: Int

    /// Maximum image size in bytes (50 MB default).
    public let maxImageSizeBytes: Int

    /// Maximum video frames to extract.
    public let maxVideoFrames: Int

    /// Whether to cache vision embeddings.
    public let enableVisionEmbeddingCache: Bool

    /// Maximum entries in vision embedding cache.
    public let visionCacheMaxEntries: Int

    public init(
        schedulerConfig: SchedulerConfig = .autoDetect(),
        maxImagesPerRequest: Int = 10,
        maxImageSizeBytes: Int = 50 * 1024 * 1024,
        maxVideoFrames: Int = 64,
        enableVisionEmbeddingCache: Bool = true,
        visionCacheMaxEntries: Int = 100
    ) {
        self.schedulerConfig = schedulerConfig
        self.maxImagesPerRequest = maxImagesPerRequest
        self.maxImageSizeBytes = maxImageSizeBytes
        self.maxVideoFrames = maxVideoFrames
        self.enableVisionEmbeddingCache = enableVisionEmbeddingCache
        self.visionCacheMaxEntries = visionCacheMaxEntries
    }
}

/// Vision-aware scheduler for multimodal language models.
/// Extends the base Scheduler with image/video preprocessing,
/// vision embedding caching, and gen_prompt_len stripping for thinking models.
public final class MLLMScheduler: @unchecked Sendable {

    /// Base scheduler (handles text request lifecycle).
    public let scheduler: Scheduler

    /// Vision processor for image preprocessing.
    public let visionProcessor: VisionProcessor

    /// Vision embedding cache (avoids re-encoding identical images).
    public let embeddingCache: VisionEmbeddingCache?

    /// MLLM-specific config.
    public let config: MLLMSchedulerConfig

    // Stats
    public private(set) var imagesProcessed: Int = 0
    public private(set) var embeddingCacheHits: Int = 0

    public init(config: MLLMSchedulerConfig = MLLMSchedulerConfig()) {
        self.config = config
        self.scheduler = Scheduler(config: config.schedulerConfig)
        self.visionProcessor = VisionProcessor()

        if config.enableVisionEmbeddingCache {
            self.embeddingCache = VisionEmbeddingCache(
                maxEntries: config.visionCacheMaxEntries
            )
        } else {
            self.embeddingCache = nil
        }
    }

    // MARK: - Multimodal Request Handling

    /// Add a multimodal request. Preprocesses images before scheduling.
    public func addMultimodalRequest(
        _ request: InferenceRequest,
        imageURLs: [String],
        detail: ImageDetail = .auto
    ) throws {
        var req = request

        // Process images
        var processedImages: [ProcessedImage] = []

        for url in imageURLs.prefix(config.maxImagesPerRequest) {
            // Check embedding cache first
            if let cache = embeddingCache {
                let dataHash: String
                if let data = _extractImageData(from: url) {
                    dataHash = VisionEmbeddingCache.hashData(data)
                } else {
                    dataHash = url  // Use URL as hash if data not available
                }

                if let _ = cache.fetch(dataHash: dataHash) {
                    embeddingCacheHits += 1
                    continue  // Embedding already cached
                }
            }

            // Process image
            let processed = try visionProcessor.processImageURL(url, detail: detail)
            processedImages.append(processed)
            imagesProcessed += 1
        }

        // Attach pixel values to request
        if let firstImage = processedImages.first {
            req.pixelValues = firstImage.pixelValues
            req.isMultimodal = true

            // Store grid dimensions if available
            if let grid = firstImage.gridTHW {
                req.imageGridTHW = [grid.0, grid.1, grid.2]
            }
        }

        scheduler.addRequest(req)
    }

    /// Strip gen_prompt_len from token sequence for cache keying.
    /// Thinking models inject tokens like <think>\n that contaminate cache keys.
    public static func stripGenPrompt(tokens: [Int], genPromptLen: Int) -> [Int] {
        guard genPromptLen > 0, genPromptLen < tokens.count else { return tokens }
        return Array(tokens.dropLast(genPromptLen))
    }

    /// Compute gen_prompt_len: the number of tokens added by the chat template
    /// that are not part of the actual conversation content.
    /// This is the difference between rendering with and without add_generation_prompt.
    public static func computeGenPromptLen(
        tokensWithGenPrompt: [Int],
        tokensWithoutGenPrompt: [Int]
    ) -> Int {
        max(tokensWithGenPrompt.count - tokensWithoutGenPrompt.count, 0)
    }

    // MARK: - Passthrough to Base Scheduler

    public func addRequest(_ request: InferenceRequest) {
        scheduler.addRequest(request)
    }

    public func schedule() -> SchedulerOutput {
        scheduler.schedule()
    }

    public func finishRequest(_ requestId: String, reason: FinishReason) {
        scheduler.finishRequest(requestId, reason: reason)
    }

    public func abortRequest(_ requestId: String) {
        scheduler.abortRequest(requestId)
    }

    public func recordOutput(requestId: String, tokenId: Int, text: String) {
        scheduler.recordOutput(requestId: requestId, tokenId: tokenId, text: text)
    }

    public var runningCount: Int { scheduler.runningCount }
    public var waitingCount: Int { scheduler.waitingCount }

    public func shutdown() {
        scheduler.shutdown()
    }

    // MARK: - Private

    private func _extractImageData(from url: String) -> Data? {
        if url.hasPrefix("data:image"), let commaIdx = url.firstIndex(of: ",") {
            let base64 = String(url[url.index(after: commaIdx)...])
            return Data(base64Encoded: base64)
        }
        return nil
    }
}
