import Foundation
import MLX

/// How a VLM architecture handles image tokens in the input sequence.
public enum VLMImageTokenStrategy: String, Sendable {
    /// Image tokens replace placeholder tokens in the sequence (Qwen-VL, InternVL).
    case replacement
    /// Image embeddings are prepended to the text sequence (LLaVA).
    case prepend
    /// Image features are concatenated with text at specific positions (Pixtral).
    case interleave
}

/// Configuration for a specific VLM architecture.
public struct VLMConfig: Sendable {
    /// Model family name.
    public let family: String

    /// How image tokens are inserted into the sequence.
    public let imageTokenStrategy: VLMImageTokenStrategy

    /// Special token ID that marks where an image should be inserted.
    /// E.g., Qwen-VL uses <|image_pad|>, InternVL uses <IMG_CONTEXT>.
    public let imageTokenId: Int?

    /// Number of image tokens per image (fixed models) or nil (variable/dynamic).
    public let imageTokenCount: Int?

    /// Whether the model uses grid-based tiling (Qwen-VL variable resolution).
    public let usesGridTiling: Bool

    /// Maximum number of images per request.
    public let maxImages: Int

    /// Maximum image size in bytes.
    public let maxImageSizeBytes: Int

    /// Maximum number of video frames.
    public let maxVideoFrames: Int

    /// Vision encoder output dimension.
    public let visionDim: Int?

    public init(
        family: String,
        imageTokenStrategy: VLMImageTokenStrategy = .replacement,
        imageTokenId: Int? = nil,
        imageTokenCount: Int? = nil,
        usesGridTiling: Bool = false,
        maxImages: Int = 5,
        maxImageSizeBytes: Int = 50 * 1024 * 1024,  // 50MB
        maxVideoFrames: Int = 64,
        visionDim: Int? = nil
    ) {
        self.family = family
        self.imageTokenStrategy = imageTokenStrategy
        self.imageTokenId = imageTokenId
        self.imageTokenCount = imageTokenCount
        self.usesGridTiling = usesGridTiling
        self.maxImages = maxImages
        self.maxImageSizeBytes = maxImageSizeBytes
        self.maxVideoFrames = maxVideoFrames
        self.visionDim = visionDim
    }
}

/// Known VLM configurations.
public struct VLMConfigRegistry: Sendable {

    public static let configs: [String: VLMConfig] = [
        "qwen2.5-vl": VLMConfig(
            family: "qwen2.5-vl",
            imageTokenStrategy: .replacement,
            usesGridTiling: true,
            maxImages: 10,
            visionDim: 1280
        ),
        "qwen3.5-vl": VLMConfig(
            family: "qwen3.5-vl",
            imageTokenStrategy: .replacement,
            usesGridTiling: true,
            maxImages: 10,
            visionDim: 1280
        ),
        "pixtral": VLMConfig(
            family: "pixtral",
            imageTokenStrategy: .interleave,
            maxImages: 5,
            visionDim: 1024
        ),
        "internvl": VLMConfig(
            family: "internvl",
            imageTokenStrategy: .replacement,
            maxImages: 5,
            visionDim: 3200
        ),
        "llava": VLMConfig(
            family: "llava",
            imageTokenStrategy: .prepend,
            imageTokenCount: 576,  // Fixed grid
            maxImages: 1,
            visionDim: 1024
        ),
        "gemma-3n": VLMConfig(
            family: "gemma-3n",
            imageTokenStrategy: .replacement,
            maxImages: 5
        ),
        "phi-3-vision": VLMConfig(
            family: "phi-3-vision",
            imageTokenStrategy: .replacement,
            imageTokenCount: 256,
            maxImages: 3,
            visionDim: 768
        ),
    ]

    /// Auto-detect VLM config from model name.
    public static func detect(modelName: String) -> VLMConfig? {
        let name = modelName.lowercased()
        for (key, config) in configs {
            if name.contains(key) { return config }
        }
        return nil
    }
}

/// Prepared multimodal input ready for model forward pass.
public struct VLMInput: @unchecked Sendable {
    /// Token IDs with image placeholders (if replacement strategy).
    public let tokenIds: [Int]

    /// Image embeddings from vision encoder, one per image.
    public let imageEmbeddings: [MLXArray]

    /// Grid dimensions per image (for variable resolution models).
    /// Each entry is (temporal, height_tiles, width_tiles).
    public let gridTHW: [(Int, Int, Int)]

    /// Positions in tokenIds where images should be inserted.
    public let imagePositions: [Int]

    public init(
        tokenIds: [Int],
        imageEmbeddings: [MLXArray] = [],
        gridTHW: [(Int, Int, Int)] = [],
        imagePositions: [Int] = []
    ) {
        self.tokenIds = tokenIds
        self.imageEmbeddings = imageEmbeddings
        self.gridTHW = gridTHW
        self.imagePositions = imagePositions
    }

    /// Whether this input has any images.
    public var hasImages: Bool { !imageEmbeddings.isEmpty }

    /// Total number of image tokens across all images.
    public var totalImageTokens: Int {
        imageEmbeddings.reduce(0) { total, emb in
            total + emb.shape[emb.ndim >= 2 ? emb.ndim - 2 : 0]
        }
    }
}

/// Protocol for VLM model wrappers.
/// Implementations handle the specific architecture's vision+text fusion.
public protocol VLMModelProtocol: Sendable {
    /// Encode images through the vision encoder.
    func encodeImages(_ images: [ProcessedImage]) async throws -> [MLXArray]

    /// Prepare input for the language model (combine text tokens + image embeddings).
    func prepareInput(tokenIds: [Int], images: [ProcessedImage]) async throws -> VLMInput
}
