//
//  InferenceProgressManager.swift
//  osaurus
//
//  Observable singleton that broadcasts prefill progress so the UI can show
//  "Processing N tokens…" while the GPU is doing its initial prompt forward pass.
//

import Foundation

/// Singleton observable that tracks in-flight prefill progress.
///
/// Stored-property mutations are always dispatched to the MainActor so that
/// SwiftUI bindings are updated correctly.  Call sites that are NOT on the
/// MainActor use the fire-and-forget `*Async` variants.
final class InferenceProgressManager: ObservableObject, @unchecked Sendable {
    static let shared = InferenceProgressManager()

    /// Non-nil while a prefill is in progress.  Set to the prompt token count
    /// just before `prepareAndGenerate` is called; cleared as soon as the first
    /// generated token arrives (or on error / cancellation).
    @MainActor @Published var prefillTokenCount: Int? = nil

    /// Wall-clock time when the current prefill started.
    @MainActor @Published var prefillStartedAt: Date? = nil

    init() {}

    #if DEBUG
        /// Test-only factory: creates an isolated instance so tests don't share
        /// state with the `shared` singleton.
        static func _testMake() -> InferenceProgressManager { InferenceProgressManager() }
    #endif

    /// Called from the MainActor just before prefill begins.
    @MainActor func prefillWillStart(tokenCount: Int) {
        if prefillTokenCount == nil { prefillStartedAt = Date() }
        prefillTokenCount = tokenCount
    }

    /// Called from the MainActor when the first token is generated (prefill done)
    /// or on error / cancellation.
    @MainActor func prefillDidFinish() {
        prefillTokenCount = nil
        prefillStartedAt = nil
    }

    /// Fire-and-forget variant for call sites that are not on MainActor.
    func prefillWillStartAsync(tokenCount: Int) {
        Task { @MainActor in self.prefillWillStart(tokenCount: tokenCount) }
    }

    /// Fire-and-forget variant for call sites that are not on MainActor.
    func prefillDidFinishAsync() {
        Task { @MainActor in self.prefillDidFinish() }
    }

    // MARK: - Generation Stats (vmlx engine)

    /// Whether stats display is enabled (user preference)
    @MainActor @Published var showStats: Bool = UserDefaults.standard.bool(forKey: "showInferenceStats")

    /// Prompt tokens for the current/last generation
    @MainActor @Published var promptTokens: Int = 0
    /// Completion tokens generated so far
    @MainActor @Published var completionTokens: Int = 0
    /// Cached tokens from prefix cache
    @MainActor @Published var cachedTokens: Int = 0
    /// Cache detail string (e.g. "paged", "disk")
    @MainActor @Published var cacheDetail: String?
    /// Tokens per second (computed from deltas)
    @MainActor @Published var tokensPerSecond: Double = 0
    /// Time to first token in seconds
    @MainActor @Published var timeToFirstToken: Double? = nil
    /// Whether generation is active
    @MainActor @Published var isGenerating: Bool = false

    /// Internal: timestamp when generation started
    private var generationStartTime: Date?
    /// Internal: timestamp of first content token
    private var firstTokenTime: Date?
    /// Internal: completion token count at last TPS calculation
    private var lastTokenCount: Int = 0
    /// Internal: time of last TPS calculation
    private var lastTPSTime: Date?

    @MainActor func toggleStats() {
        showStats.toggle()
        UserDefaults.standard.set(showStats, forKey: "showInferenceStats")
    }

    @MainActor func generationDidStart() {
        isGenerating = true
        promptTokens = 0
        completionTokens = 0
        cachedTokens = 0
        cacheDetail = nil
        tokensPerSecond = 0
        timeToFirstToken = nil
        generationStartTime = Date()
        firstTokenTime = nil
        lastTokenCount = 0
        lastTPSTime = Date()
    }

    @MainActor func updateStats(prompt: Int, completion: Int, cached: Int, detail: String?) {
        promptTokens = prompt
        cachedTokens = cached
        cacheDetail = detail

        // Calculate TPS from completion token count growth
        if completion > completionTokens {
            let now = Date()
            if firstTokenTime == nil, let start = generationStartTime {
                firstTokenTime = now
                timeToFirstToken = now.timeIntervalSince(start)
            }
            if let lastTime = lastTPSTime, completion > lastTokenCount {
                let elapsed = now.timeIntervalSince(lastTime)
                if elapsed > 0.1 {  // Update TPS at most 10x/sec
                    let newTokens = completion - lastTokenCount
                    tokensPerSecond = Double(newTokens) / elapsed
                    lastTokenCount = completion
                    lastTPSTime = now
                }
            }
            completionTokens = completion
        }
    }

    @MainActor func generationDidFinish() {
        isGenerating = false
        // Keep final stats visible until next generation
    }

    func generationDidStartAsync() {
        Task { @MainActor in self.generationDidStart() }
    }

    func updateStatsAsync(prompt: Int, completion: Int, cached: Int, detail: String?) {
        Task { @MainActor in self.updateStats(prompt: prompt, completion: completion, cached: cached, detail: detail) }
    }

    func generationDidFinishAsync() {
        Task { @MainActor in self.generationDidFinish() }
    }
}
