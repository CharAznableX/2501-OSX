//
//  TranscriptionModeService.swift
//  project2501
//
//  Main service for Transcription Mode.
//  Orchestrates hotkey handling, speech transcription, keyboard simulation,
//  and the floating overlay UI.
//

import AppKit
import Combine
import Foundation

/// State of the transcription mode session
public enum TranscriptionModeState: Equatable {
    case idle
    case starting
    case transcribing
    case stopping
    case error(String)
}

/// Service that manages the Transcription Mode lifecycle
@MainActor
public final class TranscriptionModeService: ObservableObject {
    public static let shared = TranscriptionModeService()

    // MARK: - Published State

    /// Current state of transcription mode
    @Published public private(set) var state: TranscriptionModeState = .idle

    /// Whether transcription mode is enabled in settings
    @Published public private(set) var isEnabled: Bool = false

    /// Current configuration
    @Published public private(set) var configuration: TranscriptionConfiguration = .default

    // MARK: - Dependencies

    private let speechService = SpeechService.shared
    private let keyboardService = KeyboardSimulationService.shared
    private let hotkeyManager = TranscriptionHotKeyManager.shared
    private let overlayService = TranscriptionOverlayWindowService.shared

    // MARK: - Private State

    private var lastTypedText: String = ""
    private var configCancellables = Set<AnyCancellable>()
    private var transcriptionCancellables = Set<AnyCancellable>()
    private var escKeyMonitor: Any?

    private init() {
        loadConfiguration()
        setupOverlayCallbacks()
    }

    // MARK: - Public API

    public func initialize() {
        loadConfiguration()
        registerHotkeyIfNeeded()

        NotificationCenter.default.publisher(for: .transcriptionConfigurationChanged)
            .sink { [weak self] _ in
                self?.loadConfiguration()
                self?.registerHotkeyIfNeeded()
            }
            .store(in: &configCancellables)
    }

    public func toggle() {
        switch state {
        case .idle:
            startTranscription()
        case .transcribing:
            stopTranscription()
        case .starting, .stopping:
            break
        case .error:
            state = .idle
            startTranscription()
        }
    }

    public func startTranscription() {
        switch state {
        case .idle, .error: break
        default:
            print("[TranscriptionMode] Cannot start: already in state \(state)")
            return
        }

        keyboardService.checkAccessibilityPermission()
        guard keyboardService.hasAccessibilityPermission else {
            state = .error("Accessibility permission required")
            keyboardService.requestAccessibilityPermission()
            return
        }

        guard speechService.isModelLoaded || SpeechModelManager.shared.selectedModel != nil else {
            state = .error("No speech model available")
            return
        }

        state = .starting
        lastTypedText = ""
        overlayService.show()
        startEscKeyMonitoring()

        Task {
            do {
                try await speechService.startStreamingTranscription()
                state = .transcribing
                subscribeToTranscriptionUpdates()
                print("[TranscriptionMode] Started transcription")
            } catch {
                state = .error(error.localizedDescription)
                overlayService.hide()
                stopEscKeyMonitoring()
                print("[TranscriptionMode] Failed to start: \(error)")
            }
        }
    }

    public func stopTranscription() {
        guard state == .transcribing || state == .starting else { return }

        state = .stopping
        stopEscKeyMonitoring()
        transcriptionCancellables.removeAll()

        Task {
            // stop recording and get final result
            _ = await speechService.stopStreamingTranscription()

            // if using clipboard paste, do it now at the end
            if configuration.useClipboardPaste {
                let fullText = speechService.confirmedTranscription
                if !fullText.isEmpty {
                    print("[TranscriptionMode] Pasting \(fullText.count) characters via clipboard")
                    keyboardService.pasteText(fullText)
                }
            }

            speechService.clearTranscription()
            overlayService.hide()
            lastTypedText = ""
            state = .idle
            print("[TranscriptionMode] Stopped transcription")
        }
    }

    // MARK: - Private Helpers

    private func loadConfiguration() {
        configuration = TranscriptionConfigurationStore.load()
        isEnabled = configuration.transcriptionModeEnabled
    }

    private func registerHotkeyIfNeeded() {
        if isEnabled, let hotkey = configuration.hotkey {
            hotkeyManager.register(hotkey: hotkey) { [weak self] in
                Task { @MainActor in
                    self?.toggle()
                }
            }
            print("[TranscriptionMode] Hotkey registered: \(hotkey.displayString)")
        } else {
            hotkeyManager.unregister()
            print("[TranscriptionMode] Hotkey unregistered")
        }
    }

    private func setupOverlayCallbacks() {
        overlayService.onDone = { [weak self] in
            self?.stopTranscription()
        }
        overlayService.onCancel = { [weak self] in
            self?.stopTranscription()
        }
    }

    private func subscribeToTranscriptionUpdates() {
        speechService.$confirmedTranscription
            .combineLatest(speechService.$currentTranscription)
            .receive(on: RunLoop.main)
            .sink { [weak self] _, _ in
                self?.handleTranscriptionUpdate()
            }
            .store(in: &transcriptionCancellables)

        speechService.$audioLevel
            .sink { [weak self] level in
                self?.overlayService.updateAudioLevel(level)
            }
            .store(in: &transcriptionCancellables)
    }

    private func handleTranscriptionUpdate() {
        guard state == .transcribing else { return }

        // skip live typing if clipboard paste is enabled
        if configuration.useClipboardPaste {
            return
        }

        let fullText: String
        if speechService.confirmedTranscription.isEmpty {
            fullText = speechService.currentTranscription
        } else if speechService.currentTranscription.isEmpty {
            fullText = speechService.confirmedTranscription
        } else {
            fullText = speechService.confirmedTranscription + " " + speechService.currentTranscription
        }

        typeNewText(fullText)
    }

    /// Diff-based typing: compares `fullText` against `lastTypedText` and
    /// issues only the minimal keystrokes (append, delete, or correct) needed.
    private func typeNewText(_ fullText: String) {
        if fullText.hasPrefix(lastTypedText) {
            let newPart = String(fullText.dropFirst(lastTypedText.count))
            if !newPart.isEmpty {
                keyboardService.typeText(newPart)
                lastTypedText = fullText
            }
        } else if lastTypedText.hasPrefix(fullText) {
            let charsToDelete = lastTypedText.count - fullText.count
            if charsToDelete > 0 {
                keyboardService.typeBackspace(count: charsToDelete)
                lastTypedText = fullText
            }
        } else {
            let commonPrefixLength = zip(lastTypedText, fullText).prefix(while: { $0 == $1 }).count
            let charsToDelete = lastTypedText.count - commonPrefixLength
            let newPart = String(fullText.dropFirst(commonPrefixLength))

            if charsToDelete > 0 {
                keyboardService.typeBackspace(count: charsToDelete)
            }
            if !newPart.isEmpty {
                keyboardService.typeText(newPart)
            }
            lastTypedText = fullText
        }
    }

    // MARK: - Esc Key Monitoring

    private func startEscKeyMonitoring() {
        stopEscKeyMonitoring()

        escKeyMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if event.keyCode == 53 {  // Esc
                Task { @MainActor in
                    self?.stopTranscription()
                }
            }
        }
    }

    private func stopEscKeyMonitoring() {
        if let monitor = escKeyMonitor {
            NSEvent.removeMonitor(monitor)
            escKeyMonitor = nil
        }
    }
}
