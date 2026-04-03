//
//  PythonSetupOverlay.swift
//  osaurus
//
//  Blocking overlay shown in ChatView when the Python inference environment
//  is not provisioned. Walks the user through one-time setup via uv.
//  Follows the SecretPromptOverlay pattern (bottom-aligned card, spring animation).
//

import SwiftUI

struct PythonSetupOverlay: View {
    @ObservedObject var manager: PythonEnvironmentManager
    let onSkip: () -> Void

    @Environment(\.theme) private var theme
    @State private var isAppearing = false

    var body: some View {
        ZStack {
            theme.primaryBackground.opacity(0.4)
                .ignoresSafeArea()
                .opacity(isAppearing ? 1 : 0)

            VStack {
                Spacer()

                PythonSetupCard(manager: manager, onSkip: onSkip)
                    .opacity(isAppearing ? 1 : 0)
                    .offset(y: isAppearing ? 0 : 30)
                    .padding(.horizontal, 20)
                    .padding(.bottom, 20)
            }
        }
        .onAppear {
            withAnimation(theme.springAnimation()) {
                isAppearing = true
            }
        }
    }
}

// MARK: - Card

private struct PythonSetupCard: View {
    @ObservedObject var manager: PythonEnvironmentManager
    let onSkip: () -> Void

    @Environment(\.theme) private var theme

    var body: some View {
        VStack(spacing: 14) {
            switch manager.state {
            case .checking, .notProvisioned:
                promptContent
            case .provisioning(let step, let detail):
                provisioningContent(step: step, detail: detail)
            case .failed(let message):
                failedContent(message: message)
            case .ready:
                EmptyView()
            }
        }
        .padding(20)
        .frame(maxWidth: 480)
        .background(overlayBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(borderOverlay)
        .shadow(color: theme.shadowColor.opacity(0.12), radius: 16, x: 0, y: 6)
    }

    // MARK: - Prompt (Not Provisioned)

    private var promptContent: some View {
        VStack(spacing: 14) {
            header(icon: "cpu", title: "Set Up Local Inference")

            descriptionBox {
                VStack(alignment: .leading, spacing: 10) {
                    Text("One-time setup to run AI models on your Mac.")
                        .font(theme.font(size: CGFloat(theme.bodySize), weight: .medium))
                        .foregroundColor(theme.primaryText)

                    VStack(alignment: .leading, spacing: 6) {
                        bulletPoint("Python 3.12 runtime")
                        bulletPoint("MLX framework for Apple Silicon")
                        bulletPoint("Model tools and dependencies")
                    }

                    HStack(spacing: 4) {
                        Image(systemName: "internaldrive")
                            .font(.system(size: 10))
                        Text("~400 MB on disk \u{00B7} takes about a minute")
                            .font(theme.font(size: CGFloat(theme.captionSize) - 1, weight: .regular))
                    }
                    .foregroundColor(theme.tertiaryText.opacity(0.7))
                }
            }

            HStack(spacing: 10) {
                Button(action: onSkip) {
                    Text("Skip")
                        .font(theme.font(size: CGFloat(theme.captionSize), weight: .medium))
                        .foregroundColor(theme.tertiaryText)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 9)
                }
                .buttonStyle(.plain)

                Button {
                    Task { await manager.provision() }
                } label: {
                    Text("Set Up Now")
                        .font(theme.font(size: CGFloat(theme.captionSize), weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 9)
                        .background(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(theme.accentColor)
                        )
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Provisioning (Progress)

    private func provisioningContent(step: PythonEnvironmentManager.ProvisionStep, detail: String) -> some View {
        VStack(spacing: 14) {
            header(icon: "arrow.down.circle", title: "Setting Up Local Inference")

            descriptionBox {
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)

                        Text(step.label)
                            .font(theme.font(size: CGFloat(theme.bodySize), weight: .medium))
                            .foregroundColor(theme.primaryText)
                    }

                    ProgressView(
                        value: Double(step.rawValue),
                        total: Double(PythonEnvironmentManager.ProvisionStep.totalSteps)
                    )
                    .progressViewStyle(.linear)
                    .tint(theme.accentColor)

                    HStack {
                        Text(detail)
                            .font(theme.font(size: CGFloat(theme.captionSize) - 1, weight: .regular))
                            .foregroundColor(theme.tertiaryText)

                        Spacer()

                        Text("Step \(step.rawValue) of \(PythonEnvironmentManager.ProvisionStep.totalSteps)")
                            .font(theme.font(size: CGFloat(theme.captionSize) - 1, weight: .medium))
                            .foregroundColor(theme.tertiaryText)
                    }
                }
            }
        }
    }

    // MARK: - Failed (Error + Retry)

    private func failedContent(message: String) -> some View {
        VStack(spacing: 14) {
            header(icon: "exclamationmark.triangle", title: "Setup Failed")

            descriptionBox {
                VStack(alignment: .leading, spacing: 8) {
                    Text(message)
                        .font(theme.font(size: CGFloat(theme.bodySize) - 1, weight: .regular))
                        .foregroundColor(theme.primaryText)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }

            HStack(spacing: 10) {
                Button(action: onSkip) {
                    Text("Dismiss")
                        .font(theme.font(size: CGFloat(theme.captionSize), weight: .medium))
                        .foregroundColor(theme.tertiaryText)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 9)
                }
                .buttonStyle(.plain)

                Button {
                    Task { await manager.repair() }
                } label: {
                    Text("Retry")
                        .font(theme.font(size: CGFloat(theme.captionSize), weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 9)
                        .background(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(theme.accentColor)
                        )
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Shared Components

    private func header(icon: String, title: String) -> some View {
        HStack(alignment: .center, spacing: 12) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(theme.accentColor)

                Text(title)
                    .font(theme.font(size: CGFloat(theme.captionSize), weight: .semibold))
                    .foregroundColor(theme.accentColor)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                Capsule()
                    .fill(theme.accentColor.opacity(theme.isDark ? 0.15 : 0.1))
            )

            Spacer()
        }
    }

    private func descriptionBox<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        content()
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 14)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(theme.inputBackground)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(theme.inputBorder, lineWidth: 1)
            )
    }

    private func bulletPoint(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 11))
                .foregroundColor(theme.accentColor.opacity(0.8))
                .padding(.top, 1)

            Text(text)
                .font(theme.font(size: CGFloat(theme.captionSize), weight: .regular))
                .foregroundColor(theme.secondaryText)
        }
    }

    // MARK: - Background & Border

    private var overlayBackground: some View {
        ZStack {
            if theme.glassEnabled {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.ultraThinMaterial)
            }

            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(theme.cardBackground.opacity(theme.isDark ? 0.85 : 0.92))

            LinearGradient(
                colors: [theme.accentColor.opacity(theme.isDark ? 0.08 : 0.05), Color.clear],
                startPoint: .top,
                endPoint: .center
            )
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
    }

    private var borderOverlay: some View {
        RoundedRectangle(cornerRadius: 16, style: .continuous)
            .strokeBorder(
                LinearGradient(
                    colors: [
                        theme.glassEdgeLight.opacity(0.2),
                        theme.cardBorder,
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                lineWidth: 1
            )
    }
}
