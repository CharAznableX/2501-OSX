//
//  DirectoryPickerView.swift
//  osaurus
//
//  Created by Kamil Andrusz on 8/22/25.
//

import SwiftUI

/// View for selecting and managing the models directory
struct DirectoryPickerView: View {
    @ObservedObject private var directoryPicker = DirectoryPickerService.shared
    @Environment(\.theme) private var theme
    @State private var showFilePicker = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Directory display field with theme styling
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(directoryDisplayText)
                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                        .foregroundColor(theme.primaryText)
                        .lineLimit(1)
                        .truncationMode(.middle)

                    if directoryPicker.hasValidDirectory {
                        Text("Custom directory selected")
                            .font(.system(size: 11))
                            .foregroundColor(theme.secondaryText)
                    } else {
                        Text("Using default location")
                            .font(.system(size: 11))
                            .foregroundColor(theme.tertiaryText)
                    }
                }

                Spacer()

                // Action buttons with consistent styling
                HStack(spacing: 6) {
                    Button(action: {
                        showFilePicker = true
                    }) {
                        Image(systemName: "folder.badge.gearshape")
                            .font(.system(size: 12))
                            .foregroundColor(theme.primaryText)
                            .frame(width: 24, height: 24)
                            .background(
                                Circle()
                                    .fill(theme.buttonBackground)
                                    .overlay(
                                        Circle()
                                            .stroke(theme.buttonBorder, lineWidth: 1)
                                    )
                            )
                    }
                    .buttonStyle(PlainButtonStyle())
                    .help("Select custom directory")

                    if directoryPicker.hasValidDirectory {
                        Button(action: {
                            directoryPicker.resetDirectory()
                        }) {
                            Image(systemName: "arrow.counterclockwise")
                                .font(.system(size: 12))
                                .foregroundColor(theme.primaryText)
                                .frame(width: 24, height: 24)
                                .background(
                                    Circle()
                                        .fill(theme.buttonBackground)
                                        .overlay(
                                            Circle()
                                                .stroke(theme.buttonBorder, lineWidth: 1)
                                        )
                                )
                        }
                        .buttonStyle(PlainButtonStyle())
                        .help("Reset to default directory")
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(theme.inputBackground)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(theme.inputBorder, lineWidth: 1)
                    )
            )

            // Help text
            Text("Models will be organized in subfolders by repository name")
                .font(.system(size: 11))
                .foregroundColor(theme.tertiaryText)

            // Additional scan directories
            AdditionalDirectoriesView()
        }
        .fileImporter(
            isPresented: $showFilePicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    directoryPicker.saveDirectoryFromFilePicker(url)
                }
            case .failure(let error):
                print("Directory selection failed: \(error)")
            }
        }
    }

    private var directoryDisplayText: String {
        if directoryPicker.hasValidDirectory,
            let selectedDirectory = directoryPicker.selectedDirectory
        {
            return selectedDirectory.path
        } else {
            // Show effective default (env override, old default if exists, else new default)
            let defaultURL = DirectoryPickerService.shared.effectiveModelsDirectory
            return defaultURL.path.replacingOccurrences(of: NSHomeDirectory(), with: "~")
        }
    }
}

// MARK: - Additional Directories

/// Shows user-added model directories with add/remove controls.
struct AdditionalDirectoriesView: View {
    @ObservedObject private var userDirs = UserModelDirectories.shared
    @Environment(\.theme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Additional Model Folders")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(theme.secondaryText)
                Spacer()
                Button(action: { userDirs.pickAndAddDirectory() }) {
                    HStack(spacing: 4) {
                        Image(systemName: "plus")
                            .font(.system(size: 10))
                        Text("Add Folder")
                            .font(.system(size: 11))
                    }
                    .foregroundColor(theme.accentColor)
                }
                .buttonStyle(PlainButtonStyle())
            }

            if userDirs.directories.isEmpty {
                Text("No additional folders. Add folders containing MLX or JANG models.")
                    .font(.system(size: 11))
                    .foregroundColor(theme.tertiaryText)
                    .padding(.vertical, 4)
            } else {
                ForEach(Array(userDirs.directories.enumerated()), id: \.offset) { index, dir in
                    HStack(spacing: 8) {
                        Image(systemName: "folder")
                            .font(.system(size: 11))
                            .foregroundColor(theme.secondaryText)
                        Text(dir.path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(theme.primaryText)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button(action: { userDirs.removeDirectory(at: index) }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 12))
                                .foregroundColor(theme.tertiaryText)
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(theme.inputBackground)
                    )
                }
            }
        }
        .padding(.top, 8)
    }
}

#Preview {
    DirectoryPickerView()
}
