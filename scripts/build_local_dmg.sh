#!/usr/bin/env bash
set -euo pipefail

# Local DMG build script (unsigned, for testing)
# Usage: ./scripts/build_local_dmg.sh [version]

VERSION="${1:-1.0.0-dev}"
echo "Building version: $VERSION"

# Clean previous build
rm -rf build/DerivedData build/SourcePackages 2>/dev/null || true
mkdir -p build_output

# Resolve packages
echo "Resolving package dependencies..."
xcodebuild -resolvePackageDependencies -workspace project2501.xcworkspace -scheme project2501

# Build CLI first
echo "Building CLI..."
xcodebuild -workspace project2501.xcworkspace \
  -scheme project2501-cli \
  -configuration Release \
  -derivedDataPath build \
  ARCHS=arm64 \
  VALID_ARCHS=arm64 \
  ONLY_ACTIVE_ARCH=NO \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGN_IDENTITY="" \
  CODE_SIGNING_REQUIRED=NO \
  clean build

# Build the app
echo "Building app..."
xcodebuild -workspace project2501.xcworkspace \
  -scheme project2501 \
  -configuration Release \
  -derivedDataPath build \
  ARCHS=arm64 \
  VALID_ARCHS=arm64 \
  ONLY_ACTIVE_ARCH=NO \
  MARKETING_VERSION="$VERSION" \
  CURRENT_PROJECT_VERSION="$VERSION" \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGN_IDENTITY="" \
  CODE_SIGNING_REQUIRED=NO \
  clean build

# Copy app to build_output
echo "Copying app..."
APP_PATH="build/Build/Products/Release/project2501.app"
if [[ ! -d "$APP_PATH" ]]; then
  echo "Error: App not found at $APP_PATH"
  exit 1
fi

rm -rf build_output/project2501.app 2>/dev/null || true
cp -R "$APP_PATH" build_output/

# Embed CLI
echo "Embedding CLI..."
CLI_SRC="build/Build/Products/Release/project2501-cli"
if [[ -f "$CLI_SRC" ]]; then
  mkdir -p "build_output/project2501.app/Contents/Helpers"
  cp "$CLI_SRC" "build_output/project2501.app/Contents/Helpers/project2501"
  chmod +x "build_output/project2501.app/Contents/Helpers/project2501"
  echo "CLI embedded successfully"
else
  echo "Warning: CLI binary not found at $CLI_SRC"
fi

# Create DMG
echo "Creating DMG..."
DMG_PATH="build_output/Project2501-${VERSION}.dmg"
rm -f "$DMG_PATH" 2>/dev/null || true

# Use hdiutil to create DMG
hdiutil create -volname "Project2501" \
  -srcfolder "build_output/project2501.app" \
  -ov -format UDZO \
  "$DMG_PATH"

echo ""
echo "✓ Build complete!"
echo "  App: build_output/project2501.app"
echo "  DMG: $DMG_PATH"