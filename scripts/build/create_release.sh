#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"

# Target repository for release assets (keep in sync with generate_and_deploy_appcast.sh)
PUBLIC_REPO="${PUBLIC_REPO:-$GITHUB_REPOSITORY}"

IS_BETA="${IS_BETA:-false}"

git config --global user.name "github-actions[bot]"
git config --global user.email "github-actions[bot]@users.noreply.github.com"

RELEASE_FLAGS=()
if [ "$IS_BETA" = "true" ]; then
  RELEASE_FLAGS+=(--prerelease)
  RELEASE_FLAGS+=(--title "Project2501 ${VERSION} (Beta)")
else
  RELEASE_FLAGS+=(--latest)
  RELEASE_FLAGS+=(--title "Project2501 ${VERSION}")
fi

gh release create "${VERSION}" \
  "build_output/Project2501-${VERSION}.dmg" \
  "build_output/Project2501.dmg" \
  --repo "${PUBLIC_REPO}" \
  --notes-file RELEASE_NOTES.md \
  "${RELEASE_FLAGS[@]}"

echo "✅ Release created successfully"
if [ "$IS_BETA" = "true" ]; then
  echo "🧪 Beta release URL: https://github.com/${PUBLIC_REPO}/releases/tag/${VERSION}"
else
  echo "📥 Latest download URL: https://github.com/${PUBLIC_REPO}/releases/latest/download/Project2501.dmg"
fi
