#!/usr/bin/env bash
set -euo pipefail

: "${MACOS_CERTIFICATE_BASE64:?MACOS_CERTIFICATE_BASE64 is required}"

CERTIFICATE_PATH="$RUNNER_TEMP/build_certificate.p12"
KEYCHAIN_PATH="$RUNNER_TEMP/app-signing.keychain-db"

echo -n "$MACOS_CERTIFICATE_BASE64" | base64 --decode -o "$CERTIFICATE_PATH"

security create-keychain -p "" "$KEYCHAIN_PATH"
security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH"
security unlock-keychain -p "" "$KEYCHAIN_PATH"

# Ensure the temporary keychain is the default and in the search list
security default-keychain -d user -s "$KEYCHAIN_PATH"

security import "$CERTIFICATE_PATH" -P "freedom" -T /usr/bin/codesign -k "$KEYCHAIN_PATH"
security list-keychain -d user -s "$KEYCHAIN_PATH"

security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "" "$KEYCHAIN_PATH"

# Extract the Developer ID name from the certificate
DEVELOPER_ID_NAME=$(security find-identity -v -p codesigning "$KEYCHAIN_PATH" | grep "Developer ID Application" | sed 's/.*"\(.*\)".*/\1/' | head -1)

if [[ -z "$DEVELOPER_ID_NAME" ]]; then
  echo "::error::Could not find Developer ID Application identity in certificate"
  exit 1
fi

echo "DEVELOPER_ID_NAME=${DEVELOPER_ID_NAME}" >> $GITHUB_OUTPUT
echo "Found signing identity: $DEVELOPER_ID_NAME"