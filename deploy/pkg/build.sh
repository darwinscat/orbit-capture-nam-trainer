#!/bin/bash
# Build the macOS .pkg installer for namtrainerd, signing it when a Developer ID
# Installer identity is available and notarizing+stapling when a notary profile is
# too. A stapled pkg installs by double-click with no Gatekeeper prompt, offline.
#
# Usage: deploy/pkg/build.sh <version> <path-to-namtrainerd-binary> [out-dir]
# Env:
#   INSTALLER_ID    "Developer ID Installer: Name (TEAMID)" — unset ⇒ UNSIGNED (dev only).
#   NOTARY_PROFILE  notarytool keychain profile — set (and signed) ⇒ notarize + staple.
#
# The binary handed in should already be Developer-ID-*Application*-signed (the pkg
# signature is separate from the payload binary's own signature).
set -euo pipefail

VER="${1:?version required}"
BIN="${2:?path to namtrainerd binary required}"
OUT="${3:-.}"
LABEL="net.lafox.namtrainerd"
here="$(cd "$(dirname "$0")" && pwd)"

[ -f "$BIN" ] || { echo "no binary at $BIN" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/root/usr/local/bin" \
         "$work/root/usr/local/share/OrbitCaptureNamTrainer" \
         "$work/scripts"
install -m 0755 "$BIN" "$work/root/usr/local/bin/namtrainerd"
install -m 0644 "$here/../launchd/$LABEL.plist" \
        "$work/root/usr/local/share/OrbitCaptureNamTrainer/$LABEL.plist"
install -m 0755 "$here/scripts/preinstall" "$work/scripts/preinstall"
install -m 0755 "$here/scripts/postinstall" "$work/scripts/postinstall"

# Tool chatter → stderr so the only stdout line is the final pkg path (callers can
# capture it). The binary's com.apple.provenance xattr shows up as a harmless
# AppleDouble (._*) entry in the payload listing; the installer applies it as an
# xattr, so nothing lands on the installed disk beyond the file itself.
comp="$work/component.pkg"
pkgbuild --root "$work/root" --identifier "$LABEL" --version "$VER" \
         --install-location / --scripts "$work/scripts" "$comp" 1>&2

mkdir -p "$OUT"
final="$OUT/namtrainerd-$VER-macos-arm64.pkg"
if [ -n "${INSTALLER_ID:-}" ]; then
  productsign --sign "$INSTALLER_ID" "$comp" "$final" 1>&2
  if [ -n "${NOTARY_PROFILE:-}" ]; then
    xcrun notarytool submit "$final" --keychain-profile "$NOTARY_PROFILE" --wait 1>&2
    xcrun stapler staple "$final" 1>&2
  fi
else
  echo "::warning::no INSTALLER_ID — producing an UNSIGNED pkg" >&2
  mv "$comp" "$final"
fi
echo "$final"
