#!/usr/bin/env bash
# Rootless host dependency setup for building g1_ctrl.
#
# libfmt-dev (static libfmt.a) is fetched with `apt-get download` (no root)
# and unpacked into deploy_ws/.deps/, which g1_29dof/CMakeLists.txt adds to
# the link path.  Nothing is installed system-wide.  Equivalent system
# install: `sudo apt install libfmt-dev`.
#
# Idempotent; safe to re-run.  Does not start anything.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="$WS/.deps"
ARCH_TRIPLET="$(gcc -print-multiarch)"

if [ -f "/usr/lib/$ARCH_TRIPLET/libfmt.a" ] || [ -f "/usr/lib/$ARCH_TRIPLET/libfmt.so" ]; then
    echo "[fmt] system libfmt present — nothing to do"
elif [ -f "$DEPS/usr/lib/$ARCH_TRIPLET/libfmt.a" ]; then
    echo "[fmt] workspace-local libfmt already in $DEPS"
else
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    (cd "$tmp" && apt-get download libfmt-dev)
    deb="$(ls "$tmp"/libfmt-dev_*.deb)"
    # Verify against the SHA256 recorded in the apt index.
    expected="$(apt-get download --print-uris libfmt-dev | awk '{print $4}' | sed 's/^SHA256://')"
    echo "$expected  $deb" | sha256sum -c -
    mkdir -p "$DEPS"
    dpkg -x "$deb" "$DEPS"
    echo "[fmt] unpacked $(basename "$deb") into $DEPS"
fi
