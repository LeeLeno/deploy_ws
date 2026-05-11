#!/usr/bin/env bash
# Reproduce the deploy_ws workspace: clone upstream Unitree repos at pinned
# commits, download onnxruntime, and apply local patches.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PIN_MUJOCO=1a37b051a10be723405b7ed6dc839361af036d88
PIN_RL_LAB=4960b84732b0c2ec593dccbfe963fda1bcd7b1e3
PIN_SDK2=6d03531e109135b18500c7c3e6f8408ba857134e

ONNX_VERSION=1.21.0
ONNX_ARCHIVE="onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_ARCHIVE}"
ONNX_SHA256=7485c7e7aac6501b27e353dcbe068e45c61ab51fbaf598d13970dfae669d20bf

clone_pinned() {
    local name="$1" url="$2" sha="$3"
    if [ -d "$name/.git" ]; then
        echo "[$name] already cloned, fetching..."
        git -C "$name" fetch --quiet origin
    else
        echo "[$name] cloning..."
        git clone --quiet "$url" "$name"
    fi
    git -C "$name" checkout --quiet "$sha"
    echo "[$name] checked out $sha"
}

apply_patch_idempotent() {
    local repo="$1" patch="$2"
    if git -C "$repo" apply --check --reverse "$patch" 2>/dev/null; then
        echo "[$repo] patch already applied: $(basename "$patch")"
        return
    fi
    git -C "$repo" apply "$patch"
    echo "[$repo] applied patch: $(basename "$patch")"
}

echo "=== 1/3 cloning Unitree repos at pinned commits ==="
clone_pinned unitree_mujoco https://github.com/unitreerobotics/unitree_mujoco.git "$PIN_MUJOCO"
clone_pinned unitree_rl_lab https://github.com/unitreerobotics/unitree_rl_lab.git "$PIN_RL_LAB"
clone_pinned unitree_sdk2   https://github.com/unitreerobotics/unitree_sdk2.git   "$PIN_SDK2"

echo ""
echo "=== 2/3 fetching onnxruntime $ONNX_VERSION ==="
if [ ! -d "onnxruntime-linux-x64-${ONNX_VERSION}" ]; then
    if [ ! -f "$ONNX_ARCHIVE" ]; then
        echo "downloading $ONNX_URL"
        curl -L --fail -o "$ONNX_ARCHIVE" "$ONNX_URL"
    fi
    echo "verifying SHA256..."
    echo "${ONNX_SHA256}  ${ONNX_ARCHIVE}" | sha256sum -c -
    tar -xzf "$ONNX_ARCHIVE"
    echo "extracted onnxruntime-linux-x64-${ONNX_VERSION}/"
else
    echo "onnxruntime-linux-x64-${ONNX_VERSION}/ already present"
fi

echo ""
echo "=== 3/3 applying local patches ==="
apply_patch_idempotent unitree_mujoco "$REPO_ROOT/patches/unitree_mujoco-g1-scene.patch"

echo ""
echo "✓ setup complete. workspace layout:"
ls -d unitree_mujoco unitree_rl_lab unitree_sdk2 "onnxruntime-linux-x64-${ONNX_VERSION}" assets 2>/dev/null
