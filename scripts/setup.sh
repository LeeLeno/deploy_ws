#!/usr/bin/env bash
# Reproduce the deploy_ws workspace: clone upstream Unitree repos at pinned
# commits, ensure onnxruntime matches the host arch, and apply local patches.
#
# Arch handling: upstream unitree_rl_lab vendors a x86-64 build of onnxruntime
# 1.22.0 under deploy/thirdparty/.  On aarch64 hosts (e.g. Jetson Orin NX) we
# overlay an aarch64 build into the same directory after `git checkout` runs,
# so the upstream CMakeLists (which hard-codes the x64 path) compiles and
# links unchanged.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PIN_MUJOCO=1a37b051a10be723405b7ed6dc839361af036d88
PIN_RL_LAB=4960b84732b0c2ec593dccbfe963fda1bcd7b1e3
PIN_SDK2=6d03531e109135b18500c7c3e6f8408ba857134e

# onnxruntime version vendored by upstream unitree_rl_lab.  Must match the
# directory name + .so soname referenced by g1_29dof/CMakeLists.txt.
ORT_VERSION=1.22.0
ORT_VENDORED_DIR="unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-${ORT_VERSION}"

HOST_ARCH="$(uname -m)"
case "$HOST_ARCH" in
    x86_64)
        # Upstream's vendored .so is already x86-64; nothing to swap.
        ORT_NEED_OVERLAY=0
        ;;
    aarch64)
        ORT_NEED_OVERLAY=1
        ORT_AARCH64_ARCHIVE="onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
        ORT_AARCH64_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_AARCH64_ARCHIVE}"
        ORT_AARCH64_SHA256=bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702
        ;;
    *)
        echo "ERROR: unsupported host arch '$HOST_ARCH' (need x86_64 or aarch64)" >&2
        exit 1
        ;;
esac

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

overlay_aarch64_onnxruntime() {
    # Overlay onnxruntime aarch64 binaries onto the upstream-vendored x64 path
    # so g1_29dof/CMakeLists.txt builds unchanged on aarch64.  Idempotent: we
    # check the existing .so's ELF machine before re-downloading / overwriting.
    local existing_so="$ORT_VENDORED_DIR/lib/libonnxruntime.so.${ORT_VERSION}"
    if [ -f "$existing_so" ] \
       && file -L "$existing_so" 2>/dev/null | grep -q "ARM aarch64"; then
        echo "[onnxruntime] aarch64 overlay already in place"
        return
    fi

    if [ ! -f "$ORT_AARCH64_ARCHIVE" ]; then
        echo "[onnxruntime] downloading $ORT_AARCH64_URL"
        curl -L --fail -o "$ORT_AARCH64_ARCHIVE" "$ORT_AARCH64_URL"
    fi
    echo "[onnxruntime] verifying SHA256..."
    echo "${ORT_AARCH64_SHA256}  ${ORT_AARCH64_ARCHIVE}" | sha256sum -c -

    local tmp; tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    tar -xzf "$ORT_AARCH64_ARCHIVE" -C "$tmp"
    local src="$tmp/onnxruntime-linux-aarch64-${ORT_VERSION}"
    if [ ! -d "$src/include" ] || [ ! -d "$src/lib" ]; then
        echo "ERROR: unexpected archive layout in $ORT_AARCH64_ARCHIVE" >&2
        return 1
    fi

    mkdir -p "$ORT_VENDORED_DIR"
    rm -rf "$ORT_VENDORED_DIR/include" "$ORT_VENDORED_DIR/lib"
    cp -r "$src/include" "$src/lib" "$ORT_VENDORED_DIR/"
    echo "[onnxruntime] overlaid aarch64 $ORT_VERSION into $ORT_VENDORED_DIR"
}

echo "=== 1/3 cloning Unitree repos at pinned commits ==="
clone_pinned unitree_mujoco https://github.com/unitreerobotics/unitree_mujoco.git "$PIN_MUJOCO"
clone_pinned unitree_rl_lab https://github.com/unitreerobotics/unitree_rl_lab.git "$PIN_RL_LAB"
clone_pinned unitree_sdk2   https://github.com/unitreerobotics/unitree_sdk2.git   "$PIN_SDK2"

echo ""
echo "=== 2/3 ensuring onnxruntime matches host arch ($HOST_ARCH) ==="
if [ "$ORT_NEED_OVERLAY" -eq 1 ]; then
    overlay_aarch64_onnxruntime
else
    echo "[onnxruntime] x86_64 host — using upstream-vendored $ORT_VENDORED_DIR/"
fi

echo ""
echo "=== 3/3 applying local patches ==="
apply_patch_idempotent unitree_mujoco "$REPO_ROOT/patches/unitree_mujoco-g1-scene.patch"
apply_patch_idempotent unitree_rl_lab "$REPO_ROOT/patches/unitree_rl_lab-wbt-fsm.patch"

echo ""
echo "✓ setup complete.  workspace layout:"
ls -d unitree_mujoco unitree_rl_lab unitree_sdk2 assets 2>/dev/null
echo "  onnxruntime: $(file -L "$ORT_VENDORED_DIR/lib/libonnxruntime.so.${ORT_VERSION}" | cut -d: -f2-)"
