#!/usr/bin/env bash
# Offline State_WBT parity: mimiclite_reference.py (NumPy port of the UniLab
# MimicLite actor terms) vs WBTPolicy (C++).  No DDS, no robot, no g1_ctrl.
#
#   PYTHON=/path/to/python [UNILAB=/path/to/UniLab] scripts/parity/run_parity.sh [out_dir]
#
# PYTHON needs numpy, pyyaml and onnxruntime >= 1.18 (policy.onnx is IR v10
# with external weights in policy.onnx.data).  UNILAB (default: UniLab next to
# deploy_ws) provides the G1 MJCF from origin/feat/mimiclite-obs.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-/tmp/wbt_parity}"
PYTHON="${PYTHON:-python3}"
ORT="$WS/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0"   # host-arch overlay, see setup.sh
G1="$WS/unitree_rl_lab/deploy/robots/g1_29dof"
mkdir -p "$OUT/lib"
ln -sf "$ORT/lib/libonnxruntime.so.1.22.0" "$OUT/lib/libonnxruntime.so.1"

"$PYTHON" "$WS/scripts/parity/mimiclite_reference.py" --out "$OUT" ${UNILAB:+--unilab "$UNILAB"}

g++ -std=c++17 -O2 \
    -I /usr/include/eigen3 \
    -I "$WS/unitree_rl_lab/deploy/include" \
    -I "$G1/include" \
    -I "$ORT/include" \
    -I "$WS/unitree_sdk2/include" \
    "$WS/scripts/parity/wbt_parity.cpp" "$G1/src/WBT_Policy.cpp" \
    "$ORT/lib/libonnxruntime.so.1.22.0" -lyaml-cpp -lpthread \
    -o "$OUT/wbt_parity"

LD_LIBRARY_PATH="$OUT/lib" "$OUT/wbt_parity" \
    "$WS/assets/mimiclite_dr_140k_deploy_config.yaml" "$G1/config/config.yaml" \
    "$WS/assets/policy.onnx" "$WS/assets/dance1_subject2_part.bin" "$OUT" Mimic_WBT_Dance1
