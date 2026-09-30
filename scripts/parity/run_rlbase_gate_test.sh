#!/usr/bin/env bash
# Offline check of the Velocity (State_RLBase) publish gate.  Fake LowCmd /
# LowState (scripts/parity/fake_sdk/Types.h) — no DDS, no robot, no g1_ctrl.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-/tmp/rlbase_gate}"
ORT="$WS/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0"
G1="$WS/unitree_rl_lab/deploy/robots/g1_29dof"
mkdir -p "$OUT"
# fake_sdk must precede every other include dir so its Types.h is picked up.
g++ -std=c++17 -O2 \
    -I "$WS/scripts/parity/fake_sdk" \
    -I /usr/include/eigen3 \
    -I "$WS/unitree_rl_lab/deploy/include" \
    -I "$ORT/include" \
    -I "$WS/unitree_sdk2/include" \
    "$WS/scripts/parity/rlbase_gate_test.cpp" "$G1/src/State_RLBase.cpp" \
    "$ORT/lib/libonnxruntime.so.1.22.0" -lyaml-cpp -lpthread \
    -Wl,-rpath,"$ORT/lib" \
    -o "$OUT/rlbase_gate_test"
"$OUT/rlbase_gate_test" "$G1"
