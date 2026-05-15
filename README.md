# deploy_ws

Deployment-side workspace for **UniLab** (SAC + WBT G1-29DOF dance). Adapts
Unitree's open-source robot stack (`unitree_rl_lab` + `unitree_mujoco` +
`unitree_sdk2`) to consume policies trained by the WBT pipeline in UniLab and
runs them on a real Unitree G1 over DDS via `g1_ctrl`.

> The training-side `UniLab` repo is private — request access separately if you
> need to regenerate the assets under `assets/`.

## Layout

```
deploy_ws/
├── assets/                 # in-git: artifacts consumed at runtime
│   ├── policy.onnx         #   154-d obs → 29-d action SAC actor
│   ├── dance1.bin          #   50fps × 870 frames WBT motion clip
│   └── deploy_config.yaml  #   obs_layout, kp/kd, action_scale, joint_ids_map
├── patches/                # in-git: diffs applied on top of pinned upstream
│   ├── unitree_mujoco-g1-scene.patch       # sim2sim: switch default scene to G1
│   └── unitree_rl_lab-wbt-fsm.patch        # adds State_WBT to g1_ctrl FSM
├── scripts/
│   ├── setup.sh                # fetch upstream + apply patches + arch overlay
│   └── smoke_test_wbt.cpp      # offline ONNX + motion-loader sanity check
├── unitree_mujoco/         # not in git — cloned by setup.sh at a pinned SHA
├── unitree_rl_lab/         # not in git — cloned by setup.sh at a pinned SHA
└── unitree_sdk2/           # not in git — cloned by setup.sh at a pinned SHA
```

## Prerequisites

- Linux host, x86_64 **or** aarch64 (tested: Ubuntu 22.04 / Jetson Orin NX).
- A Unitree G1-29DOF on a wired Ethernet link from this host, on a safety rig.
  Hardware setup follows the upstream Unitree docs.

## End-to-end reproduction

Linear, copy-paste from a fresh clone of this repo to a built `g1_ctrl`. Each
block is idempotent — re-running any step is safe.

```bash
# 0. system deps (one time)
sudo apt update
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev \
                    libspdlog-dev libfmt-dev curl file \
                    build-essential cmake git
```

```bash
# 1. clone this repo
git clone https://github.com/LeeLeno/deploy_ws.git
cd deploy_ws
```

```bash
# 2. fetch upstream Unitree repos at pinned SHAs and apply patches.
#    On aarch64 hosts this auto-downloads onnxruntime-linux-aarch64-1.22.0
#    and overlays it into the path upstream's CMakeLists expects.
#    On x86_64 it uses the upstream-vendored x64 build as-is.
./scripts/setup.sh
```

```bash
# 3. install unitree_sdk2 system-wide.
#    Provides /usr/local/lib/libunitree_sdk2.a and DDS headers that g1_ctrl's
#    link step needs.  The header-only dds_wrapper/* pieces come from our
#    pinned tree via the patch's include-path addition; the .a still resolves
#    from /usr/local/lib.
cd unitree_sdk2 && mkdir -p build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install
cd ../..
```

```bash
# 4. build g1_ctrl.  WBT support is compiled in — no separate target.
#    The WBT source files were placed into the upstream tree by step 2's
#    patch, and upstream's CMakeLists.txt picks them up via GLOB_RECURSE.
cd unitree_rl_lab/deploy/robots/g1_29dof && mkdir -p build && cd build
cmake .. && make -j$(nproc)
# produces: ./g1_ctrl
cd ../../../../../..
```

```bash
# 5. (optional) offline smoke test — no DDS, no robot.
#    Exercises the motion loader, the yaw-aligned motion_anchor_ori_b math,
#    and a 154→29 ONNX forward pass.
g++ -std=c++17 -O2 \
    -I /usr/include/eigen3 \
    -I unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/include \
    scripts/smoke_test_wbt.cpp \
    unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0 \
    -o scripts/smoke_test_wbt

LD_LIBRARY_PATH=unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib \
./scripts/smoke_test_wbt
# expected: all checks green; motion_anchor_ori_b[frame 0] ≈ [1,0,0, 1,0,0]
```

At this point the deployment environment is fully reproduced. First-time run on
NX takes ~15–25 min (most of that is compiling `unitree_sdk2` + `g1_ctrl`); a
re-run of `./scripts/setup.sh` afterwards finishes in seconds.

## Run on the real G1

The robot must be on a safety rig and a human must be on the controller before
running this. Swap `eth0` for whichever NIC is wired to the G1.

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network eth0
```

Controller transitions (from upstream `config.yaml` + our `State_WBT` block;
trigger names match the YAML — `LT` is the left analog trigger / L2, `RB` is
the right bumper / R1):

| From            | Gesture                                | To                    |
| --------------- | -------------------------------------- | --------------------- |
| Passive         | `LT + Up`                              | FixStand              |
| FixStand        | `RB + X`                               | Velocity              |
| Velocity        | `LT held ≥2s + Down`                   | Mimic_Dance_102       |
| Velocity        | `LT held ≥2s + Left`                   | Mimic_Gangnam_Style   |
| Velocity        | `LT held ≥2s + Right`                  | **Mimic_WBT_Dance1** (ours) |
| Mimic_WBT_Dance1 | `RB + X`                              | FixStand (clean exit) |
| any RL state    | `LT + B`                               | Passive (emergency)   |

`LT(2s)` is an edge-triggered DSL: hold `LT` for ≥2 s **and keep holding it**
while you press the direction key. Releasing `LT` resets the timer.

Mimic_WBT_Dance1 also auto-falls back to `FixStand` when the clip ends, and to
`Passive` on bad orientation — same safety nets as upstream `State_Mimic`.

## Patches: what they are and how they get into `g1_ctrl`

`patches/*.patch` are plain `git diff` files against the pinned upstream SHAs.
`setup.sh` runs `git apply` after `git checkout <SHA>`, so the patched on-disk
tree is bit-stable across machines. The WBT patch:

- adds `include/State_WBT.h` + `src/State_WBT.cpp` into `unitree_rl_lab/deploy/robots/g1_29dof/`;
- the upstream `CMakeLists.txt` uses `file(GLOB_RECURSE src/*.cpp)`, so the new
  source is picked up by the next `cmake ..` without any further wiring;
- `main.cpp` gains a single `#include "State_WBT.h"` line so the `REGISTER_FSM(State_WBT)`
  static-init macro is not dead-stripped — that registration is what makes
  `type: WBT` in `config.yaml` resolvable at runtime;
- `config.yaml` declares `Mimic_WBT_Dance1` and the `LT(2s) + Right` transition;
- `CMakeLists.txt` adds our pinned `unitree_sdk2/include` path so the
  header-only `dds_wrapper/*` headers resolve even on systems whose
  `/usr/local/include/unitree` predates them.

Net effect: `g1_ctrl` is the WBT-capable binary. There is no separate target.

## Pinned versions

| Component       | Source                                       | Commit / version |
| --------------- | -------------------------------------------- | ---------------- |
| unitree_mujoco  | unitreerobotics/unitree_mujoco               | `1a37b05`        |
| unitree_rl_lab  | unitreerobotics/unitree_rl_lab               | `4960b84`        |
| unitree_sdk2    | unitreerobotics/unitree_sdk2                 | `6d03531`        |
| onnxruntime     | microsoft/onnxruntime release                | `v1.22.0` (linux-x64 vendored by `unitree_rl_lab`; aarch64 overlaid by `setup.sh`) |

To bump a pin: edit the SHA in `scripts/setup.sh`, re-run setup, regenerate any
patch whose context shifted, and commit.

## Updating a patch

```bash
cd unitree_rl_lab            # or unitree_mujoco
# edit files in place
git diff > ../patches/unitree_rl_lab-<short-description>.patch
```

When the diff against upstream grows past ~"a couple of files", switch that
sub-repo to a personal fork pinned as a submodule instead of a patch.

## Sim2Sim (optional)

For mujoco-based testing without a real robot, see upstream
`unitree_rl_lab/README.md` "Deploy → Sim2Sim". Our `unitree_mujoco-g1-scene.patch`
already flips `simulate/config.yaml` to `g1` / `scene_29dof.xml`.
