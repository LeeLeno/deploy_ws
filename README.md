# deploy_ws

Deployment-side workspace for **UniLab** (SAC + WBT G1-29DOF dance). Adapts
Unitree's open-source stack (`unitree_rl_lab` + `unitree_mujoco` +
`unitree_sdk2`) to run WBT-trained policies on a real G1 over DDS via
`g1_ctrl`.

> The training-side `UniLab` repo is private — request access separately if
> you need to regenerate `assets/`.

## Layout

```
deploy_ws/
├── assets/                 # in-git runtime artifacts
│   ├── policy.onnx         #   514-d obs (5-step proprio history) → 29-d action
│   ├── dance1.bin          #   WBT motion clip, header-encoded fps & length
│   └── deploy_config.yaml  #   obs_layout, kp/kd, action_scale, joint_ids_map
├── patches/                # diffs against pinned upstream
│   ├── unitree_mujoco-g1-scene.patch    # sim2sim: default scene → G1
│   └── unitree_rl_lab-wbt-fsm.patch     # adds State_WBT to g1_ctrl FSM
├── scripts/
│   ├── setup.sh                 # fetch upstream + apply patches + arch overlay
│   └── smoke_test_wbt.cpp       # offline ONNX + motion-loader sanity check
├── unitree_mujoco/         # cloned by setup.sh at a pinned SHA
├── unitree_rl_lab/         # cloned by setup.sh at a pinned SHA
└── unitree_sdk2/           # cloned by setup.sh at a pinned SHA
```

## Run on the real G1

Robot on a safety rig, human on the controller, NIC wired to the G1.

```bash
cd ~/lxf/deploy_ws/unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network enP8p1s0     # Jetson Orin NX NIC; use `ip -br link` to find yours
```

Controller transitions (trigger names match `config/config.yaml` — `LT` is L2,
`RB` is R1):

| From               | Gesture               | To                          |
| ------------------ | --------------------- | --------------------------- |
| Passive            | `LT + Up`             | FixStand                    |
| FixStand           | `RB + X`              | Velocity                    |
| Velocity           | `LT(2s) + Down`       | Mimic_Dance_102             |
| Velocity           | `LT(2s) + Left`       | Mimic_Gangnam_Style         |
| Velocity           | `LT(2s) + Right`      | **Mimic_WBT_Dance1** (ours) |
| Mimic_WBT_Dance1   | `RB + X`              | FixStand (clean exit)       |
| any RL state       | `LT + B`              | Passive (emergency)         |

`LT(2s)` = hold `LT` ≥2 s **and keep holding** while pressing the direction
key. Releasing `LT` resets the timer.

Mimic_WBT_Dance1 also auto-falls back to `FixStand` when the clip ends and to
`Passive` on bad orientation — same safety nets as upstream `State_Mimic`.

## Redeploy a new training run

Drop new files into `assets/` (paths in `config.yaml` resolve here):

```
assets/policy.onnx          # new actor
assets/dance1.bin           # new motion clip
assets/deploy_config.yaml   # obs_layout, gains, joint_ids_map — must match training
```

No rebuild needed. `g1_ctrl` reads these at FSM-entry into `Mimic_WBT_Dance1`.

## First-time build

```bash
# 0. system deps (one time)
sudo apt update
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev \
                    libspdlog-dev libfmt-dev curl file \
                    build-essential cmake git

# 1. clone
git clone https://github.com/LeeLeno/deploy_ws.git
cd deploy_ws

# 2. fetch upstream at pinned SHAs + apply patches
#    aarch64 hosts auto-overlay onnxruntime-linux-aarch64-1.22.0
./scripts/setup.sh

# 3. install unitree_sdk2 system-wide (provides libunitree_sdk2.a + DDS headers)
cd unitree_sdk2 && mkdir -p build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install
cd ../..

# 4. build g1_ctrl (WBT is compiled in — no separate target)
cd unitree_rl_lab/deploy/robots/g1_29dof && mkdir -p build && cd build
cmake .. && make -j$(nproc)        # produces ./g1_ctrl
```

First build on NX: ~15–25 min. Incremental rebuilds after editing `State_WBT.*`
finish in seconds with `make -j$(nproc)` from the same `build/` dir.

## Rebuild after editing patched sources

`patches/*.patch` are applied into the upstream tree on-disk. To iterate:

```bash
cd unitree_rl_lab            # or unitree_mujoco
# edit files in place — they're tracked by the sub-repo's git
git diff > ../patches/unitree_rl_lab-<short-description>.patch   # regenerate when done

# rebuild
cd deploy/robots/g1_29dof/build && make -j$(nproc)
```

`setup.sh` is idempotent — re-run it any time to refresh upstream + reapply
patches. When the diff grows past a couple of files, switch to a personal fork
pinned as a submodule.

## Offline smoke test (optional)

Exercises the motion loader, the yaw-aligned `motion_anchor_ori_b` math, and a
forward pass through `policy.onnx` — no DDS, no robot.

```bash
g++ -std=c++17 -O2 \
    -I /usr/include/eigen3 \
    -I unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/include \
    scripts/smoke_test_wbt.cpp \
    unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0 \
    -o scripts/smoke_test_wbt

LD_LIBRARY_PATH=unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib \
./scripts/smoke_test_wbt
```

## Pinned versions

| Component       | Source                            | Commit / version |
| --------------- | --------------------------------- | ---------------- |
| unitree_mujoco  | unitreerobotics/unitree_mujoco    | `1a37b05`        |
| unitree_rl_lab  | unitreerobotics/unitree_rl_lab    | `4960b84`        |
| unitree_sdk2    | unitreerobotics/unitree_sdk2      | `6d03531`        |
| onnxruntime     | microsoft/onnxruntime release     | `v1.22.0` (x64 vendored by `unitree_rl_lab`; aarch64 overlaid by `setup.sh`) |

To bump a pin: edit the SHA in `scripts/setup.sh`, re-run setup, regenerate
any patch whose context shifted, and commit.

## Sim2Sim

See upstream `unitree_rl_lab/README.md` → "Deploy → Sim2Sim". Our
`unitree_mujoco-g1-scene.patch` already flips `simulate/config.yaml` to `g1` /
`scene_29dof.xml`.
