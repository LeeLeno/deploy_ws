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
│   ├── policy.onnx(+.data)                     #   G1SonicMimicDR 140k: obs[1,1570] → action[1,29]; weights in .data (same dir)
│   ├── dance1_subject2_part.npz / .bin         #   training motion; .bin = scripts/export_motion_bin.py(npz), all 31 bodies
│   └── sonicmimic_dr_140k_deploy_config.yaml   #   obs_layout, future_steps, default_angles, action_scale
├── patches/                # diffs against pinned upstream
│   ├── unitree_mujoco-g1-scene.patch    # sim2sim: default scene → G1
│   └── unitree_rl_lab-wbt-fsm.patch     # adds State_WBT to g1_ctrl FSM
├── scripts/
│   ├── setup.sh                 # fetch upstream + apply patches + arch overlay
│   ├── export_motion_bin.py     # training NPZ → State_WBT motion .bin
│   ├── setup_host_deps.sh       # rootless libfmt for the g1_ctrl build
│   └── parity/                  # offline parity / closed-loop sim / Velocity gate tests (no DDS)
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
| Mimic_WBT_Dance1   | `RB + X`              | Velocity (clean exit)       |
| any RL state       | `LT + B`              | Passive (emergency)         |

`LT(2s)` = hold `LT` ≥2 s **and keep holding** while pressing the direction
key. Releasing `LT` resets the timer.

Mimic_WBT_Dance1 also auto-falls back to `Velocity` when the clip ends and to
`Passive` on bad orientation or a policy fault (non-finite obs/action).

## Redeploy a new training run

Drop new files into `assets/` (paths in `config.yaml` resolve here):

```
assets/policy.onnx + policy.onnx.data            # new actor (keep together)
assets/<motion>.bin                              # python3 scripts/export_motion_bin.py <motion>.npz <motion>.bin
assets/sonicmimic_dr_*_deploy_config.yaml        # must match training
```

Paths and PD gains (`kp`/`kd`, = training MJCF actuator gains) live in
`config/config.yaml` → `Mimic_WBT_Dance1`.  No rebuild needed; `g1_ctrl` loads
them at start-up (State_WBT constructor).  Before any robot run, verify offline:
`PYTHON=<py with mujoco+onnxruntime>=1.18> MESHDIR=<g1 meshes> scripts/parity/run_parity.sh`
must print `PARITY PASS`.

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

# 4. build g1_ctrl (WBT is compiled in — no separate target).  Links the pinned
#    unitree_sdk2/lib/<arch>/libunitree_sdk2.a (not /usr/local/lib), so step 3 is
#    optional.  Without root, `./scripts/setup_host_deps.sh` unpacks libfmt-dev
#    into deploy_ws/.deps instead of `apt install libfmt-dev`.
cd unitree_rl_lab/deploy/robots/g1_29dof && mkdir -p build && cd build
cmake .. && make -j$(nproc)        # produces ./g1_ctrl (building does not start it)
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

## Offline numerical checks (no DDS, no robot)

```bash
# C++ WBTPolicy vs UniLab sim_prototype.py on the same sensor trace (tol 1e-4)
PYTHON=<python with mujoco, onnxruntime>=1.18> MESHDIR=<g1 mesh dir> scripts/parity/run_parity.sh
# Velocity state never publishes before its first inference
scripts/parity/run_rlbase_gate_test.sh
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
