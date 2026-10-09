# deploy_ws

Deployment-side workspace for **UniLab** (WarpSAC + WBT G1-29DOF dance). Adapts
Unitree's open-source stack (`unitree_rl_lab` + `unitree_mujoco` +
`unitree_sdk2`) to run WBT-trained policies on a real G1 over DDS via
`g1_ctrl`.

> The training-side `UniLab` repo is private — request access separately if
> you need to regenerate `assets/`.

## Layout

```
deploy_ws/
├── assets/                 # in-git runtime artifacts
│   ├── policy.onnx(+.data)                     #   MimicLite DR 140k: obs[1,397] → action[1,29]; weights in .data
│   ├── dance1_subject2_part.npz / .bin         #   training motion; .bin = scripts/export_motion_bin.py(npz), all 31 bodies
│   └── mimiclite_dr_140k_deploy_config.yaml     #   obs_layout, future_steps, default_angles, latency
├── patches/                # diffs against pinned upstream
│   ├── unitree_mujoco-g1-scene.patch    # sim2sim: default scene → G1
│   └── unitree_rl_lab-wbt-fsm.patch     # adds State_WBT to g1_ctrl FSM
├── scripts/
│   ├── setup.sh                 # fetch upstream + apply patches + arch overlay
│   ├── export_motion_bin.py     # training NPZ → State_WBT motion .bin
│   ├── setup_host_deps.sh       # rootless libfmt for the g1_ctrl build
│   └── parity/                  # offline parity / numeric / Velocity gate tests (no DDS)
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

## Current policy contract

The checked-in actor is exported from UniLab `feat/mimiclite-obs` commit
`35590827`, checkpoint
`logs/warp_sac/G1MotionTrackingSAC/2026-09-30_14-53-45_mjwarp/model_140000.pt`.
Its 397-float actor order is `last_action`, the three eight-frame reference
windows, `base_ang_vel`, `projected_gravity`, raw `joint_pos`, and raw
`joint_vel`.  The physical target uses the previous raw action for the one
control-step delay configured by training; observation noise is not injected
on the robot.

LowState has no global root-position sensor.  The deploy path anchors the
current reference pelvis at the measured robot pose and supplies future root
displacements in the projected-yaw frame for `ref_root_pos_future_local`.

## Redeploy a new training run

Drop new files into `assets/` (paths in `config.yaml` resolve here):

```
assets/policy.onnx + policy.onnx.data            # new actor (keep together)
assets/<motion>.bin                              # python3 scripts/export_motion_bin.py <motion>.npz <motion>.bin
assets/mimiclite_dr_*_deploy_config.yaml         # must match training
```

Paths and PD gains (`kp`/`kd`, = training MJCF actuator gains) live in
`config/config.yaml` → `Mimic_WBT_Dance1`.  No rebuild needed; `g1_ctrl` loads
them at start-up (State_WBT constructor).  Before any robot run, verify offline:
`PYTHON=<py with onnxruntime>=1.18> [UNILAB=<UniLab checkout>] scripts/parity/run_parity.sh`
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
# Pure numerical ONNX and one-step latency check
python3 scripts/parity/mimiclite_numeric_check.py
# On aarch64 host with the vendored ONNX Runtime (works even when the system
# Python onnxruntime is too old for this IR-10 export)
ORT=unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0
g++ -std=c++17 -O2 -I unitree_rl_lab/deploy/include -I "$ORT/include" \
  scripts/parity/mimiclite_numeric_check.cpp "$ORT/lib/libonnxruntime.so.1.22.0" \
  -lpthread -o /tmp/mimiclite_numeric_check
LD_LIBRARY_PATH="$ORT/lib" /tmp/mimiclite_numeric_check assets/policy.onnx
# MimicLite MuJoCo closed-loop smoke test (from a UniLab checkout)
uv run python tmp/sim_prototype_mimiclite.py --onnx <deploy_ws>/assets/policy.onnx \
  --config <deploy_ws>/assets/mimiclite_dr_140k_deploy_config.yaml --max-steps 20
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

  cd <deploy_ws>/unitree_rl_lab/deploy/robots/g1_29dof/build
  chmod u+x g1_ctrl                 # 我之前去掉了执行权限防误启动，这一步解除
  ./g1_ctrl --network eth0 --log    # --log 会写日志到 g1_29dof/log/log.txt
  

  启动命令：

  cd <deploy_ws>/unitree_rl_lab/deploy/robots/g1_29dof/build
  chmod u+x g1_ctrl
  ./g1_ctrl --network <实际网卡名> --log

  例如网卡是 eth0 时：

  ./g1_ctrl --network eth0 --log
