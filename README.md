# deploy_ws

Deployment-side workspace for **FlashMimic / UniLab** (SAC + WBT G1-29DOF dance). Adapts
Unitree's open-source stack (`unitree_rl_lab` + `unitree_mujoco` +
`unitree_sdk2`) to run WBT-trained policies on a real G1 over DDS via
`g1_ctrl`.

> The training-side repository is private — request access separately if
> you need to regenerate `assets/`.

## Layout

```
deploy_ws/
├── assets/                 # in-git runtime artifacts
│   ├── flashmimic_fast_17500/                 # active: gravity-fixed mjwarp_fast model_17500.pt
│   │   ├── policy.onnx + policy.onnx.data     # obs[1,397] → action[1,29], reference prior embedded
│   │   ├── deploy_config.yaml / dance1_subject2_part.bin
│   │   └── deployment_manifest.json          # source/checkpoint/asset hashes and policy contract
│   └── policy.onnx / mimiclite_dr_140k_deploy_config.yaml  # retained legacy MimicLite assets
├── patches/                # diffs against pinned upstream
│   ├── unitree_mujoco-g1-scene.patch    # sim2sim: default scene → G1
│   └── unitree_rl_lab-wbt-fsm.patch     # adds State_WBT to g1_ctrl FSM + --network NIC check
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
./g1_ctrl --network eth0         # G1 onboard Orin NX: eth0 = 192.168.123.164; check with `ip -br addr`
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

The active actor is exported from FlashMimic `main@d3990125` plus the
body-gravity sensor fix, checkpoint
`logs/flash_sac/G1MotionTracking/2026-10-19_22-42-11_mjwarp_gravity_fix_60k/model_17500.pt`.
`assets/flashmimic_fast_17500/export_manifest.json` records the checkpoint,
training XML, and ONNX hashes; `training_source.patch` records the training fix.
Its 397-float actor order is `last_action`, the three eight-frame reference
windows, `base_ang_vel`, `projected_gravity`, raw `joint_pos`, and raw
`joint_vel`.  The physical target uses the previous raw action for the one
control-step delay configured by training; observation noise is not injected
on the robot. The ONNX embeds the reference-action prior and the residual scale
of 0.25; do not add the prior again in deployment.

Training and deployment both use standard body-frame gravity
`R^T * [0, 0, -1]`. The earlier FlashMimic checkpoints trained with a world-frame
pelvis-axis observation require a different input convention and must not
replace this policy under the standard contract. No temporary
`gravity_observation` adaptation is enabled in this version.

The user reports the gravity-fixed 17.5k policy runs on the physical G1;
occasional uneven stepping and larger action changes remain under investigation.
This records a functional trial, not a claim that these remaining effects are fixed.

LowState has no global root-position sensor.  The deploy path anchors the
current reference pelvis at the measured robot pose and supplies future root
displacements in the projected-yaw frame for `ref_root_pos_future_local`.

## Redeploy a new training run

Replace the active files together (paths in the patched `config.yaml` resolve here):

```
assets/flashmimic_fast_17500/policy.onnx + policy.onnx.data
assets/flashmimic_fast_17500/dance1_subject2_part.bin
assets/flashmimic_fast_17500/deploy_config.yaml  # must match training
```

Paths and PD gains (`kp`/`kd`, = training MJCF actuator gains) live in
`config/config.yaml` → `Mimic_WBT_Dance1`.  No rebuild needed; `g1_ctrl` loads
them at start-up (State_WBT constructor). Changing C++ sources or applying a new
source patch requires rebuilding. The tracked patch includes the existing DIAG v2
CSV instrumentation (205 columns, per-row `fflush`, `fsync` every 25 rows), so its
diagnostic behavior is reproducible from a fresh checkout. The CSV is closed
after joining the policy thread to avoid a concurrent write during exit.

For the retained legacy MimicLite assets, the offline parity command is
`PYTHON=<py with onnxruntime>=1.18> [UNILAB=<UniLab checkout>] scripts/parity/run_parity.sh`;
it must print `PARITY PASS`. For the active FlashMimic assets,
the exporter checks ONNX against the checkpoint (64 inputs, maximum absolute
error `1.1623e-6`); the reconstruction hashes in `deployment_manifest.json`
also verify the tracked source patch and active runtime artifacts.

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



  cd /home/unitree/ws/mimiclite_dr/deploy_ws/unitree_rl_lab/deploy/robots/g1_29dof/build
  ./g1_ctrl --network eth0 --check-network   # 只检查网卡，不初始化 DDS、不发 lowcmd
  ./g1_ctrl --network eth0 --log
  - 机载 Orin NX 上连 G1 的网卡是 eth0（192.168.123.164）。enP8p1s0 是别的 Jetson 载板上的名字，这台机器上没有这张卡。不确定时用 ip -br addr 找 192.168.123.x 那张。
  - --log 会把日志追加写到 g1_29dof/log/log.txt。
  - g1_ctrl 已经有执行权限，不需要 chmod。
  - 机器人要挂在保护架上，并且有人手握遥控器。

  遥控器顺序（LT = L2，RB = R1）

  ┌──────────┬──────────────────────────────────────────────────────┬──────────────────────────────┐
  │ 当前状态 │                         按键                         │             进入             │
  ├──────────┼──────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 启动后   │ —                                                    │ Passive（阻尼，关节是软的）  │
  ├──────────┼──────────────────────────────────────────────────────┼──────────────────────────────┤
  │ Passive  │ LT + 上                                              │ FixStand（约 3 s 插值站起）  │
  ├──────────┼──────────────────────────────────────────────────────┼──────────────────────────────┤
  │ FixStand │ RB + X                                               │ Velocity（RL 站立平衡）      │
  ├──────────┼──────────────────────────────────────────────────────┼──────────────────────────────┤
  │ Velocity │ 按住 LT 不放 ≥2 s，再按 右                           │ Mimic_WBT_Dance1（开始跳舞） │
  ├──────────┼──────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 跳舞中   │ 跳完约 17.4 s 自动切换，或手动按 RB + X              │ Velocity                     │
  ├──────────┼──────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 任意状态 │ LT + B                                               │ Passive（急停）              │
  ├──────────┼──────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 结束     │ Velocity 下按 LT + B 进入 Passive，再在终端按 Ctrl+C │ —                            │
  └──────────┴──────────────────────────────────────────────────────┴──────────────────────────────┘

  - 在 Velocity 状态下，LT(2s) + 下 和 LT(2s) + 左 会进入 Unitree 自带的舞蹈，别按错方向。
  - 跳舞时如果姿态倾倒，或者策略输出非法值（NaN 等），会自动进入 Passive。

  加载的文件

  全部在启动时读入一次。改 yaml、onnx 或 bin 后只需要重启 g1_ctrl，不用重新编译。

  1. unitree_rl_lab/deploy/robots/g1_29dof/config/config.yaml：状态机和按键定义，以及 Mimic_WBT_Dance1 的 kp/kd、下面三个文件的路径、time_start: 0。
  2. assets/mimiclite_dr_140k_deploy_config.yaml：397 维 obs 的布局、action_scale 2.0、一步动作延迟、default_angles、参考帧偏移 [-8,-4,-2,0,1,2,3,4]。
  3. assets/policy.onnx 加 policy.onnx.data（权重）。
  4. assets/dance1_subject2_part.bin：870 帧，50 fps。

  舞蹈前后的 Velocity 用的是 Unitree 自带策略 config/policy/velocity/v0。

  数据流（每 20 ms 一步）

  LowState：骨盆 IMU 四元数 + 陀螺仪，29 个关节的 q / dq
   → 拼成 397 维 obs：
       上一步动作 29 | 参考根位移 24 | 参考根朝向 48 | 参考关节角 232
       （参考取当前帧的 −8, −4, −2, 0, +1, +2, +3, +4 帧）
       | 躯干角速度 3（IMU 陀螺仪 + 腰部关节速度推算）| 重力投影 3 | 原始 q 29 | 原始 dq 29
   → policy.onnx → 动作 a（29 维，tanh 输出，范围 ±1）
   → 目标角 = 上一步的 a × 2.0 + default_angles（延迟一步，不裁剪）
   → LowCmd 发给电机，电机端做 PD（kp/kd 来自 config.yaml）
   → 参考帧 +1；播到最后一帧后回到 Velocity

  进入跳舞时，参考动作的朝向会对齐到机器人当前的朝向，起点对齐到机器人当前的位置。所以不用把机器人摆到特定方向，但舞蹈会移动，周围要留出空间。

## 新手从零编译运行（最小步骤）

在 G1 机载 Orin NX（Ubuntu 20.04，aarch64）上执行，需要能上网（要 clone GitHub，aarch64 还要下载 onnxruntime）。

```bash
# 1. 系统依赖
sudo apt update
sudo apt install -y build-essential cmake git curl file \
    libyaml-cpp-dev libboost-program-options-dev libeigen3-dev libspdlog-dev libfmt-dev

# 2. 拉取本仓库，再拉 pinned 上游并打 patch
git clone -b feat/mimiclite-obs-deploy https://github.com/LeeLeno/deploy_ws.git
cd deploy_ws
./scripts/setup.sh

# 3. 安装 unitree_sdk2 自带的 CycloneDDS（g1_ctrl 编译和运行都要用 /usr/local 下的 libddsc）
cd unitree_sdk2 && mkdir -p build && cd build
cmake .. -DBUILD_EXAMPLES=OFF && sudo make install && sudo ldconfig
cd ../..

# 4. 编译 g1_ctrl（只编译，不会启动）
cd unitree_rl_lab/deploy/robots/g1_29dof && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# 5. 运行（机器人挂在保护架上，有人手握遥控器）
./g1_ctrl --network eth0 --check-network   # 只检查网卡就退出，不初始化 DDS；名字不对会列出可用网卡，192.168.123.x 那张就是 G1
./g1_ctrl --network eth0 --log
```

遥控器操作见上面的「遥控器顺序」。之后改了 `assets/` 或 `config.yaml`，只需要重启 g1_ctrl，不用重新编译。




 ssh unitree@192.168.110.18

  cd /home/unitree/ws/mimiclite_dr/deploy_ws/unitree_rl_lab/deploy/robots/g1_29dof/build
  ./g1_ctrl --network eth0 --log

  手柄操作：

  1. 启动后为 Passive
  2. LT + 上：进入 FixStand
  3. RB + X：进入 Velocity
  4. 在 Velocity 中，按住 LT ≥2 秒，再按方向键“右”：启动新的 dance
  5. 约 17.4 秒后自动回到 Velocity
  6. 手动退出：RB + X
  7. 紧急停止：任意状态按 LT + B，回到 Passive
  8. 结束后确认处于 Passive，终端按 Ctrl+C


  cd /home/unitree/ws/mimiclite_dr/deploy_ws/unitree_rl_lab/deploy/robots/g1_29dof/build
  ./g1_ctrl --network eth0 --log

  cd /home/unitree/ws/mimiclite_dr/deploy_ws/unitree_rl_lab/deploy/robots/g1_29dof/build
  ./g1_ctrl --network eth0 --log
