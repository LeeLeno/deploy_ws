# 17.5k 调参公平性与训练随机种子审计

本次只读取本机源码与训练产物，运行纯配置组合和 CPU RNG 检查；没有修改任何仓库源码、启动训练或访问实机。

## 结论

1. baseline 是 `max_iterations=60000` 训练中的 `model_17500.pt`，本轮候选是 `max_iterations=17500`。这项差异没有压缩前 17.5k 的 LR、熵温度、探索噪声、归一化或采样课程调度。调度统一使用独立参数或已完成的更新次数。
2. `algo.seed=1` 没有接到 Manager 环境独立的 `env.seed`。baseline 和全部五个现有候选的 `run_config.json` 都未设置 `env.seed`。环境因此使用 `secrets.randbits(63)` 创建独立 NumPy RNG。startup 质量/质心 DR 以及初始 episode counters 会在训练批次之间变化。
3. Manager 的 Torch RNG 则是另外一种问题：新建 `torch.Generator` 后没有接上 `actual_seed`，使用固定默认 seed，不能说它“每次随机取了不同 seed”。它没有跟随 `algo.seed` 或 `env.seed`。
4. 这些结果支持按固定评估初态和评估 DR 比较实际产物；不能将当前单次训练差异解释为 action-rate 系数的严格配对因果效应，也不能宣称“同 seed 所以同训练轨迹”。不能仅凭这些混杂因素判断 tracking 退化一定来自随机性。

## max_iterations 的用途与算法调度

以下路径相对 `/home/user/flashmimic/flashsac/FlashMimic/.venv/lib/python3.11/site-packages/`。

| 检查项 | 代码证据 | 判断 |
|---|---|---|
| 入口传参 | `unilab/scripts/train_offpolicy.py:704–708` | `max_iterations` 只传给 `runner.learn`，不传 learner 或环境 |
| 主循环和结束 | `uni_rl/offpolicy/double_buffer_runner.py:1796,1986–1995,2302–2308,2339–2345` | 限制循环、最后一步不再准备下一批、退出及最终 checkpoint 命名；另用于 logger 进度/ETA |
| LR | `uni_rl/algos/flash_sac/learner.py:334–344`，`update.py:10–28` | 独立 warmup/decay 参数；当前 warmup=0、decay=140000，并非按 max_iterations 比例 |
| 更新次数 | `uni_rl/algos/flash_sac/learner.py:802–816,945–968` | actor/temperature 按 optimizer 更新推进；同一更新次数对应同一 LR。当前 updates_per_step=8，因此 17.5k 外层迭代约对应 140k optimizer 更新，并非 17.5k 个 LR step |
| 温度/熵 | `uni_rl/algos/flash_sac/update.py:78–87`；`learner.py:295–299,806–816` | target entropy 由 action_dim 和固定 sigma=0.15 决定，温度由数据/梯度学习，无训练总长度调度 |
| 探索噪声 | `uni_rl/algos/flash_sac/network.py:55–59,125–128,140–164` | 固定 zeta 分布参数和 episode done 触发的重采样，无 max_iterations 退火 |
| 归一化 | `uni_rl/algos/flash_sac/learner.py:301–311`，`uni_rl/offpolicy/runner.py:54–85` | obs normalization=false；reward normalization 根据已收集 rewards/dones 在线更新，不依赖训练总长度 |
| replay 采样 | `uni_rl/algos/flash_sac/double_buffer.py:187–191`；`replay.py:18–27,53–60` | 独立 decay_step，当前缺省=0，使用 uniform replay；没有根据 max_iterations 的课程 |
| replay RNG | `uni_rl/ipc/replay_pipelines/gpu_resident.py:488–490` | 每 tick 使用 base_seed+tick_id；最后一步是否准备下一批不会修改当前 batch 的采样 seed |
| 启训阈值 | `uni_rl/offpolicy/runner.py:19–35` | batch_size / learning_starts / num_envs 决定，不依赖训练总长度 |
| motion sampling | `unilab/tasks/motion_tracking/common/manager_terms.py:1263–1268` | 根据 motion sampler 和失败统计使用 Manager Torch RNG；环境未配置额外 curriculum，无总训练长度输入 |

最后一步是否预取下一批可能改变收尾 GPU 调度时间，但没有改变当前更新的 LR、损失或参数；不足以作为一个隐藏的“短训练末期退火”解释。

## 实际 seed 接线

父进程在 `unilab/scripts/train_offpolicy.py:660` 和 `uni_rl/algos/flash_sac/double_buffer.py:66` 设置了算法全局随机种子。collector 在 `uni_rl/offpolicy/worker.py:327` 只调用 `apply_training_seed(seed, torch_runtime=False, cuda=False)`，随后第 337 行构造环境。

collector 传入的是派生 worker seed（`double_buffer_runner.py:1744`，`utils/seed.py:57–63`）。但是它只设置 Python `random` 和 NumPy **全局** RNG；没有写入 `env_cfg_override['seed']`，也不能影响 `np.random.default_rng(secrets_seed)`。`_collector_env_cfg_override()` 在 `double_buffer_runner.py:1036–1055` 没有 seed 接线。

`unilab/envs/manager_based_rl_env.py:111` 的 `seed` 缺省为 None，第 344–349 行分别建立：

```python
actual_seed = cfg.seed if cfg.seed is not None else secrets.randbits(63)
self.rng = np.random.default_rng(actual_seed)
self._torch_rng_owner = TorchManagerRng(device=self.device)
```

两个流影响的路径不同：

| RNG 流 | 当前影响路径 | 代码证据 |
|---|---|---|
| `env.rng`，secrets seed | startup body_mass，pelvis/torso 质量 0.9–1.1 倍 | `unilab/envs/mdp/events.py:1022–1041,1127–1129` |
| `env.rng`，secrets seed | startup body_com，pelvis/torso 质心偏置 | `unilab/envs/mdp/events.py:1320–1345,1414–1416` |
| `env.rng`，secrets seed | 初始 episode counters | `unilab/envs/manager_based_rl_env.py:1103–1114` |
| `env.torch_rng`，固定默认 seed | joint_default startup bias | `src/flashmimic/tasks/motion_tracking/motion_command.py:140–146` |
| `env.torch_rng` | reset friction / PD gains | `unilab/envs/mdp/events.py:610–624,864–884` |
| `env.torch_rng` | motion frame、reset pose/velocity/joint noise | `unilab/tasks/motion_tracking/common/manager_terms.py:1263–1305` |
| `env.torch_rng` | interval 时间与 push 幅值 | `unilab/managers/event_manager.py:309–313`；`src/flashmimic/tasks/motion_tracking/events.py:76–86` |
| `env.torch_rng` | Torch actor/critic observation uniform noise | `unilab/managers/observation_manager.py:160–166,564–566`；`unilab/managers/_noise/noise_cfg.py:156–161` |
| learner 全局 Torch RNG | 参数初始化、随机策略与探索噪声 | `uni_rl/algos/flash_sac/double_buffer.py:66`，`network.py:125–128,152–162` |

startup body_mass/body_com 在 CUDA 后端也先走 `_startup` 的 host 路径，不能根据 `_apply_device` 函数的存在误判它们使用 Torch RNG。startup 值被缓存，不会在每次 reset 重新抽样。

`TorchManagerRng.__init__` 在 `unilab/managers/torch_rng.py:84–100` 只创建 `torch.Generator`；`manual_seed` 在 106–110 行，环境构造期间未调用。CPU 纯数值核验：先后设置全局 `torch.manual_seed(1/2/999)`，每次新建独立 Generator 的 `initial_seed()` 均为 `67280421310721`，前三个样本均为 `[0.26729524, 0.87252831, 0.33528882]`。这里直接数值验证了 CPU 行为，未为审计初始化 CUDA；CUDA 构造代码同样没有显式 seed 接线。

尽管 Torch 流的初始 seed 固定，质量/质心和 episode counters 的差异会改变终止时机及 selected reset 行数，随后随机抽样消费顺序也会分叉。因此不能将固定默认 Torch seed 等同于训练轨迹相同。

## 后续可复现性建议（本轮不实施）

不改代码的最小补充 CLI 是为**所有新比较组，包括新的 -0.20 对照**显式加上：

```bash
algo.seed=1 +env.seed=1
```

已通过纯 Hydra compose。这能固定 host startup mass/COM 与初始 counters，也使同一构造流程下的 Torch 默认流起点一致；它还没有让 Torch Manager 流跟随配置 seed，不构成跨设备位级确定性保证。原 baseline 的真实 env secrets seed 未记录在其 `run_config.json`，不能通过补一个 CLI 恢复它当时的 startup 表。

训练框架的完整修复应在环境构造时、`_load_managers()` 之前把 `actual_seed` 同时用于两个独立 RNG，例如将 Manager RNG 改为 `TorchManagerRng.seeded(actual_seed, device=self.device)`；再由 env adapter 将配置/派生的 worker seed 明确传入 `env.seed`，记录 effective environment seed 与 startup mass/COM 摘要。只在构造结束后调用 `env.seed()` 太晚：它不能替换已经缓存的 startup mass/COM，且 interval 初始化可能已消费随机数。

本轮保持训练代码和 RNG 行为不变，跑完现有候选，以固定评估初态/DR 的测量选实际产物；评价结果描述为“这次训练产物的观测差异”，不外推成已证明的系数因果效果。
