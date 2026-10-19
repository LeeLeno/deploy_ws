"""Compare deterministic policies under identical seeded deployment DR.

Errors are sampled after physics, before reference advancement and autoreset.
The full-clip protocol starts at frame zero and scores only the first episode;
uniform samples motion phases independently of each policy's failure curriculum.
By default neither protocol disables observation corruption, action latency, or
any DR. The optional noise-free actor observation control retains all DR,
physics, and action latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from unilab.base.config_adapter import create_env
from unilab.base.process_device import configure_backend_process_device
from unilab.scripts.train_offpolicy import build_offpolicy_env_cfg_override
from unilab.training import ensure_registries
from unilab.visualization.interactive_playback import (
    OffPolicyPlaybackSession,
    build_play_actor,
    extract_play_obs,
    load_play_actor,
    resolve_play_obs_dims,
)

from flashmimic import train as _train  # noqa: F401 -- registers asset resolvers
from flashmimic.reference_actor import reference_actor_context

METRICS = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos",
    "error_body_rot",
    "error_body_lin_vel",
    "error_body_ang_vel",
    "error_joint_pos",
    "error_joint_vel",
)
REPORTED_METRICS = (*METRICS, "action_delta_rms", "actuator_torque_rms_nm")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
ANKLE_JOINT_INDICES = (4, 5, 10, 11)


def _target_delta_statistics(values: np.ndarray) -> dict:
    """Pool absolute joint-command increments, retaining their physical units."""
    if not values.size:
        return {"scalar_samples": 0, "rms_rad": None, "p95_abs_rad": None, "max_abs_rad": None}
    absolute = np.abs(values)
    return {
        "scalar_samples": int(values.size),
        "rms_rad": float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))),
        "p95_abs_rad": float(np.quantile(absolute, 0.95)),
        "max_abs_rad": float(absolute.max()),
    }


def _deployment_diagnostic_statistics(samples: np.ndarray, joint_names: list[str]) -> dict:
    """Summarize valid transitions without hiding either foot in a body average.

    Columns are twelve physical target-command increments, two foot-height
    errors, then two foot vertical-speed errors. These are kinematic errors;
    no contact force or impact measurement is inferred from them.
    """
    count = int(samples.shape[0])
    target_delta = samples[:, :12]
    return {
        "valid_transitions": count,
        "target_delta": {
            "lower_body": {
                "joint_names": joint_names[:12],
                **_target_delta_statistics(target_delta),
            },
            "ankles": {
                "joint_names": [joint_names[i] for i in ANKLE_JOINT_INDICES],
                **_target_delta_statistics(target_delta[:, ANKLE_JOINT_INDICES]),
            },
            "by_joint": {name: _target_delta_statistics(target_delta[:, i]) for i, name in enumerate(joint_names[:12])},
        },
        "foot_tracking": {
            name: {
                "valid_transitions": count,
                "height_rmse_m": float(np.sqrt(np.mean(np.square(samples[:, 12 + i], dtype=np.float64))))
                if count
                else None,
                "vertical_speed_rmse_m_s": float(np.sqrt(np.mean(np.square(samples[:, 14 + i], dtype=np.float64))))
                if count
                else None,
            }
            for i, name in enumerate(FOOT_BODIES)
        },
    }


@contextmanager
def seeded_manager_generator(seed: int):
    """Seed device DR before startup events in pinned UniLab 1.3.4.

    That version seeds NumPy from cfg.seed but constructs its independent Torch
    generator without manual_seed. Calling env.seed after construction is too
    late for startup mass/COM DR, so scope the fix to evaluator construction.
    """
    from unilab.envs import manager_based_rl_env

    original = manager_based_rl_env.TorchManagerRng

    def factory(*, device):
        return original.seeded(seed, device=device)

    with patch.object(manager_based_rl_env, "TorchManagerRng", factory):
        yield


def evaluate(
    checkpoint: Path,
    *,
    seed: int,
    num_envs: int,
    steps: int,
    protocol: str,
    trajectory_dir: Path | None = None,
    disable_observation_noise: bool = False,
) -> dict:
    saved = json.loads((checkpoint.parent / "run_config.json").read_text())
    cfg = OmegaConf.create(saved["config"])
    original_env = OmegaConf.to_container(cfg.env, resolve=True)
    if disable_observation_noise:
        cfg.env.observations.actor.enable_corruption = False
    cfg.env.commands.motion.params.sampling_mode = "mixed"
    cfg.env.commands.motion.params.sampling_start_ratio = 1.0 if protocol == "full_clip" else 0.0
    if protocol == "full_clip":
        # The packaged clip is 17.4 s, longer than the training episode limit.
        cfg.env.max_episode_seconds = 1000.0
    evaluated_env = OmegaConf.to_container(cfg.env, resolve=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    configure_backend_process_device("mjwarp", "cuda:0")
    override = build_offpolicy_env_cfg_override("flashsac", cfg)
    override["seed"] = seed
    with seeded_manager_generator(seed):
        env = create_env(cfg, num_envs=num_envs, env_cfg_override=override)
    if env.torch_rng.generator.initial_seed() != seed or env._cfg.seed != seed:
        raise RuntimeError("Evaluation must seed both NumPy and the independent device DR generator")
    try:
        obs_dim, critic_dim = resolve_play_obs_dims(env.obs_groups_spec)
        with reference_actor_context(cfg):
            actor, normalizer, actor_type, _ = build_play_actor(
                "flashsac",
                cfg,
                obs_dim=obs_dim,
                critic_obs_dim=critic_dim,
                action_dim=int(env.action_space.shape[0]),
                device=str(env.device),
            )
        load_play_actor(
            "flashsac", actor, normalizer, torch.load(checkpoint, map_location=env.device, weights_only=True)
        )
        session = OffPolicyPlaybackSession(
            env=env,
            device=str(env.device),
            action_mode="policy",
            actor=actor,
            actor_algo_type=actor_type,
            normalizer=normalizer,
            num_envs=num_envs,
            obs_extractor=extract_play_obs,
        )
        command = env.command_manager.get_term("motion")
        if protocol == "full_clip":
            steps = command.motion.num_frames
        foot_ids = [list(command.cfg.body_names).index(name) for name in FOOT_BODIES]
        joint_names = list(cfg.env.scene.entities.robot.joint_names)
        if float(cfg.env.actions.joint_pos.scale) != 2.0:
            raise ValueError("Physical target-delta diagnostics require the G1 action scale of 2.0")
        active = torch.ones(num_envs, device=env.device, dtype=torch.bool)
        failed = torch.zeros_like(active)
        finished = torch.zeros_like(active)
        sums = torch.zeros((num_envs, len(REPORTED_METRICS)), device=env.device, dtype=torch.float64)
        counts = torch.zeros(num_envs, device=env.device, dtype=torch.float64)
        terminations = torch.zeros((), device=env.device, dtype=torch.int64)
        timeouts = torch.zeros_like(terminations)
        peak_torque = torch.zeros((), device=env.device)
        joint_squared_sums = torch.zeros(command.motion.num_joints, device=env.device, dtype=torch.float64)
        trajectory = None
        if trajectory_dir is not None and protocol == "full_clip":
            trajectory = torch.zeros((steps, 2 * command.motion.num_joints + 1), device=env.device)
        transition_index = 0
        failure_terms = [
            name
            for name in env.termination_manager.active_terms
            if not env.termination_manager.get_term_cfg(name).time_out
        ]
        failure_codes = torch.zeros((steps, num_envs), dtype=torch.int32, device=env.device)
        failure_frames = torch.zeros_like(failure_codes)
        failure_episode_steps = torch.zeros_like(failure_codes)
        # Keep a small device history so percentiles do not synchronize each step.
        # Existing aggregate metrics and their inclusion rules remain unchanged.
        diagnostic_history = torch.zeros((steps, num_envs, 16), device=env.device)
        diagnostic_valid = torch.zeros((steps, num_envs), dtype=torch.bool, device=env.device)
        original_compute = env.reward_manager.compute

        def record_transition(*args, **kwargs):
            nonlocal transition_index
            result = original_compute(*args, **kwargs)
            code = torch.zeros(num_envs, dtype=torch.int32, device=env.device)
            for bit, name in enumerate(failure_terms):
                code.bitwise_or_((env.termination_manager.get_term(name) & active).int() * (1 << bit))
            failure_codes[transition_index] = code
            failure_frames[transition_index] = command.time_steps
            failure_episode_steps[transition_index] = env.episode_length_buf
            command._refresh_robot_state()
            command._refresh_relative_state_torch()
            command._update_torch_error_metrics(command._tensor_all_rows)
            action_delta = env.action_manager.action - env.action_manager.prev_action
            diagnostic_valid[transition_index] = active & (env.episode_length_buf > 1)
            diagnostic_history[transition_index, :, :12] = 2.0 * action_delta[:, :12]
            diagnostic_history[transition_index, :, 12:14] = (
                command.robot_body_pos_w[:, foot_ids, 2] - command.body_pos_relative_w[:, foot_ids, 2]
            )
            diagnostic_history[transition_index, :, 14:16] = (
                command.robot_body_lin_vel_w[:, foot_ids, 2] - command.body_lin_vel_w[:, foot_ids, 2]
            )
            sensors = env.scene._tensor_read_plan.sensor_tensor_views(command.robot, command.torque_sensor_names).values
            torques = torch.cat([sensors[name] for name in command.torque_sensor_names], dim=-1)
            peak_torque.copy_(torch.maximum(peak_torque, torch.where(active[:, None], torques.abs(), 0.0).max()))
            values = torch.stack(
                [
                    *[command.metrics[name] for name in METRICS],
                    action_delta.square().mean(dim=-1).sqrt(),
                    torques.square().mean(dim=-1).sqrt(),
                ],
                dim=-1,
            )
            sums.add_(torch.where(active[:, None], values.double(), 0.0))
            counts.add_(active)
            joint_delta = command.joint_pos - command.device_robot_joint_pos
            joint_squared_sums.add_(torch.where(active[:, None], joint_delta.double().square(), 0.0).sum(0))
            if trajectory is not None:
                width = command.motion.num_joints
                trajectory[transition_index, :width] = torch.where(active[:, None], command.joint_pos, 0.0).sum(0)
                trajectory[transition_index, width : 2 * width] = torch.where(
                    active[:, None], command.device_robot_joint_pos, 0.0
                ).sum(0)
                trajectory[transition_index, -1] = active.sum()
            transition_index += 1
            return result

        env.reward_manager.compute = record_transition
        started = time.perf_counter()
        with torch.inference_mode():
            session.reset()
            initial_motion_frames = command.time_steps.detach().cpu().tolist()
            for _ in range(steps):
                session.step_once()
                state = env.state
                terminations.add_((state.terminated & active).sum())
                timeouts.add_((state.truncated & active & ~state.terminated).sum())
                failed.logical_or_(state.terminated & active)
                finished.logical_or_(state.truncated & active & ~state.terminated)
                if protocol == "full_clip":
                    active.logical_and_(~(state.terminated | state.truncated))
        torch.cuda.synchronize()
        count_array = counts.cpu().numpy()
        values = sums.cpu().numpy() / np.maximum(count_array[:, None], 1)
        result = {
            "evaluation_protocol_version": 2,
            "manager_generator_seed": env.torch_rng.generator.initial_seed(),
            "initial_motion_frames": initial_motion_frames,
            "checkpoint": str(checkpoint.resolve()),
            "seed": seed,
            "protocol": protocol,
            "num_envs": num_envs,
            "steps": steps,
            "evaluation_seconds": time.perf_counter() - started,
            "scored_transitions": int(count_array.sum()),
            "terminations": int(terminations.item()),
            "timeouts": int(timeouts.item()),
            "failed_environment_fraction": float(failed.float().mean().item()),
            "first_episode_completion_fraction": float(finished.float().mean().item())
            if protocol == "full_clip"
            else None,
            "training_env_sha256": hashlib.sha256(json.dumps(original_env, sort_keys=True).encode()).hexdigest(),
            "evaluation_env_sha256": hashlib.sha256(json.dumps(evaluated_env, sort_keys=True).encode()).hexdigest(),
            "observation_noise_disabled": disable_observation_noise,
            "actor_observation_corruption_enabled": bool(cfg.env.observations.actor.enable_corruption),
            "training_reward_sha256": hashlib.sha256(
                json.dumps(OmegaConf.to_container(cfg.reward, resolve=True), sort_keys=True).encode()
            ).hexdigest(),
            "peak_abs_actuator_torque_nm": float(peak_torque.item()),
            "metrics": {},
            "joint_position_rmse_by_name": dict(
                zip(
                    cfg.env.scene.entities.robot.joint_names,
                    (joint_squared_sums / counts.sum()).sqrt().cpu().tolist(),
                    strict=True,
                )
            ),
        }
        codes = failure_codes.cpu().numpy()
        frames = failure_frames.cpu().numpy()
        episode_steps = failure_episode_steps.cpu().numpy()
        diagnostic_samples = diagnostic_history.cpu().numpy()
        valid = diagnostic_valid.cpu().numpy()
        result["deployment_diagnostics"] = {
            "schema_version": 1,
            "sample_filter": "active and episode_length_buf > 1; after physics, before reference advance/autoreset",
            "target_delta_definition": "2 * (action[t] - action[t-1]), in radians per control step",
            "target_delta_latency_steps": 1,
            "target_delta_timing": "command increments applied on the following step by action latency",
            "control_dt_s": float(cfg.env.ctrl_dt),
            "all_valid_transitions": _deployment_diagnostic_statistics(diagnostic_samples[valid], joint_names),
        }
        if protocol == "full_clip":
            result["deployment_diagnostics"]["dynamic_frame_ge_100"] = _deployment_diagnostic_statistics(
                diagnostic_samples[valid & (frames >= 100)], joint_names
            )
        result["termination_events"] = [
            {
                "evaluation_step": int(step),
                "environment": int(environment),
                "motion_frame": int(frames[step, environment]),
                "episode_step": int(episode_steps[step, environment]),
                "terms": [name for bit, name in enumerate(failure_terms) if codes[step, environment] & (1 << bit)],
            }
            for step, environment in zip(*np.nonzero(codes), strict=True)
        ]
        if trajectory is not None:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            path = trajectory_dir / f"{checkpoint.parent.name}_{checkpoint.stem}_seed{seed}.npz"
            data = trajectory.cpu().numpy()
            width = command.motion.num_joints
            denominator = np.maximum(data[:, -1:], 1)
            np.savez_compressed(
                path,
                reference=data[:, :width] / denominator,
                measured=data[:, width : 2 * width] / denominator,
                active_envs=data[:, -1],
                joint_names=np.asarray(cfg.env.scene.entities.robot.joint_names),
                control_dt=float(cfg.env.ctrl_dt),
            )
            result["mean_joint_trajectory"] = str(path.resolve())
        for i, name in enumerate(REPORTED_METRICS):
            scale = math.sqrt(command.motion.num_joints) if name.startswith("error_joint_") else 1.0
            label = name.replace("error_joint_pos", "joint_rmse_rad").replace(
                "error_joint_vel", "joint_velocity_rmse_rad_s"
            )
            row = values[:, i] / scale
            result["metrics"][label] = {"mean": float(row.mean()), "sem": float(row.std(ddof=1) / math.sqrt(num_envs))}
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 456])
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--protocols", nargs="+", choices=["uniform", "full_clip"], default=["uniform", "full_clip"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument(
        "--disable-observation-noise",
        action="store_true",
        help="Disable actor observation corruption only; retain domain randomization, physics, and action latency",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    ensure_registries()
    results = []
    for checkpoint in args.checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for protocol in args.protocols:
            for seed in args.seeds:
                result = evaluate(
                    checkpoint,
                    seed=seed,
                    num_envs=args.num_envs,
                    steps=args.steps,
                    protocol=protocol,
                    trajectory_dir=args.trajectory_dir,
                    disable_observation_noise=args.disable_observation_noise,
                )
                results.append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
