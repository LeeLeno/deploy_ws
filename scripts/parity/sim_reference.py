#!/usr/bin/env python3
"""Reference side of the State_WBT offline parity test (no DDS, no robot).

Runs the SONICMimic deploy chain of UniLab tmp/sim_prototype.py (its own
load_motion / ObsAssembler / compute_segments, imported unmodified) in closed
loop in MuJoCo, and records per control step:

  trace.bin  sensors the real G1 provides, as float32: pelvis IMU quat (wxyz),
             gyro, joint q, joint dq  -> fed to the C++ WBTPolicy
  ref.bin    prototype obs (1570), raw action (29), q_target (29)

The prototype consumes the same float32 sensor values as C++ (only the torso
quaternion comes from MuJoCo's xquat, exactly as in the prototype; C++
reconstructs it from the IMU quat + waist joints).

Episodes (each a fresh prototype start: new ObsAssembler, zero last action):
  A  RSI at frame 0, runs through the last frame (reset, first frame, 10th
     frame, future clamp at the clip tail)
  B  RSI at frame 400, stopped after 60 steps (mid-clip exit)
  C  A's first 20 sensor rows again (re-entry after B; must reproduce A)

File layout (little-endian):
  trace.bin: int32 n_episodes, then per episode int32 start_frame, int32 steps,
             steps x 65 float32 [quat4, gyro3, q29, dq29]
  ref.bin / cpp.bin: per episode, steps x (obs_dim + 2*action_dim) float32
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import struct
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import yaml

DEPLOY_WS = Path(__file__).resolve().parents[2]
UNILAB = DEPLOY_WS.parent / "flashmimic_deploy"


def import_prototype(unilab: Path):
    # unilab/utils/__init__.py pulls in torch; the prototype only needs the
    # numpy-only unilab.utils.rotation, so register bare package shells.
    for name, sub in (("unilab", "src/unilab"), ("unilab.utils", "src/unilab/utils")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(unilab / sub)]
        sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location("sim_prototype", unilab / "tmp/sim_prototype.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_episode(proto, cfg, motion, session, model, data, ids, start_frame, steps):
    """Prototype main() loop body with RSI init; returns (sensor rows, ref rows)."""
    pelvis_id, anchor_id, gyro_adr, key_id = ids
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    root_idx = int(cfg["root_body_idx_in_motion"])
    data.qpos[0:3] = motion["body_pos_w"][start_frame, root_idx]
    data.qpos[3:7] = motion["body_quat_w"][start_frame, root_idx]
    data.qpos[7:] = motion["joint_pos"][start_frame]
    data.qvel[0:3] = motion["body_lin_vel_w"][start_frame, root_idx]
    data.qvel[3:6] = motion["body_ang_vel_w"][start_frame, root_idx]
    data.qvel[6:] = motion["joint_vel"][start_frame]
    mujoco.mj_forward(model, data)

    assembler = proto.ObsAssembler(cfg)
    default = np.asarray(cfg["default_angles"], dtype=np.float32)
    scale, ema = float(cfg["action_scale"]), float(cfg.get("ema_alpha", 1.0))
    substeps = max(1, int(round(float(cfg["ctrl_dt"]) / float(model.opt.timestep))))
    in_name, out_name = session.get_inputs()[0].name, session.get_outputs()[0].name
    last_action = np.zeros(int(cfg["action_dim"]), dtype=np.float32)
    q_target = default.copy()

    sensors, refs = [], []
    for step in range(steps):
        frame = start_frame + step
        root_quat = np.asarray(data.xquat[pelvis_id], dtype=np.float32)
        anchor_quat = np.asarray(data.xquat[anchor_id], dtype=np.float32)
        gyro = np.asarray(data.sensordata[gyro_adr:gyro_adr + 3], dtype=np.float32)
        q = np.asarray(data.qpos[7:], dtype=np.float32)
        dq = np.asarray(data.qvel[6:], dtype=np.float32)
        sensors.append(np.concatenate([root_quat, gyro, q, dq]))

        segments = proto.compute_segments(cfg, motion, frame, root_quat, anchor_quat,
                                          gyro, q, dq, last_action)
        obs = assembler.push(segments)
        action = np.asarray(session.run([out_name], {in_name: obs[None, :]})[0][0],
                            dtype=np.float32)
        if not np.all(np.isfinite(action)):
            raise SystemExit(f"non-finite action at frame {frame}")
        last_action = action
        q_target = ema * (action * scale + default) + (1.0 - ema) * q_target
        refs.append(np.concatenate([obs, action, q_target]).astype(np.float32))

        data.ctrl[:] = q_target
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        # mj_step leaves xquat/sensordata at the pre-integration state while
        # qpos/qvel are post-integration.  Re-sync (state unchanged) so the
        # IMU quat, gyro and encoders form one consistent sample, as on the
        # robot; C++ rebuilds the torso quat from IMU quat + waist q.
        mujoco.mj_forward(model, data)
    return np.asarray(sensors, np.float32), np.asarray(refs, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unilab", type=Path, default=UNILAB)
    parser.add_argument("--config", type=Path,
                        default=DEPLOY_WS / "assets/sonicmimic_dr_140k_deploy_config.yaml")
    parser.add_argument("--onnx", type=Path, default=DEPLOY_WS / "assets/policy.onnx")
    parser.add_argument("--motion", type=Path,
                        default=DEPLOY_WS / "assets/dance1_subject2_part.npz")
    parser.add_argument("--out", type=Path, default=Path("/tmp/wbt_parity"))
    parser.add_argument("--meshdir", type=Path, default=None,
                        help="G1 mesh directory when the UniLab checkout has no "
                             "robots/g1/assets (meshes are visual-only: density 0, no contact)")
    args = parser.parse_args()

    import onnxruntime as ort

    proto = import_prototype(args.unilab)
    cfg = yaml.safe_load(args.config.read_text())
    motion = proto.load_motion(args.motion)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    g1_dir = args.unilab / "src/unilab/assets/robots/g1"
    scene = g1_dir / "scene_flat.xml"
    if args.meshdir is not None:
        # Stage the unmodified MJCF next to the external meshes.
        staged = args.out / "mjcf"
        staged.mkdir(parents=True, exist_ok=True)
        for name in ("scene_flat.xml", "g1.xml"):
            shutil.copy(g1_dir / name, staged / name)
        link = staged / "assets"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(args.meshdir.resolve())
        scene = staged / "scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["anchor_body_name"])
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "pelvis_gyro")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    ids = (pelvis_id, anchor_id, int(model.sensor_adr[gyro_id]), key_id)

    last = int(motion["num_frames"]) - 1
    sens_a, ref_a = run_episode(proto, cfg, motion, session, model, data, ids, 0, last + 1)
    sens_b, ref_b = run_episode(proto, cfg, motion, session, model, data, ids, 400, 60)
    episodes = [(0, sens_a, ref_a), (400, sens_b, ref_b), (0, sens_a[:20], ref_a[:20])]

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "trace.bin").open("wb") as trace, (args.out / "ref.bin").open("wb") as ref:
        trace.write(struct.pack("<i", len(episodes)))
        for start, sens, rows in episodes:
            trace.write(struct.pack("<ii", start, len(sens)))
            trace.write(sens.astype("<f4").tobytes())
            ref.write(rows.astype("<f4").tobytes())

    # Closed-loop sanity: newest projected_gravity z of every step.
    tilt = np.degrees(np.arccos(np.clip(-ref_a[:, 1569], -1.0, 1.0))).max()
    print(f"A: {len(sens_a)} steps (frames 0..{last}), B: {len(sens_b)} steps from 400, "
          f"C: 20 steps; max pelvis tilt over A = {tilt:.1f} deg")
    print(f"wrote {args.out}/trace.bin, {args.out}/ref.bin")


if __name__ == "__main__":
    main()
