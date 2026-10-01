#!/usr/bin/env python3
"""Offline closed-loop evaluation of the deployed SONICMimic chain in MuJoCo.

No DDS, no robot.  The policy step is the UniLab tmp/sim_prototype.py chain
(compute_segments / ObsAssembler / ONNX, bit-matched to the C++ WBTPolicy by
run_parity.sh) plus the two deploy-side pieces WBTPolicy adds:
  * torso quat rebuilt from the pelvis IMU quat and the *measured* waist
    encoders (instead of MuJoCo's true torso xquat), and
  * yaw alignment init = yaw(torso) * yaw(ref anchor @ start)^-1 at begin.
Both are fed to the unmodified prototype via
    robot_anchor_quat := init^-1 * torso_quat
so that R(robot_anchor^-1 * ref) == R(torso^-1 * init * ref) (WBTPolicy).

Scenarios combine: start (rsi = training reset; stand / stand_vel = upright
real-robot entry proxy in the default / Velocity-policy stance), heading offset, training DR (PD gains, foot
friction, pelvis mass/CoM, encoder bias), actor observation noise, training
pushes, and actuation latency (whole physics substeps, 6.67 ms each; training
has none).  --dump writes C++ replay traces (trace.bin / ref.bin, same format
as sim_reference.py) for chosen scenarios.
"""
from __future__ import annotations

import argparse
import copy
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim_reference import DEPLOY_WS, UNILAB, import_prototype

EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_yaw_link", "right_wrist_yaw_link")


@dataclass
class Scenario:
    name: str
    start: str = "rsi"          # rsi | stand | stand_vel
    yaw_deg: float = 0.0        # robot heading offset at start (stand only)
    dr: bool = False
    noise: bool = False
    push: bool = False
    latency: int = 0            # substeps the new target is delayed by
    seeds: int = 1


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dtype=np.float32)


def axis_quat(axis, angle):
    h = 0.5 * float(angle)
    q = np.zeros(4, np.float32)
    q[0] = np.cos(h)
    q[1 + axis] = np.sin(h)
    return q


def yaw_quat(q):
    w, x, y, z = (float(v) for v in q)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], np.float32)


def conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], np.float32)


def gravity_z(q):
    """z component of world gravity (0,0,-1) expressed in frame q."""
    w, x, y, z = q
    return -(1.0 - 2.0 * (x * x + y * y))


class Evaluator:
    def __init__(self, args):
        import onnxruntime as ort
        self.proto = import_prototype(args.unilab)
        self.cfg = yaml.safe_load(args.config.read_text())
        self.motion = self.proto.load_motion(args.motion)
        self.session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        g1_dir = args.unilab / "src/unilab/assets/robots/g1"
        scene = g1_dir / "scene_flat.xml"
        if args.meshdir is not None:
            staged = args.out / "mjcf"
            staged.mkdir(parents=True, exist_ok=True)
            for name in ("scene_flat.xml", "g1.xml"):
                (staged / name).write_text((g1_dir / name).read_text())
            link = staged / "assets"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(args.meshdir.resolve())
            scene = staged / "scene_flat.xml"
        self.model0 = mujoco.MjModel.from_xml_path(str(scene))
        m = self.model0
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.pelvis = body("pelvis")
        self.torso = body(self.cfg["anchor_body_name"])
        self.ee = [body(n) for n in EE_BODIES]
        self.gyro_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "pelvis_gyro")])
        self.key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.foot_geoms = [i for i in range(m.ngeom)
                           if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(("left_foot", "right_foot"))
                           and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_collision")]
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        vel = yaml.safe_load((DEPLOY_WS / "unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/"
                              "params/deploy.yaml").read_text())
        self.vel_stance = np.zeros(len(vel["joint_ids_map"]), np.float32)
        for i, motor in enumerate(vel["joint_ids_map"]):
            self.vel_stance[int(motor)] = vel["default_joint_pos"][i]   # policy order -> SDK order
        self.substeps = int(round(float(self.cfg["ctrl_dt"]) / float(m.opt.timestep)))
        self.default = np.asarray(self.cfg["default_angles"], np.float32)
        scale_cfg = self.cfg["action_scale"]
        self.scale = np.asarray(scale_cfg if isinstance(scale_cfg, list) else [scale_cfg] * len(self.default), np.float32)
        if self.scale.shape != self.default.shape:
            raise ValueError("action_scale must be scalar or action_dim-sized list")
        clip_cfg = self.cfg.get("final_target_clip")
        self.final_clip = None if clip_cfg is None else np.asarray(clip_cfg, np.float32)
        if self.final_clip is not None and self.final_clip.shape != (len(self.default), 2):
            raise ValueError("final_target_clip must have shape [action_dim, 2]")
        self.lower = np.asarray(self.cfg["joint_lower"], np.float32)
        self.upper = np.asarray(self.cfg["joint_upper"], np.float32)
        self.anchor_idx = int(self.cfg["anchor_body_idx_in_motion"])
        self.root_idx = int(self.cfg["root_body_idx_in_motion"])
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        self.ref_ee = [names.index(n) for n in EE_BODIES]   # NPZ bodies == MJCF bodies (31)

    def run(self, sc: Scenario, seed: int, dump: bool = False):
        rng = np.random.default_rng(seed)
        model = self._fresh_model()
        data = mujoco.MjData(model)
        J = len(self.default)
        bias = np.zeros(J, np.float32)
        if sc.dr:
            kp = rng.uniform(0.9, 1.1, model.nu)
            kd = rng.uniform(0.85, 1.15, model.nu)
            model.actuator_gainprm[:, 0] *= kp
            model.actuator_biasprm[:, 1] *= kp
            model.actuator_biasprm[:, 2] *= kd
            model.geom_friction[self.foot_geoms, 0] = rng.uniform(0.3, 1.2)
            model.body_mass[self.pelvis] *= rng.uniform(0.8, 1.3)
            model.body_ipos[self.pelvis] += [rng.uniform(-0.025, 0.025), rng.uniform(-0.05, 0.05),
                                             rng.uniform(-0.05, 0.05)]
            bias = rng.uniform(-0.01, 0.01, J).astype(np.float32)

        mo = self.motion
        f0, last = 0, int(mo["num_frames"]) - 1
        mujoco.mj_resetDataKeyframe(model, data, self.key)
        if sc.start == "rsi":
            data.qpos[0:3] = mo["body_pos_w"][f0, self.root_idx]
            data.qpos[3:7] = mo["body_quat_w"][f0, self.root_idx]
            data.qpos[7:] = mo["joint_pos"][f0]
            data.qvel[0:3] = mo["body_lin_vel_w"][f0, self.root_idx]
            data.qvel[3:6] = mo["body_ang_vel_w"][f0, self.root_idx]
            data.qvel[6:] = mo["joint_vel"][f0]
        else:
            data.qpos[3:7] = axis_quat(2, np.radians(sc.yaw_deg))
        mujoco.mj_forward(model, data)
        if sc.start in ("stand", "stand_vel"):
            # Real-robot entry proxy: upright, zero velocity, feet resting on the
            # floor, in the training default pose ("stand") or the Velocity
            # policy's nominal stance ("stand_vel", the state WBT is entered
            # from).  No PD settle: a PD-only G1 is not statically stable in
            # MuJoCo (it tips within ~2 s), unlike the balancing Velocity policy.
            data.qpos[7:] = self.default if sc.start == "stand" else self.vel_stance
            data.qvel[:] = 0.0
            data.qpos[2] = 1.0
            mujoco.mj_forward(model, data)
            gap = min(mujoco.mj_geomDistance(model, data, self.floor, g, 1.0, None) for g in self.foot_geoms)
            data.qpos[2] -= gap
            mujoco.mj_forward(model, data)
        assembler = self.proto.ObsAssembler(self.cfg)
        last_action = np.zeros(J, np.float32)
        q_target = (np.asarray(data.qpos[7:], np.float32).copy() if sc.start.startswith("stand")
                    else mo["joint_pos"][f0].astype(np.float32))
        init = None
        sensors, refs = [], []
        stats = dict(fail_frame=None, fail_reason=None, jerr=[], zerr=[], tilt=[], arate=[],
                     oob=0, sat=0, n=0, q_oob=np.zeros(J))
        prev_action = None
        push_every = int(round(4.0 / float(self.cfg["ctrl_dt"])))
        forcerange = model.actuator_forcerange.copy()

        for step in range(last - f0 + 1):
            frame = f0 + step
            root_q = np.asarray(data.xquat[self.pelvis], np.float32)
            gyro = np.asarray(data.sensordata[self.gyro_adr:self.gyro_adr + 3], np.float32)
            q = np.asarray(data.qpos[7:], np.float32) + bias
            dq = np.asarray(data.qvel[6:], np.float32)
            if sc.noise:
                gyro = gyro + rng.uniform(-0.2, 0.2, 3).astype(np.float32)
                q = q + rng.uniform(-0.01, 0.01, J).astype(np.float32)
                dq = dq + rng.uniform(-0.5, 0.5, J).astype(np.float32)
            gyro, q, dq = gyro.astype(np.float32), q.astype(np.float32), dq.astype(np.float32)
            sensors.append(np.concatenate([root_q, gyro, q, dq]))

            torso_q = quat_mul(quat_mul(quat_mul(root_q, axis_quat(2, q[12])), axis_quat(0, q[13])),
                               axis_quat(1, q[14]))
            if init is None:
                init = quat_mul(yaw_quat(torso_q), conj(yaw_quat(mo["body_quat_w"][frame, self.anchor_idx])))
            anchor_for_proto = quat_mul(conj(init), torso_q)
            seg = self.proto.compute_segments(self.cfg, mo, frame, root_q, anchor_for_proto,
                                              gyro, q, dq, last_action)
            obs = assembler.push(seg)
            action = np.asarray(self.session.run([self.out_name], {self.in_name: obs[None]})[0][0], np.float32)
            last_action = action
            new_target = (action * self.scale + self.default).astype(np.float32)
            if self.final_clip is not None:
                new_target = np.clip(new_target, self.final_clip[:, 0], self.final_clip[:, 1])
            refs.append(np.concatenate([obs, action, new_target]))

            # metrics / safety counters
            stats["n"] += 1
            stats["oob"] += int(np.any((new_target < self.lower) | (new_target > self.upper)))
            if prev_action is not None:
                stats["arate"].append(float(np.abs(action - prev_action).mean()))
            prev_action = action

            if sc.push and step > 0 and step % push_every == 0:
                data.qvel[0] += rng.uniform(-0.18, 0.18)
                data.qvel[1] += rng.uniform(-0.18, 0.18)
                data.qvel[2] += rng.uniform(-0.072, 0.072)
            for k in range(self.substeps):
                data.ctrl[:] = q_target if k < sc.latency else new_target
                mujoco.mj_step(model, data)
                stats["sat"] += int(np.any(np.abs(data.actuator_force) >= forcerange[:, 1] * 0.999))
            q_target = new_target
            if sc.latency > self.substeps:
                raise SystemExit("latency > one control period is not modelled")
            mujoco.mj_forward(model, data)

            # training terminations, evaluated against the frame just tracked
            ref_torso_q = mo["body_quat_w"][frame, self.anchor_idx]
            rob_torso_q = np.asarray(data.xquat[self.torso])
            z_err = abs(float(data.xpos[self.torso, 2]) - float(mo["body_pos_w"][frame, self.anchor_idx, 2]))
            ori_err = abs(gravity_z(ref_torso_q) - gravity_z(rob_torso_q))
            ee_err = max(abs(float(data.xpos[b, 2]) - float(mo["body_pos_w"][frame, rb, 2]))
                         for b, rb in zip(self.ee, self.ref_ee))
            tilt = float(np.degrees(np.arccos(np.clip(-gravity_z(np.asarray(data.xquat[self.pelvis])), -1, 1))))
            q_true = np.asarray(data.qpos[7:])
            stats["q_oob"] = np.maximum(stats["q_oob"], np.maximum(q_true - self.upper, self.lower - q_true))
            stats["jerr"].append(float(np.abs(q_true - mo["joint_pos"][frame]).mean()))
            stats["zerr"].append(z_err)
            stats["tilt"].append(tilt)
            reason = ("anchor_z" if z_err > 0.5 else "anchor_ori" if ori_err > 0.8
                      else "ee_z" if ee_err > 0.5 else None)
            if tilt > np.degrees(1.0) or data.xpos[self.pelvis, 2] < 0.35:
                reason = "FALL(deploy bad_orientation/height)"
            if reason and stats["fail_frame"] is None:
                stats["fail_frame"], stats["fail_reason"] = frame, reason
            if reason and reason.startswith("FALL"):
                break
            if not np.all(np.isfinite(data.qpos)):
                stats["fail_frame"], stats["fail_reason"] = frame, "nonfinite"
                break

        n = stats["n"]
        out = dict(scenario=sc.name, seed=seed, steps=n, completed=(n == last - f0 + 1) and
                   not (stats["fail_reason"] or "").startswith(("FALL", "nonfinite")),
                   first_term_frame=stats["fail_frame"], first_term=stats["fail_reason"],
                   joint_err_mean=float(np.mean(stats["jerr"])),
                   joint_err_first25_mean=float(np.mean(stats["jerr"][:25])),
                   anchor_z_err_max=float(np.max(stats["zerr"])),
                   tilt_max_deg=float(np.max(stats["tilt"])),
                   action_rate_mean=float(np.mean(stats["arate"])) if stats["arate"] else 0.0,
                   q_target_oob_step_frac=stats["oob"] / n,
                   joint_pos_oob_max=float(stats["q_oob"].max()),
                   joint_pos_oob_joint=int(stats["q_oob"].argmax()),
                   torque_sat_substep_frac=stats["sat"] / (n * self.substeps))
        if dump:
            return out, (f0, np.asarray(sensors, np.float32), np.asarray(refs, np.float32))
        return out, None

    def _fresh_model(self):
        # DR mutates a copy only.
        return copy.deepcopy(self.model0)


SCENARIOS = [
    Scenario("rsi_nominal"),
    Scenario("rsi_dr_noise_push", dr=True, noise=True, push=True, seeds=16),
    Scenario("rsi_lat6.7ms", latency=1),
    Scenario("rsi_lat13.3ms", latency=2),
    Scenario("rsi_lat20ms", latency=3),
    Scenario("stand_nominal", start="stand"),
    Scenario("stand_yaw90_nominal", start="stand", yaw_deg=90.0),
    Scenario("stand_dr_noise_push_lat6.7ms", start="stand", dr=True, noise=True, push=True, latency=1, seeds=16),
    Scenario("standvel_nominal", start="stand_vel"),
    Scenario("standvel_lat13.3ms", start="stand_vel", latency=2),
    Scenario("standvel_dr_noise_push", start="stand_vel", dr=True, noise=True, push=True, seeds=16),
    Scenario("standvel_dr_noise_push_lat6.7ms", start="stand_vel", dr=True, noise=True, push=True, latency=1, seeds=16),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unilab", type=Path, default=UNILAB)
    p.add_argument("--config", type=Path, default=DEPLOY_WS / "assets/sonicmimic_dr_ankle_limit_140k_deploy_config.yaml")
    p.add_argument("--onnx", type=Path, default=DEPLOY_WS / "assets/policy.onnx")
    p.add_argument("--motion", type=Path, default=DEPLOY_WS / "assets/dance1_subject2_part.npz")
    p.add_argument("--meshdir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("/tmp/wbt_eval"))
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--dump", nargs="*", default=["stand_yaw90_nominal", "standvel_dr_noise_push_lat6.7ms"],
                   help="scenarios (seed 0) to write as C++ replay traces")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(args)

    results = []
    for sc in SCENARIOS:
        if args.only and sc.name not in args.only:
            continue
        for seed in range(sc.seeds):
            dump = seed == 0 and sc.name in args.dump
            res, tr = ev.run(sc, seed, dump)
            results.append(res)
            print(json.dumps(res), flush=True)
            if tr is not None:
                d = args.out / f"replay_{sc.name}"
                d.mkdir(parents=True, exist_ok=True)
                start, sens, rows = tr
                with (d / "trace.bin").open("wb") as t, (d / "ref.bin").open("wb") as r:
                    t.write(struct.pack("<i", 1))
                    t.write(struct.pack("<ii", start, len(sens)))
                    t.write(sens.astype("<f4").tobytes())
                    r.write(rows.astype("<f4").tobytes())
    (args.out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
