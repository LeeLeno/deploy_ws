#!/usr/bin/env python3
"""Reference side of the MimicLite State_WBT offline check (no DDS, no robot, no MuJoCo).

The training contract is UniLab ``feat/mimiclite-obs``
``warpsac/task/g1_motion_tracking/mjwarp_mimiclite_dr.yaml``; the actor terms
below are NumPy ports of ``unilab/tasks/motion_tracking/common/obs_terms.py``.

Sensor stream = the reference clip itself ("perfect tracking"): pelvis IMU quat
and gyro, joint q / dq taken from the training NPZ, with the IMU yaw rotated by
a per-episode offset (the real IMU yaw is arbitrary).  The reference obs is the
exact training observation in a world where robot and reference coincide, so a
correct deployment must reproduce it for ANY yaw offset.

Independent checks printed here:
  1. NPZ joint column order: MJCF forward kinematics from NPZ joint_pos vs the
     NPZ body poses (SDK/MJCF order vs the IsaacLab order, for contrast).
  2. torso_gyro: the deployment reconstruction (pelvis gyro + waist rates) vs
     the NPZ torso angular velocity, and the error of using the pelvis gyro.
  3. Policy output statistics on in-distribution input.

Writes trace.bin / ref.bin in the layout scripts/parity/wbt_parity.cpp reads.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

DEPLOY_WS = Path(__file__).resolve().parents[2]
ISAACLAB_ORDER = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
                  17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


# --- quaternion helpers (wxyz, float64) -------------------------------------

def qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qmat(q):
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def qaxis(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def projected_yaw_quat(q, thr=0.1):
    """obs_terms._projected_yaw_quat for one quaternion."""
    R = qmat(q)
    x, z = R[:, 0], R[:, 2]
    if np.linalg.norm(x[:2]) > thr:
        heading = x[:2]
    else:
        heading = z[:2] if x[2] < 0.0 else -z[:2]
    return yaw_quat(np.arctan2(heading[1], heading[0]))


def angle_between(qa, qb):
    d = np.abs(np.sum(qa * qb, axis=-1) / (np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1)))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


# --- MJCF forward kinematics (joint-order check) -----------------------------

def parse_bodies(xml_text):
    """Bodies in MuJoCo id order (DFS) as (name, parent, pos, quat, hinge axis or None)."""
    root = ET.fromstring(xml_text)
    out = []

    def walk(elem, parent):
        for b in elem.findall("body"):
            pos = np.array([float(v) for v in b.get("pos", "0 0 0").split()])
            quat = np.array([float(v) for v in b.get("quat", "1 0 0 0").split()])
            joints = [j for j in b.findall("joint") if j.get("type", "hinge") == "hinge"]
            axis = None if not joints else np.array([float(v) for v in joints[0].get("axis", "0 0 1").split()])
            idx = len(out) + 1   # MuJoCo body id (world = 0)
            out.append((b.get("name"), parent, pos, quat, axis))
            walk(b, idx)

    walk(root.find("worldbody"), 0)
    return out


def fk_errors(bodies, motion, joint_cols):
    """Max position / orientation error of MJCF FK vs the NPZ body poses."""
    pos_w, quat_w = motion["body_pos_w"].astype(float), motion["body_quat_w"].astype(float)
    q_all = motion["joint_pos"].astype(float)[:, joint_cols]
    F = q_all.shape[0]
    P = np.zeros((F, len(bodies) + 1, 3))
    Q = np.zeros((F, len(bodies) + 1, 4))
    Q[:, 0, 0] = 1.0
    j = 0
    for i, (_, parent, pos, quat, axis) in enumerate(bodies, start=1):
        if i == 1:   # free-floating pelvis: take the NPZ root pose
            P[:, 1], Q[:, 1] = pos_w[:, 1], quat_w[:, 1]
            continue
        P[:, i] = P[:, parent] + np.einsum("fij,j->fi", qmat(Q[:, parent]), pos)
        Q[:, i] = qmul(Q[:, parent], np.broadcast_to(quat, (F, 4)))
        if axis is not None:
            Q[:, i] = qmul(Q[:, i], np.stack([qaxis(axis, a) for a in q_all[:, j]]))
            j += 1
    n = len(bodies) + 1
    return (np.abs(P[:, 2:n] - pos_w[:, 2:n]).max(),
            np.degrees(angle_between(Q[:, 2:n], quat_w[:, 2:n]).max()))


# --- torso gyro --------------------------------------------------------------

def body_gyro(motion, f, col):
    """Body-frame angular velocity of NPZ body ``col`` at frame f (what a gyro on it reads)."""
    R = qmat(motion["body_quat_w"][f, col].astype(float))
    return R.T @ motion["body_ang_vel_w"][f, col].astype(float)


def torso_gyro_reconstructed(pelvis_gyro, q, dq):
    """unitree_articulation.h: pelvis IMU + waist yaw/roll/pitch (SDK motors 12/13/14)."""
    Ry = qmat(qaxis([0, 0, 1], q[12]))
    Rr = qmat(qaxis([1, 0, 0], q[13]))
    Rp = qmat(qaxis([0, 1, 0], q[14]))
    omega = (pelvis_gyro + np.array([0, 0, dq[12]]) + Ry @ np.array([dq[13], 0, 0])
             + Ry @ Rr @ np.array([0, dq[14], 0]))
    return (Ry @ Rr @ Rp).T @ omega


# --- episode -----------------------------------------------------------------

def run_episode(cfg, motion, sess, start, steps, yaw_offset, root_col, torso_col, stats):
    J = int(cfg["action_dim"])
    last = motion["joint_pos"].shape[0] - 1
    fsteps = np.asarray(cfg["future_steps"], int)
    scale = float(cfg["action_scale"])
    default = np.asarray(cfg["default_angles"], float)
    qz = yaw_quat(yaw_offset)
    Rz = qmat(qz)
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    handoff_steps = int(np.ceil(float(cfg.get("handoff_duration_s", 0.0)) /
                         float(cfg["ctrl_dt"])))
    entry_q = motion["joint_pos"][start].astype(np.float32)
    entry_dq = motion["joint_vel"][start].astype(np.float32)
    goal_q = entry_q.copy()
    goal_dq = entry_dq.copy()
    prev_action = ((entry_q.astype(float) - default) / scale).astype(np.float32)
    if handoff_steps == 0:
        prev_action.fill(0.0)
    sensors, rows = [], []
    for k in range(steps):
        # The hand-off holds the selected motion frame; the clip clock starts
        # advancing only after the transition has completed.
        f = start + max(0, k - handoff_steps)
        f = min(f, last)
        # sensors the G1 provides, as float32 (what C++ receives)
        imu_q = qmul(qz, motion["body_quat_w"][f, root_col].astype(float)).astype(np.float32)
        gyro = body_gyro(motion, f, root_col).astype(np.float32)
        q = motion["joint_pos"][f].astype(np.float32)
        dq = motion["joint_vel"][f].astype(np.float32)
        sensors.append(np.concatenate([imu_q, gyro, q, dq]))

        # training actor obs (obs_terms.py), robot == reference, world rotated by qz
        robot_q = imu_q.astype(float)
        fs = np.clip(f + fsteps, 0, last)
        ref_pos = (Rz @ motion["body_pos_w"][fs, root_col].astype(float).T).T
        robot_pos = Rz @ motion["body_pos_w"][f, root_col].astype(float)
        robot_yaw = projected_yaw_quat(robot_q)
        root_pos_local = (qmat(robot_yaw).T @ (ref_pos - robot_pos).T).T.reshape(-1)
        ref_q = qmul(qz, motion["body_quat_w"][fs, root_col].astype(float))
        rel = qmul(np.broadcast_to(qconj(robot_q), ref_q.shape), ref_q)
        root_ori = qmat(rel)[:, :2, :].reshape(-1)
        ref_joint = motion["joint_pos"][fs].astype(float).reshape(-1)
        torso_gyro = body_gyro(motion, f, torso_col)
        grav = qmat(robot_q).T @ np.array([0.0, 0.0, -1.0])
        segs = {
            "last_action": prev_action.astype(float),
            "ref_root_pos_future_local": root_pos_local,
            "ref_root_ori_future_b": root_ori,
            "ref_joint_pos_future": ref_joint,
            "base_ang_vel": torso_gyro,
            "projected_gravity": grav,
            "joint_pos": q.astype(float),
            "joint_vel": dq.astype(float),
        }
        obs = np.concatenate([segs[s["name"]] for s in cfg["obs_layout"]]).astype(np.float32)
        if k < handoff_steps:
            # Match WBTPolicy's minimum-jerk pre-positioning.  No actor
            # inference is consumed before the measured state reaches the
            # selected NPZ frame; its first inference happens at frame start.
            T = float(handoff_steps) * float(cfg["ctrl_dt"])
            t = min(T, float(k + 1) * float(cfg["ctrl_dt"]))
            s = np.clip(t / T, 0.0, 1.0) if T > 0.0 else 1.0
            s2, s3, s4, s5 = s*s, s*s*s, s*s*s*s, s*s*s*s*s
            h00 = 1.0 - 10.0*s3 + 15.0*s4 - 6.0*s5
            h10 = s - 6.0*s3 + 8.0*s4 - 3.0*s5
            h01 = 10.0*s3 - 15.0*s4 + 6.0*s5
            h11 = -4.0*s3 + 7.0*s4 - 3.0*s5
            q_target = (h00 * entry_q + h10 * T * entry_dq
                        + h01 * goal_q + h11 * T * goal_dq).astype(np.float32)
            action = ((q_target.astype(float) - default) / scale).astype(np.float32)
        else:
            action = sess.run([out_name], {in_name: obs[None]})[0][0].astype(np.float32)
            q_target = (prev_action * scale + default).astype(np.float32)   # one-step latency
        rows.append(np.concatenate([obs, action, q_target]))

        # contrast: the pre-fix deploy formula (displacement not rotated into the robot world)
        old = (qmat(robot_yaw).T @ (motion["body_pos_w"][fs, root_col].astype(float)
                                    - motion["body_pos_w"][f, root_col].astype(float)).T).T.reshape(-1)
        stats["old_pos_err"] = max(stats.get("old_pos_err", 0.0), float(np.abs(old - root_pos_local).max()))
        if abs(yaw_offset) > 1e-6 and k >= handoff_steps:
            obs_old = obs.copy()
            off = 0
            for s in cfg["obs_layout"]:
                if s["name"] == "ref_root_pos_future_local":
                    obs_old[off:off + s["dim"]] = old
                off += s["dim"]
            a_old = sess.run([out_name], {in_name: obs_old[None]})[0][0]
            stats["old_act_err"] = max(stats.get("old_act_err", 0.0), float(np.abs(a_old - action).max()))
        stats.setdefault("actions", []).append(action)
        stats.setdefault("targets_next", []).append(q_target)
        stats.setdefault("ref_q", []).append(q)
        stats.setdefault("gyro_rec_err", []).append(
            np.abs(torso_gyro_reconstructed(gyro.astype(float), q.astype(float), dq.astype(float)) - torso_gyro).max())
        stats.setdefault("gyro_pelvis_err", []).append(np.abs(gyro.astype(float) - torso_gyro).max())
        prev_action = action
    return np.asarray(sensors, np.float32), np.asarray(rows, np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEPLOY_WS / "assets/mimiclite_dr_140k_deploy_config.yaml")
    p.add_argument("--onnx", type=Path, default=DEPLOY_WS / "assets/policy.onnx")
    p.add_argument("--motion", type=Path, default=DEPLOY_WS / "assets/dance1_subject2_part.npz")
    p.add_argument("--unilab", type=Path, default=DEPLOY_WS.parent / "UniLab")
    p.add_argument("--unilab-ref", default="origin/feat/mimiclite-obs")
    p.add_argument("--out", type=Path, default=Path("/tmp/mimiclite_parity"))
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    motion = dict(np.load(args.motion))
    root_col = int(cfg["root_body_idx_in_motion"])
    torso_col = 16
    xml = subprocess.run(["git", "-C", str(args.unilab), "show",
                          f"{args.unilab_ref}:src/unilab/assets/robots/g1/g1.xml"],
                         check=True, capture_output=True, text=True).stdout
    bodies = parse_bodies(xml)
    names = [b[0] for b in bodies]
    print(f"MJCF bodies: {len(bodies)} (+world) vs NPZ columns {motion['body_pos_w'].shape[1]}; "
          f"col {root_col}={names[root_col - 1]}, col {torso_col}={names[torso_col - 1]}")

    # 1. joint column order
    e_pos, e_ang = fk_errors(bodies, motion, list(range(29)))
    i_pos, i_ang = fk_errors(bodies, motion, ISAACLAB_ORDER)
    print(f"[1] NPZ joint order, FK vs NPZ body poses over {motion['joint_pos'].shape[0]} frames:")
    print(f"    SDK/MJCF order : max pos err {e_pos:.2e} m, max ori err {e_ang:.2e} deg")
    print(f"    IsaacLab order : max pos err {i_pos:.2e} m, max ori err {i_ang:.2e} deg  (contrast)")

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    last = motion["joint_pos"].shape[0] - 1
    stats = {}
    plan = [(0, last + 1, 0.0), (400, 60, np.pi / 2), (200, 100, -2.8)]
    episodes = []
    for start, steps, yaw in plan:
        sens, rows = run_episode(cfg, motion, sess, start, steps, yaw, root_col, torso_col, stats)
        episodes.append((start, sens, rows))
        print(f"    episode start={start:3d} steps={steps:3d} imu_yaw_offset={np.degrees(yaw):+7.1f} deg")
    episodes.append((0, episodes[0][1][:20], episodes[0][2][:20]))   # re-entry replay of A

    # 2. torso gyro
    rec, pel = np.asarray(stats["gyro_rec_err"]), np.asarray(stats["gyro_pelvis_err"])
    print(f"[2] torso_gyro vs NPZ torso angular velocity: reconstruction max err {rec.max():.2e} rad/s, "
          f"p99 {np.percentile(rec, 99):.2e}; pelvis gyro instead: max err {pel.max():.2f} rad/s, "
          f"mean {pel.mean():.2f} (training noise +/-0.35)")

    # 3. policy output on in-distribution input
    act = np.asarray(stats["actions"])
    tgt = np.asarray(stats["targets_next"])
    refq = np.asarray(stats["ref_q"])
    lo, hi = np.asarray(cfg["joint_lower"]), np.asarray(cfg["joint_upper"])
    groups = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29)}
    print(f"[3] policy: |raw action| max {np.abs(act).max():.3f}, mean {np.abs(act).mean():.3f}; "
          f"action rate |a_t - a_t-1| p99 {np.percentile(np.abs(np.diff(act[:last + 1], axis=0)), 99):.3f}")
    print("    target - reference q (mean abs / max abs, rad): " + ", ".join(
        f"{g} {np.abs(tgt[:, s] - refq[:, s]).mean():.3f}/{np.abs(tgt[:, s] - refq[:, s]).max():.3f}"
        for g, s in groups.items()))
    print(f"    targets outside joint limits: {int(((tgt < lo) | (tgt > hi)).sum())} of {tgt.size}")
    print(f"    pre-fix ref_root_pos_future_local error with yaw offset: {stats['old_pos_err']:.3f} m "
          f"-> raw action change up to {stats.get('old_act_err', 0.0):.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "trace.bin").open("wb") as tr, (args.out / "ref.bin").open("wb") as rf:
        tr.write(struct.pack("<i", len(episodes)))
        for start, sens, rows in episodes:
            tr.write(struct.pack("<ii", start, len(sens)))
            tr.write(sens.astype("<f4").tobytes())
            rf.write(rows.astype("<f4").tobytes())
    print(f"wrote {args.out}/trace.bin, {args.out}/ref.bin")


if __name__ == "__main__":
    main()
