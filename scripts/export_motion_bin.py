#!/usr/bin/env python3
"""Convert a training motion NPZ into the State_WBT motion binary.

The binary keeps the legacy WBT layout (little-endian, frames-major float32
after a 16-byte header of four int32: fps, num_frames, num_joints,
num_bodies):

    joint_pos [F][J], joint_vel [F][J], body_pos_w [F][B][3],
    body_quat_w [F][B][4] (wxyz), body_lin_vel_w [F][B][3],
    body_ang_vel_w [F][B][3]

Unlike the old 14-tracked-body bins, ALL NPZ bodies are written unchanged
(column 0 is the MuJoCo world body), so the deploy config's
anchor_body_idx_in_motion / root_body_idx_in_motion index the file directly,
exactly as the training command and sim_prototype.py index the raw NPZ.
Values are copied bit-for-bit (float32); nothing is resampled or normalised.

Usage (from deploy_ws/):
    python3 scripts/export_motion_bin.py assets/dance1_subject2_part.npz \
        assets/dance1_subject2_part.bin
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
          "body_lin_vel_w", "body_ang_vel_w")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    with np.load(args.npz) as data:
        missing = [k for k in FIELDS if k not in data]
        if missing:
            raise SystemExit(f"{args.npz}: missing fields {missing}")
        arrays = {k: np.ascontiguousarray(data[k], dtype="<f4") for k in FIELDS}
        fps = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50

    nf, nj = arrays["joint_pos"].shape
    nb = arrays["body_pos_w"].shape[1]
    expected = {"joint_pos": (nf, nj), "joint_vel": (nf, nj),
                "body_pos_w": (nf, nb, 3), "body_quat_w": (nf, nb, 4),
                "body_lin_vel_w": (nf, nb, 3), "body_ang_vel_w": (nf, nb, 3)}
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise SystemExit(f"{name}: shape {arrays[name].shape} != {shape}")
        if not np.all(np.isfinite(arrays[name])):
            raise SystemExit(f"{name}: non-finite values")
    norm_err = np.abs(np.linalg.norm(arrays["body_quat_w"], axis=-1) - 1.0).max()
    if norm_err > 1e-3:
        raise SystemExit(f"body_quat_w is not unit-norm (max |n-1| = {norm_err:.2e})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as stream:
        stream.write(struct.pack("<iiii", fps, nf, nj, nb))
        for name in FIELDS:
            stream.write(arrays[name].tobytes(order="C"))
    print(f"wrote {args.out}: fps={fps} frames={nf} joints={nj} bodies={nb} "
          f"({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
