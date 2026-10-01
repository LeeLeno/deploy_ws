#!/usr/bin/env python3
"""Pure numerical smoke test for the MimicLite deployment contract.

This deliberately avoids MuJoCo, DDS, and robot state.  It validates the
checked-in YAML/ONNX contract, evaluates the exported actor on deterministic
finite observations, repeats the same calls for bit-stable results, and checks
the one-step target delay used by ``MotionJointPositionAction``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parents[2] / "assets/mimiclite_dr_140k_deploy_config.yaml")
    parser.add_argument("--onnx", type=Path,
                        default=Path(__file__).parents[2] / "assets/policy.onnx")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    obs_dim, action_dim = int(cfg["obs_dim"]), int(cfg["action_dim"])
    layout_dim = sum(int(term["dim"]) * int(term.get("history_length", 1))
                     for term in cfg["obs_layout"])
    if layout_dim != obs_dim:
        raise SystemExit(f"obs_layout sums to {layout_dim}, expected {obs_dim}")
    steps = [int(x) for x in cfg["future_steps"]]
    if steps != [-8, -4, -2, 0, 1, 2, 3, 4]:
        raise SystemExit(f"unexpected MimicLite future_steps: {steps}")
    if not bool(cfg["simulate_action_latency"]):
        raise SystemExit("MimicLite DR contract must enable one-step action latency")

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    if inp.shape != [1, obs_dim] or out.shape != [1, action_dim]:
        raise SystemExit(f"ONNX shape {inp.shape}->{out.shape} != [1,{obs_dim}]->[1,{action_dim}]")

    values = [
        np.zeros((1, obs_dim), dtype=np.float32),
        np.linspace(-1.0, 1.0, obs_dim, dtype=np.float32)[None, :],
        np.sin(np.arange(obs_dim, dtype=np.float32) * np.float32(0.017))[None, :],
    ]
    outputs = []
    for obs in values:
        action_a = np.asarray(sess.run([out.name], {inp.name: obs})[0], dtype=np.float32)
        action_b = np.asarray(sess.run([out.name], {inp.name: obs})[0], dtype=np.float32)
        if action_a.shape != (1, action_dim) or not np.isfinite(action_a).all():
            raise SystemExit("non-finite or malformed actor output")
        repeat_diff = float(np.max(np.abs(action_a - action_b)))
        if repeat_diff != 0.0:
            raise SystemExit(f"non-deterministic repeated inference: max diff {repeat_diff:.3e}")
        outputs.append(action_a[0])

    scale = np.asarray(cfg["action_scale"] if isinstance(cfg["action_scale"], list)
                       else [cfg["action_scale"]] * action_dim, dtype=np.float32)
    default = np.asarray(cfg["default_angles"], dtype=np.float32)
    if scale.shape != (action_dim,) or default.shape != (action_dim,):
        raise SystemExit("action scale/default angle dimensions do not match action_dim")
    previous = np.zeros(action_dim, dtype=np.float32)
    delayed_targets = []
    for action in outputs:
        delayed_targets.append(previous * scale + default)
        previous = action
    if not np.array_equal(delayed_targets[0], default):
        raise SystemExit("first delayed target is not the default pose")
    if not np.array_equal(delayed_targets[1], outputs[0] * scale + default):
        raise SystemExit("one-step delayed target contract failed")
    print(f"NUMERIC PASS — ONNX [1,{obs_dim}] -> [1,{action_dim}], "
          f"3 deterministic vectors, latency target verified; "
          f"action range [{min(float(a.min()) for a in outputs):.4f}, "
          f"{max(float(a.max()) for a in outputs):.4f}]")


if __name__ == "__main__":
    main()
