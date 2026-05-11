# deploy_ws

Deployment-side workspace for the **UniLab** training-side repository — adapts
Unitree's open-source robot stack (G1 + MuJoCo + SDK2) to consume policies
trained by the WBT + SAC pipeline in UniLab.

> Status: semi-development. The training-side `UniLab` repo is private; please
> request access separately.

## Layout

After running `scripts/setup.sh`, the workspace looks like:

```
deploy_ws/
├── assets/                              # policies, configs, motion clips (in git)
│   ├── policy.onnx
│   ├── dance1.bin
│   └── deploy_config.yaml
├── patches/                             # local diffs applied on top of upstream
│   └── unitree_mujoco-g1-scene.patch
├── scripts/
│   └── setup.sh
├── unitree_mujoco/                      # fetched by setup.sh
├── unitree_rl_lab/                      # fetched by setup.sh
├── unitree_sdk2/                        # fetched by setup.sh
└── onnxruntime-linux-x64-1.21.0/        # fetched by setup.sh
```

The four directories at the bottom are **not** stored in git; `setup.sh` clones
each Unitree repo at a pinned commit and downloads onnxruntime so every
collaborator starts from the same source tree.

## Setup

Requires `git`, `bash`, `curl`, `tar`, `sha256sum`.

```bash
git clone git@github.com:LeeLeno/deploy_ws.git
cd deploy_ws
./scripts/setup.sh
```

The script is idempotent — re-running it will fetch missing pieces and skip
anything already in place. Patches are applied only if not already applied.

## Pinned versions

| Component       | Source                                            | Commit / version                          |
| --------------- | ------------------------------------------------- | ----------------------------------------- |
| unitree_mujoco  | unitreerobotics/unitree_mujoco                    | `1a37b05`                                 |
| unitree_rl_lab  | unitreerobotics/unitree_rl_lab                    | `4960b84`                                 |
| unitree_sdk2    | unitreerobotics/unitree_sdk2                      | `6d03531`                                 |
| onnxruntime     | microsoft/onnxruntime release                     | `v1.21.0` linux-x64                       |

To bump a pin, edit the SHA in `scripts/setup.sh`, re-run setup, regenerate
patches if needed, and commit.

## Local modifications

Tracked as patch files under `patches/`. Currently:

- `unitree_mujoco-g1-scene.patch` — switches `simulate/config.yaml` from the
  default Go2 to G1 (`scene_29dof.xml`).

If you modify a fetched Unitree repo, regenerate its patch:

```bash
cd unitree_mujoco
git diff > ../patches/unitree_mujoco-<short-description>.patch
```

When the diff against upstream grows past "a couple of files", we'll switch
that subrepo to a personal fork + git submodule instead of a patch.
