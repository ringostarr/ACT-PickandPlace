# ACT Pick-and-Place (panda-gym)

An Action Chunking Transformer (ACT) policy trained by behaviour cloning on the panda-gym `PandaPickAndPlace-v3` task. Reaches 72% success over 50 randomized evaluation trials.

See `ACT_report.docx` for the writeup and `success_clips.mp4` for example rollouts.

## Setup

```bash
git clone <this-repo>
cd ACT_picknplace

conda create -n act_picknplace python=3.10
conda activate act_picknplace

pip install -r ACT/requirements.txt
pip install gymnasium panda-gym pybullet h5py einops opencv-python
```

The ACT codebase depends on a local DETR package. If `import detr` fails after `requirements.txt`, install it from the Shaka-Labs fork the same way they specify in `ACT/README.md`.

Tested on Python 3.10, PyTorch 2.x, single CUDA GPU.

## Layout

```
ACT_picknplace/
├── ACT/                              # forked Shaka-Labs ACT codebase
│   ├── checkpoints/
│   │   └── picknplaceJointsAbs4camnoisyqpos/  
│   │       ├── eval_results.json               # per-trial eval output
│   │       ├── train_val_loss_seed_42.png      # total loss curve
│   │       ├── train_val_l1_seed_42.png        # L1 (reconstruction) curve
│   │       └── train_val_kl_seed_42.png        # KL curve
│   ├── config/config.py              # TASK_CONFIG, POLICY_CONFIG, TRAIN_CONFIG
│   ├── training/                     # policy, dataset, utils
│   ├── train.py                      # training entrypoint
│   ├── evaluate.py                   # Shaka-Labs default eval (not used here)
│   ├── requirements.txt
│   ├── README.md                     # original Shaka-Labs README
│   └── (record_episodes.py, robot.py, teleoperation.py, dynamixel.py
│        are from Shaka-Labs' real-robot pipeline and not used in this sim project)
├── data/                             # collected HDF5 demonstration episodes
├── Data_collection_4cam_qpos.py      # main collector (4 cam, goal-in-qpos)  <-- use this
├── Data_collector_4cams.py           # earlier 4-cam version, no goal in qpos
├── data_collection_3cam_8DAction.py  # earliest 3-cam version
├── eval_absjoints.py                 # 50-trial eval, joint-action policy   <-- use this
├── eval.py                           # EE-action eval (parallel experiment)
├── inspect_demo.py                   # visualize one HDF5 episode
├── demo_frames.png                   # sample camera frames
├── README.md                         # this file
├── ACT_report.docx                   # 1-2 page writeup
└── success_clips.mp4                 # demo video, 5+ successful rollouts
```

The data path used during training is set inside `ACT/config/config.py` (`DATA_DIR`). Update it to point at whatever subfolder of `data/` your collector wrote to.

## Reproduce

### 1. Collect demonstrations

```bash
python Data_collection_4cam_qpos.py --num-episodes 100 \
    --output-dir data/pick_and_place_4cam_absjoint_noisy_goalqpos
```

Saves 100 HDF5 episodes (4 cameras at 240x320, 11-D qpos, 8-D absolute joint targets). DAgger-style noise injection is applied so the encoder sees a richer state distribution.

Optional: `python inspect_demo.py` to view frames from one episode.

### 2. Train

Update `DATA_DIR` in `ACT/config/config.py` to the dataset path from step 1, then:

```bash
cd ACT
python train.py
```

Trains for 2000 epochs and selects the best checkpoint by validation L1 (best typically lands around epoch 1600). Outputs go to `ACT/checkpoints/<run_name>/`: `policy_best.ckpt`, `policy_last.ckpt`, per-100-epoch snapshots, `dataset_stats.pkl`, and three loss curves (`train_val_loss_seed_42.png`, `train_val_l1_seed_42.png`, `train_val_kl_seed_42.png`). Our final run lives at `ACT/checkpoints/picknplaceJointsAbs4camnoisyqpos/`.

### 3. Evaluate

Update `CKPT` and `STATS_PATH` at the top of `eval_absjoints.py` to point at the checkpoint from step 2, then:

```bash
python eval_absjoints.py
```

Prints success rate, average and std episode length over 50 trials. Per-trial results saved to `eval_results.json` next to the checkpoint.

## Design choices

- **8-D absolute joint targets** as the action space. Converted to deltas at env.step time via `(target - current) / 0.05`. Absolute targets give a more stationary regression target than deltas; follows the ACT paper recommendation.
- **11-D qpos** = 7 joint angles + gripper width + 3 goal coordinates. Putting the goal in state removes the need to triangulate it from pixels.
- **4 cameras** (top, front, side, wrist). The wrist camera follows the EE link and shows the grasp moment closely. 3-camera trials during development did not produce reliable behaviour.
- **Magenta ghost goal marker.** Panda-gym's default green ghost is the same color as the real cube. Recoloring to magenta removed a visual ambiguity that was biasing the policy toward the wrong target.
- **DAgger-style noise injection** during collection (sigma 0.08 transit / 0.02 precise phases, plus small XY/Z waypoint jitter). Clean expert action is saved as the supervision label; the perturbed action is what gets executed.
- **Temporal ensembling at eval** with k_decay = 0.25 over the 50-action chunk. Gripper dim uses the latest prediction only (binarized to +/-1), since smoothing introduces transition lag on a categorical signal.

## Modifications to the Shaka-Labs ACT codebase

`detr/models/detr_vae.py` (DETR package installed alongside ACT):
- `encoder_joint_proj` and `input_proj_robot_state` use `state_dim` instead of a hardcoded literal (state_dim is 11 with goal in qpos).
- `action_head` and `encoder_action_proj` set to 8-D (originally 5 in the Shaka-Labs fork, sized for their robot).
- `build()` uses `args.state_dim` instead of a hardcoded literal.

`ACT/train.py`:
- Gradient clipping placed after `loss.backward()` and before `optimizer.step()`.
- Best checkpoint saved on validation-loss improvement.

## Other scripts in the repo

- `Data_collector_4cams.py`, `data_collection_3cam_8DAction.py`: earlier iterations of the collector. The final version is `Data_collection_4cam_qpos.py`.
- `eval.py`: parallel evaluation script for the 4-D EE-action experiment. Did not complete a fair comparison; see Section 6 of the report.
- `inspect_demo.py`: data inspection utility used during development.

## Results

| Metric (50 trials) | Value |
| --- | --- |
| Success rate | 72.0% (36/50) |
| Avg episode length (all) | 70.16 steps |
| Std episode length (all) | 26.18 steps |
| Avg episode length (success only, n=36) | 58.56 steps |
| Std episode length (success only) | 21.69 steps |

The dominant failure mode is failed-to-grasp (11 of 14 failures). See the report for full failure analysis. Per-trial JSON for this result is at `ACT/checkpoints/picknplaceJointsAbs4camnoisyqpos/eval_results.json`.

## References

- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT paper), 2023
- Gallouedec et al., *panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning*, 2021
- Shaka-Labs ACT codebase: https://github.com/Shaka-Labs/ACT


