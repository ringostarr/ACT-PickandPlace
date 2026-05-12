"""
ACT policy evaluation in PyBullet panda-gym.
4-camera setup (top, front, side, wrist) with goal-in-qpos (11-D state).
8-D ABSOLUTE joint position actions, temporal ensembling, goal-type split.

CRITICAL: This eval converts predicted absolute joint targets back to deltas
before stepping the env, since panda-gym's "joints" control mode expects deltas.

Metrics printed at the end:
  - Success rate
  - Avg and std of episode length (all 50 trials)
  - Avg and std of episode length (successful trials only)
"""
import os
import sys
import json

sys.path.insert(0, "ACT")

# Config first to set os.environ['DEVICE'] before any detr imports
from ACT.config.config import TASK_CONFIG, POLICY_CONFIG, TRAIN_CONFIG

import torch
import numpy as np
import pickle
from einops import rearrange
import gymnasium as gym
import panda_gym  # noqa: F401
import pybullet as p

from ACT.training.utils import make_policy

# Clean up any stale PyBullet connection
try:
    import pybullet
    pybullet.disconnect()
except Exception:
    pass


# ============================================================
# CONFIGURATION
# ============================================================
CKPT = "E:/ACT_picknplace/ACT/checkpoints/picknplaceJointsAbs4camNoisyqpos/policy_best.ckpt"
STATS_PATH = "E:/ACT_picknplace/ACT/checkpoints/picknplaceJointsAbs4camNoisyqpos/dataset_stats.pkl"

CHUNK_SIZE = POLICY_CONFIG["num_queries"]
EPISODE_LEN = 100
N_TRIALS = 50

TEMPORAL_AGG = True
TABLE_GOALS_ONLY = True
RENDERER = "OpenGL"

CAMERA_NAMES = ["top", "front", "side", "wrist"]
IMG_H, IMG_W = 240, 320

ACTION_DIM = 8           # joint deltas + gripper
GRIPPER_IDX = 7          # last dim of 8-D action
APPROACH_STEPS = 0      # force-open gripper for first N steps

# Collector uses: djoint = (target_arm - cur_arm) / 0.05
JOINT_DELTA_SCALE = 0.05

K_DECAY = 0.25
GHOST_RGBA = (1.0, 0.0, 1.0, 0.6)

device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Wrappers and rendering
# ============================================================
class TableGoalsOnly(gym.Wrapper):
    def __init__(self, env, tolerance=0.02, max_tries=50):
        super().__init__(env)
        self.tolerance = tolerance
        self.max_tries = max_tries

    def reset(self, **kwargs):
        for _ in range(self.max_tries):
            obs, info = self.env.reset(**kwargs)
            if abs(obs["desired_goal"][2] - obs["achieved_goal"][2]) < self.tolerance:
                return obs, info
        return obs, info


def build_cameras():
    proj = p.computeProjectionMatrixFOV(
        fov=60.0, aspect=IMG_W / IMG_H, nearVal=0.1, farVal=3.0
    )
    return {
        "top": (
            p.computeViewMatrix(
                cameraEyePosition=[0.0, 0.0, 1.2],
                cameraTargetPosition=[0.0, 0.0, 0.4],
                cameraUpVector=[0.0, 1.0, 0.0],
            ),
            proj,
        ),
        "front": (
            p.computeViewMatrix(
                cameraEyePosition=[1.2, 0.0, 0.7],
                cameraTargetPosition=[0.3, 0.0, 0.5],
                cameraUpVector=[0.0, 0.0, 1.0],
            ),
            proj,
        ),
        "side": (
            p.computeViewMatrix(
                cameraEyePosition=[0.4, 1.0, 0.8],
                cameraTargetPosition=[0.3, 0.0, 0.5],
                cameraUpVector=[0.0, 0.0, 1.0],
            ),
            proj,
        ),
    }


WRIST_CAM_LOCAL_POS    = np.array([0.0, -0.04, -0.08])
WRIST_CAM_LOCAL_TARGET = np.array([0.0,  0.00,  0.12])
WRIST_CAM_LOCAL_UP     = np.array([0.0, -1.00,  0.00])
WRIST_CAM_FOV = 70.0


def build_wrist_view(client, body_id, link_idx=11):
    state = client.getLinkState(body_id, link_idx, computeForwardKinematics=True)
    pos = np.array(state[0])
    rot = np.array(client.getMatrixFromQuaternion(state[1])).reshape(3, 3)
    cam_pos = pos + rot @ WRIST_CAM_LOCAL_POS
    target  = pos + rot @ WRIST_CAM_LOCAL_TARGET
    up      = rot @ WRIST_CAM_LOCAL_UP
    return p.computeViewMatrix(cam_pos.tolist(), target.tolist(), up.tolist())


def render_cameras(client, cameras, wrist_proj, body_id):
    out = {}
    for name, (vm, pm) in cameras.items():
        _, _, rgba, _, _ = client.getCameraImage(
            width=IMG_W, height=IMG_H,
            viewMatrix=vm, projectionMatrix=pm,
            renderer=p.ER_TINY_RENDERER,
        )
        out[name] = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]

    wrist_vm = build_wrist_view(client, body_id)
    _, _, rgba, _, _ = client.getCameraImage(
        width=IMG_W, height=IMG_H,
        viewMatrix=wrist_vm, projectionMatrix=wrist_proj,
        renderer=p.ER_TINY_RENDERER,
    )
    out["wrist"] = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
    return out


def recolor_goal_marker(env, rgba=GHOST_RGBA):
    sim = env.unwrapped.sim
    target_id = sim._bodies_idx["target"]
    sim.physics_client.changeVisualShape(target_id, -1, rgbaColor=list(rgba))


def get_qpos(env, obs):
    """11-D qpos: 7 joint angles + gripper width + 3 goal coords."""
    robot = env.unwrapped.robot
    sim = env.unwrapped.sim
    arm_q = np.array(
        [sim.get_joint_angle(robot.body_name, j) for j in range(7)],
        dtype=np.float32,
    )
    grip = float(robot.get_fingers_width())
    goal = obs["desired_goal"].astype(np.float32)
    return np.concatenate([arm_q, [grip], goal]).astype(np.float32)


def stack_camera_input(frames, device):
    """(1, num_cams, 3, H, W) float tensor in [0, 1] on device."""
    cam_tensors = []
    for cam_name in CAMERA_NAMES:
        img = frames[cam_name]
        img = rearrange(img, "h w c -> c h w")
        cam_tensors.append(img)
    stacked = np.stack(cam_tensors, axis=0)
    img = torch.from_numpy(stacked).to(device).float() / 255.0
    return img.unsqueeze(0)


def absjoint_to_env_action(predicted_action, current_qpos):
    """
    Convert policy's absolute-joint prediction to env-compatible delta action.

    predicted_action: (8,) numpy
        [0:7] = predicted absolute joint targets in radians
        [7]   = gripper command in [-1, 1]

    current_qpos: (8+,) numpy   <- can be longer (e.g., 11 with goal)
        [0:7] = current joint angles in radians

    Returns: (8,) numpy
        [0:7] = joint deltas normalized to [-1, 1]
        [7]   = gripper command in [-1, 1]
    """
    target_joints = predicted_action[:7]
    current_joints = current_qpos[:7]

    delta = (target_joints - current_joints) / JOINT_DELTA_SCALE
    delta = np.clip(delta, -1.0, 1.0)

    gripper_cmd = np.clip(predicted_action[7], -1.0, 1.0)

    return np.concatenate([delta, [gripper_cmd]]).astype(np.float32)


# ============================================================
# Main eval loop
# ============================================================
def main():
    print(f"Checkpoint:   {CKPT}")
    print(f"Stats:        {STATS_PATH}")
    print(f"Action mode:  ABSOLUTE joint targets -> converted to delta at env.step")
    print(f"Cameras:      {CAMERA_NAMES}")
    print(f"qpos:         11-D (7 joints + gripper + 3 goal)")
    print(f"TEMPORAL_AGG: {TEMPORAL_AGG}    k_decay={K_DECAY}")
    print(f"Episode len:  {EPISODE_LEN}    Chunk: {CHUNK_SIZE}    Trials: {N_TRIALS}")
    print(f"Force-open:   first {APPROACH_STEPS} steps")
    print(f"Goals:        table-only={TABLE_GOALS_ONLY}    ghost rgba={GHOST_RGBA}")

    # Load policy
    policy = make_policy(POLICY_CONFIG["policy_class"], POLICY_CONFIG)
    state_dict = torch.load(CKPT, map_location=device, weights_only=True)
    status = policy.load_state_dict(state_dict)
    if status.missing_keys or status.unexpected_keys:
        print(f"WARN load: missing={status.missing_keys}  unexpected={status.unexpected_keys}")
    policy.to(device).eval()

    # Load normalization stats
    with open(STATS_PATH, "rb") as f:
        stats = pickle.load(f)
    qpos_mean = torch.from_numpy(stats["qpos_mean"]).to(device).float()
    qpos_std = torch.from_numpy(stats["qpos_std"]).to(device).float()
    act_mean = torch.from_numpy(stats["action_mean"]).to(device).float()
    act_std = torch.from_numpy(stats["action_std"]).to(device).float()

    print(f"  qpos_mean shape: {tuple(qpos_mean.shape)}")
    print(f"  act_mean shape:  {tuple(act_mean.shape)}")

    # Build env
    base_env = gym.make(
        "PandaPickAndPlace-v3",
        render_mode="rgb_array",
        renderer=RENDERER,
        control_type="joints",
        reward_type="sparse",
        max_episode_steps=EPISODE_LEN,
    )

    env = TableGoalsOnly(base_env) if TABLE_GOALS_ONLY else base_env
    client = base_env.unwrapped.sim.physics_client
    cameras = build_cameras()
    sim = base_env.unwrapped.sim
    body_id = sim._bodies_idx[base_env.unwrapped.robot.body_name]
    wrist_proj = p.computeProjectionMatrixFOV(
        fov=WRIST_CAM_FOV, aspect=IMG_W / IMG_H, nearVal=0.02, farVal=2.0
    )
    recolor_goal_marker(env)

    results = []

    for trial in range(N_TRIALS):
        obs, _ = env.reset()
        cube_z = float(obs["achieved_goal"][2])
        goal_z = float(obs["desired_goal"][2])
        goal_type = "air" if (goal_z - cube_z) > 0.02 else "table"
        cube_start = obs["achieved_goal"].copy()
        goal_pos = obs["desired_goal"].copy()
        success_step = None      # NEW: first step where is_success becomes True

        # Buffer for temporal ensembling (uses ACTION_DIM, not state_dim)
        if TEMPORAL_AGG:
            all_time_actions = torch.zeros(
                [EPISODE_LEN, EPISODE_LEN + CHUNK_SIZE, ACTION_DIM],
                device=device,
            )

        with torch.inference_mode():
            for t in range(EPISODE_LEN):
                qpos_np = get_qpos(env, obs)
                qpos = torch.from_numpy(qpos_np).to(device).float().unsqueeze(0)
                qpos_norm = (qpos - qpos_mean) / qpos_std

                frames = render_cameras(client, cameras, wrist_proj, body_id)
                img = stack_camera_input(frames, device)

                if TEMPORAL_AGG:
                    pred_chunk_norm = policy(qpos_norm, img)  # (1, CHUNK, 8)
                    all_time_actions[[t], t: t + CHUNK_SIZE] = pred_chunk_norm

                    actions_for_t = all_time_actions[:, t]
                    populated = torch.all(actions_for_t != 0, dim=1)
                    actions_for_t = actions_for_t[populated]

                    weights = torch.exp(
                        -K_DECAY * torch.arange(len(actions_for_t), device=device).float()
                    )
                    weights = (weights / weights.sum()).unsqueeze(1)
                    raw_norm = (actions_for_t * weights).sum(dim=0, keepdim=True)
                else:
                    if t % CHUNK_SIZE == 0:
                        pred_chunk_norm = policy(qpos_norm, img)
                    raw_norm = pred_chunk_norm[:, t % CHUNK_SIZE]

                # Un-normalize
                predicted_action = (raw_norm.squeeze(0) * act_std + act_mean).cpu().numpy()

                # Convert absolute joint target -> env-compatible delta action
                env_action = absjoint_to_env_action(predicted_action, qpos_np)

                # Force-open during approach to break the closed-grip feedback loop
                if t < APPROACH_STEPS:
                    env_action[GRIPPER_IDX] = 1.0

                obs, _, _term, _trunc, info = env.step(env_action)

                if info.get("is_success", False) and success_step is None:
                    success_step = t + 1     # 1-indexed step count

        # End-of-episode evaluation
        final_cube = obs["achieved_goal"]
        env_succ = bool(info.get("is_success", False)) or (success_step is not None)
        cube_to_goal = float(np.linalg.norm(final_cube - goal_pos))
        cube_moved = float(np.linalg.norm(final_cube - cube_start))

        results.append({
            "trial": trial,
            "success": env_succ,
            "success_step": success_step,
            "goal_type": goal_type,
            "cube_to_goal": cube_to_goal,
            "cube_moved": cube_moved,
        })

        rate = sum(r["success"] for r in results) / len(results)
        step_str = f"{success_step}" if success_step is not None else "  -"
        print(
            f"Trial {trial+1:2d}/{N_TRIALS}: "
            f"{'SUCCESS' if env_succ else 'fail'}  "
            f"first_succ_step={step_str:>3s}  "
            f"({goal_type})  "
            f"cube_dist={cube_to_goal:.3f}  moved={cube_moved:.3f}  "
            f"rate={rate:.0%}"
        )

    # ============================================================
    # Aggregate metrics
    # ============================================================
    overall = sum(r["success"] for r in results)
    success_rate = overall / len(results)

    successful_steps = [r["success_step"] for r in results if r["success_step"] is not None]
    all_steps = [r["success_step"] if r["success_step"] is not None else EPISODE_LEN
                 for r in results]

    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS  (n = {len(results)})")
    print(f"  Checkpoint: {os.path.basename(CKPT)}")
    print(f"  Action mode: absolute joint targets")
    print(f"{'='*60}")
    print(f"Success rate:                  {success_rate*100:.1f}%  ({overall}/{len(results)})")
    print(f"Avg episode length (all):      {np.mean(all_steps):6.2f} steps")
    print(f"Std episode length (all):      {np.std(all_steps):6.2f} steps")
    print(f"Min/Max (all):                 {int(np.min(all_steps)):3d} / {int(np.max(all_steps)):3d} steps")

    if successful_steps:
        print()
        print(f"Avg episode length (success):  {np.mean(successful_steps):6.2f} steps")
        print(f"Std episode length (success):  {np.std(successful_steps):6.2f} steps")
        print(f"  (over {len(successful_steps)} successful trials)")
    else:
        print("\nNo successful trials.")

    # Goal-type breakdown
    table = [r for r in results if r["goal_type"] == "table"]
    air   = [r for r in results if r["goal_type"] == "air"]
    if table:
        ts = sum(r["success"] for r in table)
        print(f"\nTable goals: {ts}/{len(table)} = {ts/len(table)*100:.0f}%")
    if air:
        as_ = sum(r["success"] for r in air)
        print(f"Air goals:   {as_}/{len(air)} = {as_/len(air)*100:.0f}%")

    near_misses = [r for r in results if not r["success"] and r["cube_to_goal"] < 0.10]
    no_grasp = [r for r in results if r["cube_moved"] < 0.05]
    print(f"\nNear-misses (failed but cube within 10cm of goal): {len(near_misses)}")
    if near_misses:
        avg_dist = np.mean([r["cube_to_goal"] for r in near_misses])
        print(f"  Avg distance: {avg_dist:.3f} m")
    print(f"Failed-to-grasp (cube barely moved):                {len(no_grasp)}")
    print(f"{'='*60}")

    # Save per-trial results
    out_path = os.path.join(os.path.dirname(CKPT), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPer-trial results saved to: {out_path}")

    env.close()


if __name__ == "__main__":
    main()