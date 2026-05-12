"""
ACT policy evaluation in PyBullet panda-gym.
3-camera setup matching Data_Collection_v3_3cam_table.py.
8-D joint-space actions, temporal ensembling, goal-type split.
"""
import os
import sys

sys.path.insert(0, "ACT")

# Config first to set os.environ['DEVICE'] before any detr imports
from ACT.config.config import TASK_CONFIG, POLICY_CONFIG, TRAIN_CONFIG

import math
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
CKPT = "E:/ACT_picknplace/ACT/checkpoints/picknplace3cam8action/policy_epoch_400_seed_42.ckpt"
STATS_PATH = "E:/ACT_picknplace/ACT/checkpoints/picknplace3cam8action/dataset_stats.pkl"

CHUNK_SIZE = POLICY_CONFIG["num_queries"]      # match training (20 in your config)
EPISODE_LEN = 100
N_TRIALS = 50

TEMPORAL_AGG = True
TABLE_GOALS_ONLY = True       # match training distribution
RENDERER = "OpenGL"             # match collection

CAMERA_NAMES = ["top", "front", "side"]
IMG_H, IMG_W = 240, 320

device = os.environ.get("DEVICE", "cuda")


# ============================================================
# Wrappers and rendering — must match Data_Collection_v3_3cam_table.py
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


def render_cameras(client, cameras):
    """Render all 3 views, return dict of (H, W, 3) uint8 arrays."""
    out = {}
    for name, (vm, pm) in cameras.items():
        _, _, rgba, _, _ = client.getCameraImage(
            width=IMG_W, height=IMG_H,
            viewMatrix=vm, projectionMatrix=pm,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
        out[name] = rgb
    return out


def get_qpos(env):
    """Return 8-D qpos: 7 joint angles + gripper width."""
    robot = env.unwrapped.robot
    sim = env.unwrapped.sim
    arm_q = np.array(
        [sim.get_joint_angle(robot.body_name, j) for j in range(7)],
        dtype=np.float32,
    )
    grip = float(robot.get_fingers_width())
    return np.concatenate([arm_q, [grip]]).astype(np.float32)


def stack_camera_input(frames, device):
    """
    Take dict of {cam_name: (H,W,3) uint8} and produce
    (1, num_cams, 3, H, W) float tensor in [0, 1] on device.
    """
    cam_tensors = []
    for cam_name in CAMERA_NAMES:
        img = frames[cam_name]
        img = rearrange(img, "h w c -> c h w")  # (3, H, W)
        cam_tensors.append(img)
    stacked = np.stack(cam_tensors, axis=0)     # (num_cams, 3, H, W)
    img = torch.from_numpy(stacked).to(device).float() / 255.0
    return img.unsqueeze(0)                      # (1, num_cams, 3, H, W)


# ============================================================
# Main eval loop
# ============================================================
def main():
    print(f"Checkpoint: {CKPT}")
    print(f"Stats:      {STATS_PATH}")
    print(f"TEMPORAL_AGG: {TEMPORAL_AGG}  TABLE_GOALS_ONLY: {TABLE_GOALS_ONLY}")
    print(f"Cameras: {CAMERA_NAMES}  Renderer: {RENDERER}")
    print(f"Episode len: {EPISODE_LEN}  Chunk: {CHUNK_SIZE}  Trials: {N_TRIALS}")

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

    state_dim = TASK_CONFIG["state_dim"]

    # Build env (without AddRenderObservation — we render via PyBullet directly)
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

    results = []

    for trial in range(N_TRIALS):
        obs, _ = env.reset()
        cube_z = float(obs["achieved_goal"][2])
        goal_z = float(obs["desired_goal"][2])
        goal_type = "air" if (goal_z - cube_z) > 0.02 else "table"
        cube_start = obs["achieved_goal"].copy()
        goal_pos = obs["desired_goal"].copy()

        # Buffer for temporal ensembling
        if TEMPORAL_AGG:
            all_time_actions = torch.zeros(
                [EPISODE_LEN, EPISODE_LEN + CHUNK_SIZE, state_dim],
                device=device,
            )

        with torch.inference_mode():
            for t in range(EPISODE_LEN):
                # Build inputs
                qpos_np = get_qpos(env)
                qpos = torch.from_numpy(qpos_np).to(device).float().unsqueeze(0)
                qpos_norm = (qpos - qpos_mean) / qpos_std

                frames = render_cameras(client, cameras)
                img = stack_camera_input(frames, device)  # (1, 3, 3, H, W)

                if TEMPORAL_AGG:
                    pred_chunk_norm = policy(qpos_norm, img)  # (1, CHUNK, 8)
                    all_time_actions[[t], t : t + CHUNK_SIZE] = pred_chunk_norm

                    actions_for_t = all_time_actions[:, t]
                    populated = torch.all(actions_for_t != 0, dim=1)
                    actions_for_t = actions_for_t[populated]

                    k_decay = 0.01
                    weights = torch.exp(
                        -k_decay * torch.arange(len(actions_for_t), device=device).float()
                    )
                    weights = (weights / weights.sum()).unsqueeze(1)
                    raw_norm = (actions_for_t * weights).sum(dim=0, keepdim=True)
                else:
                    if t % CHUNK_SIZE == 0:
                        pred_chunk_norm = policy(qpos_norm, img)
                    raw_norm = pred_chunk_norm[:, t % CHUNK_SIZE]

                # Un-normalize, clip, step
                action = (raw_norm.squeeze(0) * act_std + act_mean).cpu().numpy()
                action = np.clip(action, -1.0, 1.0)
                obs, _, _term, _trunc, info = env.step(action)
                # No break — let full episode run to evaluate fairly

        # Success criteria
        final_cube = obs["achieved_goal"]
        env_succ = bool(info.get("is_success", False))
        cube_to_goal = float(np.linalg.norm(final_cube - goal_pos))
        cube_moved = float(np.linalg.norm(final_cube - cube_start))
        cube_start = obs["achieved_goal"].copy()
        goal_pos = obs["desired_goal"].copy()

        # After episode ends:
        print(f"  cube_start: {cube_start.round(3)}")
        print(f"  goal_pos:   {goal_pos.round(3)}")
        print(f"  final_cube: {final_cube.round(3)}")

        # Track where the EE ended up
        ee_final = env.unwrapped.robot.get_ee_position()
        print(f"  ee_final:   {ee_final.round(3)}")

        results.append({
            "trial": trial,
            "success": env_succ,
            "goal_type": goal_type,
            "cube_to_goal": cube_to_goal,
            "cube_moved": cube_moved,
        })

        rate = sum(r["success"] for r in results) / len(results)
        print(
            f"Trial {trial+1:2d}/{N_TRIALS}: "
            f"{'SUCCESS' if env_succ else 'fail'}  "
            f"({goal_type})  "
            f"cube_dist={cube_to_goal:.3f}  moved={cube_moved:.3f}  "
            f"rate={rate:.0%}"
        )

    # Aggregate
    overall = sum(r["success"] for r in results)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({CKPT.split('/')[-1]})")
    print(f"{'='*60}")
    print(f"Overall: {overall}/{N_TRIALS} = {overall/N_TRIALS:.0%}")

    table = [r for r in results if r["goal_type"] == "table"]
    air = [r for r in results if r["goal_type"] == "air"]
    if table:
        ts = sum(r["success"] for r in table)
        print(f"  Table goals: {ts}/{len(table)} = {ts/len(table):.0%}")
    if air:
        as_ = sum(r["success"] for r in air)
        print(f"  Air goals:   {as_}/{len(air)} = {as_/len(air):.0%}")

    # Diagnostic: how close does the policy get on failures?
    near_misses = [r for r in results if not r["success"] and r["cube_to_goal"] < 0.10]
    print(f"\nNear-misses (failed but cube within 10cm of goal): {len(near_misses)}")
    if near_misses:
        avg_dist = np.mean([r["cube_to_goal"] for r in near_misses])
        print(f"  Avg distance: {avg_dist:.3f} m")

    no_grasp = [r for r in results if r["cube_moved"] < 0.05]
    print(f"Failed-to-grasp (cube barely moved): {len(no_grasp)}")

    env.close()


if __name__ == "__main__":
    main()