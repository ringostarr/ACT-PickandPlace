"""
Data_Collection_v3_3cam_table_absjoint.py

3 cameras + ABSOLUTE joint position actions + phase-based expert.
Filters to TABLE-ONLY goals.
Strict success: expert AND env both confirm task completion.
100 timesteps per episode, 100 episodes total.

Output schema (HDF5 per episode):
  /observations/images/top    (100, 240, 320, 3)  uint8
  /observations/images/front  (100, 240, 320, 3)  uint8
  /observations/images/side   (100, 240, 320, 3)  uint8
  /observations/qpos          (100, 8)             float32   (7 joint angles + gripper width)
  /observations/qvel          (100, 8)             float32
  /action                     (100, 8)             float32   (7 ABSOLUTE joint targets in radians + gripper [-1, 1])

NOTE: panda-gym's "joints" control mode expects DELTAS. The actions saved here are
ABSOLUTE joint targets. The collector itself converts target → delta to step the env.
At eval/training time the policy learns to predict absolute targets, and eval.py must
convert prediction → delta before stepping.
"""

import os
import math
import argparse
import numpy as np
import h5py
import gymnasium as gym
import panda_gym  # noqa: F401
import pybullet as p


IMG_H, IMG_W = 240,320
EPISODE_LEN = 100
CAMERA_NAMES = ["top", "front", "side"]


# ─────────────────────────────────────────────────────────────
# Wrapper: filter to table-surface goals only
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Cameras
# ─────────────────────────────────────────────────────────────
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

def recolor_goal_marker(env, rgba=(1.0, 0.0, 1.0, 0.6)):
    """Repaint panda-gym's goal ghost so it can't be confused with the real cube.
    Bright magenta with alpha 0.6 reads clearly as 'this is a marker, not an object'."""
    sim = env.unwrapped.sim
    client = sim.physics_client
    target_id = sim._bodies_idx["target"]   # panda-gym names the ghost "target"
    client.changeVisualShape(target_id, -1, rgbaColor=list(rgba))
# ─────────────────────────────────────────────────────────────
# Phase-based expert (plans EE targets, IK gives absolute joint targets)
# ─────────────────────────────────────────────────────────────
class PhaseExpert:
    XY_TOL_GRASP = 0.018
    XY_TOL_TRANSIT = 0.025
    Z_TOL = 0.010
    GRASP_HOLD = 10
    RELEASE_HOLD = 10

    def __init__(self):
        self.phase = 0
        self.counter = 0

    def reset(self):
        self.phase = 0
        self.counter = 0

    def plan(self, ee, cube, goal):
        def xy_err(t):
            return math.hypot(ee[0] - t[0], ee[1] - t[1])
        def z_err(t):
            return abs(ee[2] - t[2])

        if self.phase == 0:
            target = np.array([cube[0], cube[1], cube[2] + 0.08])
            grip = True
            if xy_err(cube) < self.XY_TOL_GRASP and z_err(target) < self.Z_TOL:
                self.phase = 1
        elif self.phase == 1:
            target = np.array([cube[0], cube[1], cube[2] + 0.002])
            grip = True
            if z_err(target) < self.Z_TOL and xy_err(cube) < self.XY_TOL_GRASP:
                self.phase = 2
                self.counter = 0
        elif self.phase == 2:
            target = np.array([cube[0], cube[1], cube[2] + 0.002])
            grip = False
            self.counter += 1
            if self.counter >= self.GRASP_HOLD:
                self.phase = 3
        elif self.phase == 3:
            lift_h = max(goal[2] + 0.10, 0.22)
            target = np.array([ee[0], ee[1], lift_h])
            grip = False
            if ee[2] > lift_h - self.Z_TOL:
                self.phase = 4
        elif self.phase == 4:
            target = np.array([goal[0], goal[1], ee[2]])
            grip = False
            if xy_err(goal) < self.XY_TOL_TRANSIT:
                self.phase = 5
        elif self.phase == 5:
            target = np.array([goal[0], goal[1], goal[2] + 0.025])
            grip = False
            cube_to_goal = np.linalg.norm(cube - goal)
            if cube_to_goal < 0.05:
                self.phase = 6
                self.counter = 0
        elif self.phase == 6:
            target = np.array([goal[0], goal[1], goal[2] + 0.025])
            grip = True
            self.counter += 1
            if self.counter >= self.RELEASE_HOLD:
                self.phase = 7
        else:
            target = np.array([goal[0], goal[1], goal[2] + 0.025])
            grip = True

        return target, grip


# ─────────────────────────────────────────────────────────────
# IK and action computation
# ─────────────────────────────────────────────────────────────
def compute_action_and_step(robot, target_xyz, gripper_open, kp=1.5, max_step=0.015):
    """
    Compute the saved action (ABSOLUTE joint targets) and the env-step action (deltas).

    Returns:
        saved_action: (8,) float32 — [7 absolute joint targets in radians, gripper +/-1]
        step_action:  (8,) float32 — [7 joint deltas normalized to [-1,1], gripper +/-1]
    """
    cur_ee = robot.get_ee_position()
    desired_step = np.clip(kp * (target_xyz - cur_ee), -max_step, max_step)

    # IK gives absolute target joint angles
    target_arm = robot.inverse_kinematics(
        link=11,
        position=cur_ee + desired_step,
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )[:7]

    cur_arm = np.array(
        [robot.get_joint_angle(i) for i in range(7)], dtype=np.float32
    )

    # For env stepping: convert absolute target → delta in [-1, 1]
    djoint = (target_arm - cur_arm) / 0.05
    djoint = np.clip(djoint, -1.0, 1.0).astype(np.float32)

    grip = np.array([1.0 if gripper_open else -1.0], dtype=np.float32)

    saved_action = np.concatenate([target_arm.astype(np.float32), grip])
    step_action = np.concatenate([djoint, grip])

    return saved_action, step_action


# ─────────────────────────────────────────────────────────────
# Robot state
# ─────────────────────────────────────────────────────────────
def get_qpos_qvel(robot, sim):
    arm_q = np.array(
        [sim.get_joint_angle(robot.body_name, j) for j in range(7)],
        dtype=np.float32,
    )
    arm_qd = np.array(
        [sim.get_joint_velocity(robot.body_name, j) for j in range(7)],
        dtype=np.float32,
    )
    grip = float(robot.get_fingers_width())
    qpos = np.concatenate([arm_q, [grip]]).astype(np.float32)
    qvel = np.concatenate([arm_qd, [0.0]]).astype(np.float32)
    return qpos, qvel


# ─────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────
def run_episode(env, expert, cameras, client):
    obs, _ = env.reset()
    expert.reset()

    robot = env.unwrapped.robot
    sim = env.unwrapped.sim
    cube_start = obs["achieved_goal"].copy()
    goal_pos = obs["desired_goal"].copy()

    imgs = {n: [] for n in CAMERA_NAMES}
    qposs, qvels, acts = [], [], []

    for _ in range(EPISODE_LEN):
        frames = render_cameras(client, cameras)
        qpos, qvel = get_qpos_qvel(robot, sim)
        ee = robot.get_ee_position()
        cube = obs["achieved_goal"]
        goal = obs["desired_goal"]
        target_xyz, grip_open = expert.plan(ee, cube, goal)

        saved_action, step_action = compute_action_and_step(robot, target_xyz, grip_open)

        for n in CAMERA_NAMES:
            imgs[n].append(frames[n])
        qposs.append(qpos)
        qvels.append(qvel)
        acts.append(saved_action)  # save ABSOLUTE targets, step with deltas

        obs, _, _term, _trunc, _info = env.step(step_action)
        # NO break — let controller finish all phases regardless of env termination

    final_cube = obs["achieved_goal"]
    cube_moved = np.linalg.norm(final_cube - cube_start) > 0.05
    near_goal = np.linalg.norm(final_cube - goal_pos) < 0.05
    phase_done = expert.phase == 7
    succ = phase_done and cube_moved and near_goal

    if not succ:
        print(f"    [fail] phase={expert.phase}/7  "
              f"cube_moved={cube_moved} (dist={np.linalg.norm(final_cube - cube_start):.3f})  "
              f"near_goal={near_goal} (dist={np.linalg.norm(final_cube - goal_pos):.3f})")

    return succ, {
        "images": imgs,
        "qpos": np.stack(qposs).astype(np.float32),
        "qvel": np.stack(qvels).astype(np.float32),
        "action": np.stack(acts).astype(np.float32),
        "cube_start": cube_start,
        "goal_pos": goal_pos,
    }


# ─────────────────────────────────────────────────────────────
# Save HDF5
# ─────────────────────────────────────────────────────────────
def save_episode(idx, out_dir, data):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"episode_{idx}.hdf5")
    with h5py.File(path, "w") as f:
        f.attrs["sim"] = "panda-gym-v3"
        f.attrs["episode_len"] = EPISODE_LEN
        f.attrs["action_type"] = "absolute_joint_targets"  # marker for downstream tools
        f.attrs["cube_start"] = data["cube_start"]
        f.attrs["goal_pos"] = data["goal_pos"]
        og = f.create_group("observations")
        ig = og.create_group("images")
        for name in CAMERA_NAMES:
            stack = np.stack(data["images"][name]).astype(np.uint8)
            ig.create_dataset(
                name, data=stack,
                chunks=(1, IMG_H, IMG_W, 3),
                compression="lzf",
            )
        og.create_dataset("qpos", data=data["qpos"])
        og.create_dataset("qvel", data=data["qvel"])
        f.create_dataset("action", data=data["action"])
    return path


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=150)
    ap.add_argument("--output-dir", type=str,
                    default="data/pick_and_place_3cam_absjoint_highres_openGL")
    ap.add_argument("--max-attempts", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Collecting {args.num_episodes} episodes -> {args.output_dir}")
    print(f"Episode length: {EPISODE_LEN}")
    print(f"Cameras: {CAMERA_NAMES}")
    print(f"Action: 8-D ABSOLUTE joint targets (7 radians + 1 gripper)")
    print(f"Goals: TABLE-ONLY  |  Success: STRICT (phase==7 AND cube moved AND at goal)")

    base_env = gym.make(
        "PandaPickAndPlace-v3",
        render_mode="rgb_array",
        renderer="OpenGL",
        control_type="joints",
        reward_type="sparse",
        max_episode_steps=EPISODE_LEN,
    )
    env = TableGoalsOnly(base_env)
    recolor_goal_marker(env)  # <-- add this line
    client = base_env.unwrapped.sim.physics_client
    cameras = build_cameras()
    expert = PhaseExpert()

    saved = 0
    attempts = 0

    while saved < args.num_episodes and attempts < args.max_attempts:
        attempts += 1
        succ, data = run_episode(env, expert, cameras, client)
        if succ:
            save_episode(saved, args.output_dir, data)
            saved += 1
            print(
                f"  [OK] saved={saved}/{args.num_episodes}  "
                f"attempts={attempts}  rate={saved/attempts*100:.0f}%"
            )

    env.close()

    if saved < args.num_episodes:
        print(f"\nWARNING: only {saved}/{args.num_episodes} after {attempts} attempts")
    else:
        print(f"\nDone. {saved} episodes -> {os.path.abspath(args.output_dir)}")
        print(f"Total attempts: {attempts} (success rate {saved/attempts*100:.0f}%)")


if __name__ == "__main__":
    main()