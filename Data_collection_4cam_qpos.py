"""
Data_Collection_4cam_qpos.py

All accumulated changes:
  + Wrist (gripper-mounted) camera, recomputed each step from EE link pose
  + Stochastic noise on (a) the expert's waypoint and (b) the executed action
  + Image resolution 240x320 (matching eval)
  + EPISODE_LEN 120 (gives noise some headroom)
  + Ghost goal marker recolored to magenta (avoids green-on-green cube confusion)
  + Goal coordinates appended to qpos (state becomes 11-D)

Output schema (HDF5 per episode):
  /observations/images/top    (T, 240, 320, 3)  uint8
  /observations/images/front  (T, 240, 320, 3)  uint8
  /observations/images/side   (T, 240, 320, 3)  uint8
  /observations/images/wrist  (T, 240, 320, 3)  uint8
  /observations/qpos          (T, 11)            float32   <-- 7 joints + grip + 3 goal
  /observations/qvel          (T, 11)            float32
  /action                     (T, 8)             float32   ABSOLUTE joint targets

Config requirements (must match this script):
  state_dim  = 11
  action_dim = 8
  camera_names = ['top', 'front', 'side', 'wrist']
  episode_len = 120
  cam_height, cam_width = 240, 320

Noise design (DAgger-style):
  * SAVED action is always the CLEAN expert action computed from the clean
    (un-jittered) waypoint. Labels never contain noise.
  * Noise is injected ONLY into (i) the expert's waypoint used for execution
    and (ii) the joint deltas sent to env.step().
  * Expert replans every step from the current (perturbed) state, so saved
    (s_t, a*_t) pairs show clean expert response from off-policy states.
  * Noise reduced near contact (phases 1, 2, 5, 6) to preserve grasps.

Action conversion:
  panda-gym's "joints" control mode expects DELTAS. Saved actions are
  ABSOLUTE joint targets. Collector converts target -> delta to step env.
  At eval, policy predicts absolute targets, eval converts to delta.
"""

import os
import math
import argparse
import numpy as np
import h5py
import gymnasium as gym
import panda_gym  # noqa: F401
import pybullet as p


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
IMG_H, IMG_W = 240, 320
EPISODE_LEN = 100
CAMERA_NAMES = ["top", "front", "side", "wrist"]

# Action noise (added to joint deltas in [-1, 1])
ACTION_NOISE_STD_TRANSIT = 0.08
ACTION_NOISE_STD_PRECISE = 0.02

# Waypoint noise (added to expert's target_xyz, in meters)
WAYPOINT_NOISE_XY = 0.005
WAYPOINT_NOISE_Z  = 0.003

# Phases where noise is reduced (descend-grasp, grasp-hold, descend-place, release)
PRECISE_PHASES = (1, 2, 5, 6)

# Wrist camera mount, expressed in the EE link's LOCAL frame.
# With Panda's gripper-down convention (quat [1,0,0,0]):
#   local +Z  -> world -Z  (toward fingers / table)
#   local -Z  -> world +Z  (toward wrist / above gripper)
WRIST_CAM_LOCAL_POS    = np.array([0.0, -0.04, -0.08])
WRIST_CAM_LOCAL_TARGET = np.array([0.0,  0.00,  0.12])
WRIST_CAM_LOCAL_UP     = np.array([0.0, -1.00,  0.00])
WRIST_CAM_FOV = 70.0

# Goal-marker recolor (must match eval script exactly)
GHOST_RGBA = (1.0, 0.0, 1.0, 0.6)


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
# Visual setup
# ─────────────────────────────────────────────────────────────
def recolor_goal_marker(env, rgba=GHOST_RGBA):
    """Repaint panda-gym's goal ghost so it is visually distinct from the real cube.
    Visual change persists across env resets — only need to call once per env."""
    sim = env.unwrapped.sim
    client = sim.physics_client
    target_id = sim._bodies_idx["target"]
    client.changeVisualShape(target_id, -1, rgbaColor=list(rgba))


def build_static_cameras():
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


def build_wrist_view(client, body_id, link_idx=11):
    """Compute wrist camera view matrix from current EE link pose. Recomputed every step."""
    state = client.getLinkState(body_id, link_idx, computeForwardKinematics=True)
    pos = np.array(state[0])
    rot = np.array(client.getMatrixFromQuaternion(state[1])).reshape(3, 3)

    cam_pos = pos + rot @ WRIST_CAM_LOCAL_POS
    target  = pos + rot @ WRIST_CAM_LOCAL_TARGET
    up      = rot @ WRIST_CAM_LOCAL_UP

    return p.computeViewMatrix(
        cameraEyePosition=cam_pos.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist(),
    )


def render_cameras(client, static_cameras, wrist_proj, body_id):
    out = {}
    for name, (vm, pm) in static_cameras.items():
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


# ─────────────────────────────────────────────────────────────
# Phase-based expert
# ─────────────────────────────────────────────────────────────
class PhaseExpert:
    XY_TOL_GRASP = 0.018
    XY_TOL_TRANSIT = 0.025
    Z_TOL = 0.010
    GRASP_HOLD = 12
    RELEASE_HOLD = 10

    def __init__(self):
        self.phase = 0
        self.counter = 0

    def reset(self):
        self.phase = 0
        self.counter = 0

    def plan(self, ee, cube, goal):
        def xy_err(t): return math.hypot(ee[0] - t[0], ee[1] - t[1])
        def z_err(t):  return abs(ee[2] - t[2])

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
# Noise helpers
# ─────────────────────────────────────────────────────────────
def perturb_waypoint(target_xyz, phase, rng):
    if phase in PRECISE_PHASES:
        return target_xyz
    noise = rng.normal(
        0.0,
        np.array([WAYPOINT_NOISE_XY, WAYPOINT_NOISE_XY, WAYPOINT_NOISE_Z]),
    ).astype(np.float32)
    return (target_xyz + noise).astype(np.float32)


def action_noise_std_for_phase(phase):
    return ACTION_NOISE_STD_PRECISE if phase in PRECISE_PHASES else ACTION_NOISE_STD_TRANSIT


# ─────────────────────────────────────────────────────────────
# IK + action computation
# ─────────────────────────────────────────────────────────────
def compute_action_and_step(robot, target_xyz, gripper_open, kp=1.5, max_step=0.015):
    """
    saved_action: 8-D ABSOLUTE joint targets (radians) + gripper [-1, +1]
    step_action:  8-D joint deltas in [-1, 1] + gripper [-1, +1]  (what env.step expects)
    """
    cur_ee = robot.get_ee_position()
    desired_step = np.clip(kp * (target_xyz - cur_ee), -max_step, max_step)

    target_arm = robot.inverse_kinematics(
        link=11,
        position=cur_ee + desired_step,
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )[:7]

    cur_arm = np.array(
        [robot.get_joint_angle(i) for i in range(7)], dtype=np.float32
    )

    djoint = (target_arm - cur_arm) / 0.05
    djoint = np.clip(djoint, -1.0, 1.0).astype(np.float32)

    grip = np.array([1.0 if gripper_open else -1.0], dtype=np.float32)

    saved_action = np.concatenate([target_arm.astype(np.float32), grip])
    step_action = np.concatenate([djoint, grip])
    return saved_action, step_action


# ─────────────────────────────────────────────────────────────
# Robot state — now includes goal in qpos (11-D total)
# ─────────────────────────────────────────────────────────────
def get_qpos_qvel(robot, sim, obs):
    arm_q = np.array(
        [sim.get_joint_angle(robot.body_name, j) for j in range(7)],
        dtype=np.float32,
    )
    arm_qd = np.array(
        [sim.get_joint_velocity(robot.body_name, j) for j in range(7)],
        dtype=np.float32,
    )
    grip = float(robot.get_fingers_width())
    goal = obs["desired_goal"].astype(np.float32)   # 3-D

    qpos = np.concatenate([arm_q, [grip], goal]).astype(np.float32)                            # 11-D
    qvel = np.concatenate([arm_qd, [0.0], np.zeros(3, dtype=np.float32)]).astype(np.float32)   # 11-D
    return qpos, qvel


# ─────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────
def run_episode(env, expert, static_cameras, wrist_proj, client, body_id, rng):
    obs, _ = env.reset()
    expert.reset()

    robot = env.unwrapped.robot
    sim = env.unwrapped.sim
    cube_start = obs["achieved_goal"].copy()
    goal_pos = obs["desired_goal"].copy()

    imgs = {n: [] for n in CAMERA_NAMES}
    qposs, qvels, acts = [], [], []

    for _ in range(EPISODE_LEN):
        frames = render_cameras(client, static_cameras, wrist_proj, body_id)
        qpos, qvel = get_qpos_qvel(robot, sim, obs)
        ee = robot.get_ee_position()
        cube = obs["achieved_goal"]
        goal = obs["desired_goal"]

        # Expert plans from current (possibly perturbed) state
        clean_target, grip_open = expert.plan(ee, cube, goal)
        # Jitter the waypoint for execution only
        noisy_target = perturb_waypoint(clean_target, expert.phase, rng)

        # Saved label: clean expert IK target (NEVER use the noisy target here)
        saved_action, _ = compute_action_and_step(robot, clean_target, grip_open)
        # Executed: noisy waypoint -> deltas, plus per-joint Gaussian noise
        _, step_action = compute_action_and_step(robot, noisy_target, grip_open)
        sigma = action_noise_std_for_phase(expert.phase)
        joint_noise = rng.normal(0.0, sigma, size=7).astype(np.float32)
        step_action[:7] = np.clip(step_action[:7] + joint_noise, -1.0, 1.0)
        # gripper signal stays binary -- noise on it just hurts grasp learning

        for n in CAMERA_NAMES:
            imgs[n].append(frames[n])
        qposs.append(qpos)
        qvels.append(qvel)
        acts.append(saved_action)

        obs, _, _term, _trunc, _info = env.step(step_action)

    final_cube = obs["achieved_goal"]
    cube_moved = np.linalg.norm(final_cube - cube_start) > 0.05
    near_goal = np.linalg.norm(final_cube - goal_pos) < 0.05
    phase_done = expert.phase == 7
    succ = phase_done and cube_moved and near_goal

    if not succ:
        print(f"    [fail] phase={expert.phase}/7  "
              f"moved={np.linalg.norm(final_cube - cube_start):.3f}  "
              f"to_goal={np.linalg.norm(final_cube - goal_pos):.3f}")

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
        f.attrs["sim"] = "panda-gym-v5"
        f.attrs["episode_len"] = EPISODE_LEN
        f.attrs["action_type"] = "absolute_joint_targets"
        f.attrs["state_includes_goal"] = True
        f.attrs["state_dim"] = 11
        f.attrs["action_dim"] = 8
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
    ap.add_argument("--num-episodes", type=int, default=100)
    ap.add_argument("--output-dir", type=str,
                    default="data/pick_and_place_4cam_absjoint_noisy_goalqpos")
    ap.add_argument("--max-attempts", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Collecting {args.num_episodes} episodes -> {args.output_dir}")
    print(f"Episode length: {EPISODE_LEN}    Image: {IMG_H}x{IMG_W}")
    print(f"Cameras: {CAMERA_NAMES}  (wrist follows EE)")
    print(f"State:  11-D (7 joints + grip + 3 goal)")
    print(f"Action: 8-D ABSOLUTE joint targets (7 radians + 1 gripper)")
    print(f"Noise (transit / precise):  action sigma {ACTION_NOISE_STD_TRANSIT}/{ACTION_NOISE_STD_PRECISE}"
          f"   waypoint xy={WAYPOINT_NOISE_XY} z={WAYPOINT_NOISE_Z}")
    print(f"Ghost rgba: {GHOST_RGBA}")
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
    recolor_goal_marker(env)

    sim = base_env.unwrapped.sim
    client = sim.physics_client
    body_id = sim._bodies_idx[base_env.unwrapped.robot.body_name]

    static_cameras = build_static_cameras()
    wrist_proj = p.computeProjectionMatrixFOV(
        fov=WRIST_CAM_FOV, aspect=IMG_W / IMG_H, nearVal=0.02, farVal=2.0
    )
    expert = PhaseExpert()

    saved = 0
    attempts = 0
    while saved < args.num_episodes and attempts < args.max_attempts:
        attempts += 1
        succ, data = run_episode(env, expert, static_cameras, wrist_proj, client, body_id, rng)
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