import pickle
import numpy as np

STATS_PATH = "E:/ACT_picknplace/ACT/checkpoints/picknplaceJointsAbs/dataset_stats.pkl"

with open(STATS_PATH, "rb") as f:
    stats = pickle.load(f)

print("=== ACTION STATS ===")
print(f"Shape: {stats['action_mean'].shape}")
print(f"\nAction mean (per dim):")
for i in range(8):
    label = f"joint {i}" if i < 7 else "gripper"
    print(f"  {label:8s}: {stats['action_mean'][i]:+.4f}")

print(f"\nAction std (per dim):")
for i in range(8):
    label = f"joint {i}" if i < 7 else "gripper"
    print(f"  {label:8s}: {stats['action_std'][i]:.4f}")

print(f"\n=== QPOS STATS ===")
print(f"Shape: {stats['qpos_mean'].shape}")
print(f"\nQpos mean (per dim):")
for i in range(8):
    label = f"joint {i}" if i < 7 else "gripper_w"
    print(f"  {label:9s}: {stats['qpos_mean'][i]:+.4f}")

print(f"\nQpos std (per dim):")
for i in range(8):
    label = f"joint {i}" if i < 7 else "gripper_w"
    print(f"  {label:9s}: {stats['qpos_std'][i]:.4f}")

print(f"\nExample qpos (first 3 timesteps):")
print(stats['example_qpos'][:3])