import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("data/pick_and_place_ee_highres/episode_5.hdf5", "r") as f:
    qpos = f["observations/qpos"][:]
    action = f["action"][:]
    images = f["observations/images/front"][:]

print("qpos:", qpos.shape, qpos.dtype)        # (100, 8) float32
print("action:", action.shape, action.dtype)  # (100, 8) float32
print("images:", images.shape, images.dtype)  # (100, 240, 320, 3) uint8
print("action range: [{:.3f}, {:.3f}]".format(action.min(), action.max()))
print("gripper width range: [{:.4f}, {:.4f}]".format(qpos[:, 7].min(), qpos[:, 7].max()))

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for i, t in enumerate(np.linspace(0, len(images) - 1, 8, dtype=int)):
    ax = axes[i // 4, i % 4]
    ax.imshow(images[t])
    ax.set_title(f"t={t}, grip={qpos[t, 7]:.3f}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("demo_frames.png", dpi=80)
plt.show()
print("Saved demo_frames.png")