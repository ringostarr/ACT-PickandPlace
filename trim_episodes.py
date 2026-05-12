"""Trim episodes to first N timesteps. Works on any source dataset."""
import os
import argparse
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True,
                    help="Source dataset folder")
    ap.add_argument("--dst", type=str, required=True,
                    help="Destination folder for trimmed data")
    ap.add_argument("--cutoff", type=int, default=60,
                    help="Number of timesteps to keep")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted([f for f in os.listdir(args.src) if f.startswith("episode_")])
    print(f"Trimming {len(files)} episodes from {args.src} -> {args.dst}")
    print(f"Cutting to first {args.cutoff} timesteps")

    for i, fname in enumerate(files):
        src = os.path.join(args.src, fname)
        dst = os.path.join(args.dst, fname)

        with h5py.File(src, "r") as fr, h5py.File(dst, "w") as fw:
            # Copy all attributes (sim flag, etc.)
            for k, v in fr.attrs.items():
                fw.attrs[k] = v
            fw.attrs["episode_len"] = args.cutoff

            # Verify source has enough timesteps
            src_len = fr["action"].shape[0]
            if src_len < args.cutoff:
                print(f"  SKIP {fname}: only {src_len} timesteps (< {args.cutoff})")
                continue

            og = fw.create_group("observations")
            ig = og.create_group("images")

            # Auto-detect cameras (some datasets have only 'top')
            camera_names = list(fr["observations/images"].keys())
            for cam in camera_names:
                src_imgs = fr[f"observations/images/{cam}"]
                imgs = src_imgs[:args.cutoff]
                ig.create_dataset(
                    cam, data=imgs,
                    chunks=(1,) + imgs.shape[1:],
                    compression="lzf",
                )

            og.create_dataset("qpos", data=fr["observations/qpos"][:args.cutoff])
            og.create_dataset("qvel", data=fr["observations/qvel"][:args.cutoff])
            fw.create_dataset("action", data=fr["action"][:args.cutoff])

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(files)} done")

    print(f"\nDone. Trimmed dataset at: {os.path.abspath(args.dst)}")


if __name__ == "__main__":
    main()