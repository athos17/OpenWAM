#!/usr/bin/env python3
"""Estimate compressed HDF5/JPEG size and OpenWAM window VAE latent cache size.

Only reads JPEG *encoded bytes*; it does not decode images or run the VAE.
Example:
  python scripts/profiling/estimate_openwam_vae_cache.py --root /data/OpenWAM/data/robotwin2.0/dataset --limit 20
  python scripts/profiling/estimate_openwam_vae_cache.py --root /data/OpenWAM/data/robotwin2.0/dataset --limit 0
"""

import argparse
import random
from pathlib import Path

import h5py


def fmt_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024 or unit == "PiB":
            return f"{value:,.2f} {unit}"
        value /= 1024


def image_payload_size(item):
    """Byte size for RoboTwin's per-frame vlen JPEG payload (no decoding)."""
    return int(item.nbytes) if hasattr(item, "nbytes") else len(item)


def camera_dataset(f, name):
    standard = f"observation/{name}/rgb"
    if standard in f:
        return f[standard]
    if name == "front_camera" and "third_view_rgb" in f:
        return f["third_view_rgb"]
    return None


def jpeg_size_estimate(ds, sample_step):
    n = len(ds)
    if n == 0:
        return 0.0
    idx = range(0, n, sample_step)
    total = sum(image_payload_size(ds[i]) for i in idx)
    # Scale only when subsampling. step=1 is the actual payload sum.
    return total * (n / len(idx))


def count_windows(t, window_len, step, split):
    if t < 2:
        return 0
    # Matches RoboTwinDataset._window_index in current OpenWAM main.
    if split == "train":
        max_start = max(0, t - 2)  # tail windows are padded
    else:
        max_start = max(0, t - window_len)
    return max_start // step + 1


def main():
    p = argparse.ArgumentParser(description="RoboTwin JPEG vs VAE latent cache size")
    p.add_argument("--root", required=True, help="RoboTwin2.0 dataset root")
    p.add_argument("--limit", type=int, default=20,
                   help="Number of randomly sampled HDF5 episodes (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jpeg-step", type=int, default=1,
                   help="Inspect every Nth JPEG frame (1 = read every frame)")
    p.add_argument("--cameras", nargs="+", default=["head_camera", "left_camera", "right_camera"])
    p.add_argument("--height", type=int, default=384, help="Composed VAE input height")
    p.add_argument("--width", type=int, default=320, help="Composed VAE input width")
    p.add_argument("--num-frames", type=int, default=33, help="Raw frames per train window")
    p.add_argument("--video-stride", type=int, default=4)
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--latent-channels", type=int, default=48)
    p.add_argument("--bytes-per-element", type=int, default=2,
                   help="2 for FP16/BF16, 4 for FP32")
    args = p.parse_args()

    if args.jpeg_step <= 0 or args.window_stride <= 0 or args.video_stride <= 0:
        p.error("strides and --jpeg-step must be positive")
    if args.num_frames < 2 or args.height % 16 or args.width % 16:
        p.error("num-frames must be >=2, and height/width divisible by 16")
    if args.limit < 0 or args.bytes_per_element not in (2, 4):
        p.error("limit must be nonnegative, bytes-per-element must be 2 or 4")

    video_frames = len(range(0, args.num_frames, args.video_stride))
    if (video_frames - 1) % 4:
        p.error(f"video_frames={video_frames}: Wan causal VAE usually expects 1+4k frames")
    latent_t = 1 + (video_frames - 1) // 4
    latent_h = args.height // 16
    latent_w = args.width // 16
    per_window = args.latent_channels * latent_t * latent_h * latent_w * args.bytes_per_element

    root = Path(args.root).expanduser()
    files = sorted(root.rglob("episode*.hdf5"))
    if not files:
        p.error(f"No episode*.hdf5 under {root}")

    total_files = len(files)
    if args.limit and args.limit < total_files:
        selected = sorted(random.Random(args.seed).sample(files, args.limit))
    else:
        selected = files

    print(f"Discovered {total_files:,} episodes; inspecting {len(selected):,} episodes")
    print(f"Canvas={args.height}x{args.width}, 3-camera RGB composition (encode once)")
    print(f"Raw window={args.num_frames}, video_stride={args.video_stride}, "
          f"window_stride={args.window_stride}, sampled_frames={video_frames}")
    print(f"Latent/window=[{args.latent_channels}, {latent_t}, {latent_h}, {latent_w}], "
          f"dtype={args.bytes_per_element} B/element, size={fmt_size(per_window)}")
    if args.jpeg_step > 1:
        print(f"NOTE: JPEG sizes estimated using every {args.jpeg_step}th frame")
    print("-" * 115)
    print(f"{'Episode':42} {'T':>6} {'JPEG (3 cams)':>15} {'HDF5 total':>15} "
          f"{'Train win':>10} {'Train cache':>15} {'Val cache':>13}")

    sums = {"jpeg": 0.0, "hdf5": 0, "train_windows": 0,
            "val_windows": 0, "train_latent": 0, "val_latent": 0, "frames": 0}
    valid = 0
    missing_cam_files = 0

    for path in selected:
        try:
            with h5py.File(path, "r") as f:
                datasets = {cam: camera_dataset(f, cam) for cam in args.cameras}
                present = {cam: ds for cam, ds in datasets.items() if ds is not None}
                if not present:
                    raise ValueError("no requested camera dataset found")
                main_cam = datasets.get(args.cameras[0])
                if main_cam is None:
                    raise ValueError(f"missing target camera {args.cameras[0]}")
                t = len(main_cam)
                missing = [name for name, ds in datasets.items() if ds is None]
                if missing:
                    missing_cam_files += 1
                    print(f"  WARNING {path.name}: missing {missing}; counted as zero JPEG bytes")
                jpeg_size = sum(jpeg_size_estimate(ds, args.jpeg_step) for ds in present.values())
        except (OSError, ValueError, KeyError) as exc:
            print(f"  SKIP {path}: {exc}")
            continue

        n_train = count_windows(t, args.num_frames, args.window_stride, "train")
        n_val = count_windows(t, args.num_frames, args.window_stride, "val")
        train_cache = n_train * per_window
        val_cache = n_val * per_window
        hdf5_size = path.stat().st_size

        sums["jpeg"] += jpeg_size
        sums["hdf5"] += hdf5_size
        sums["train_windows"] += n_train
        sums["val_windows"] += n_val
        sums["train_latent"] += train_cache
        sums["val_latent"] += val_cache
        sums["frames"] += t
        valid += 1
        label = str(path.relative_to(root))
        if len(label) > 42:
            label = "..." + label[-39:]
        print(f"{label:42} {t:6d} {fmt_size(jpeg_size):>15} "
              f"{fmt_size(hdf5_size):>15} {n_train:10d} "
              f"{fmt_size(train_cache):>15} {fmt_size(val_cache):>13}")

    if not valid:
        raise SystemExit("No valid episodes inspected")

    print("-" * 115)
    print(f"Sample ({valid:,} episodes):")
    print(f"  Frames:                  {sums['frames']:,}")
    print(f"  JPEG payload (3 cams):   {fmt_size(sums['jpeg'])}")
    print(f"  Whole HDF5 files:       {fmt_size(sums['hdf5'])}")
    print(f"  Train windows / latent: {sums['train_windows']:,} / {fmt_size(sums['train_latent'])}")
    print(f"  Val windows / latent:   {sums['val_windows']:,} / {fmt_size(sums['val_latent'])}")
    print(f"  Additional disk if caching train latents: {fmt_size(sums['train_latent'])}")
    print(f"  HDF5 + train cache:     {fmt_size(sums['hdf5'] + sums['train_latent'])}")
    if sums['jpeg']:
        print(f"  Train latent / JPEG:    {sums['train_latent']/sums['jpeg']:.2f}x")
    if sums['hdf5']:
        print(f"  Train latent / HDF5:    {sums['train_latent']/sums['hdf5']:.2f}x")

    if valid < total_files:
        factor = total_files / valid
        print("\nEstimated entire data root (random sample extrapolation, NOT a full scan):")
        print(f"  Total episodes:         {total_files:,}")
        print(f"  HDF5 files:             {fmt_size(sums['hdf5'] * factor)}")
        print(f"  3-camera JPEG payload:  {fmt_size(sums['jpeg'] * factor)}")
        print(f"  Train latent cache:     {fmt_size(sums['train_latent'] * factor)}")
        print(f"  Extra disk for cache:   {fmt_size(sums['train_latent'] * factor)}")
        print(f"  HDF5 + train cache:     {fmt_size((sums['hdf5'] + sums['train_latent']) * factor)}")
    if missing_cam_files:
        print(f"\nWARNING: {missing_cam_files} inspected HDF5 files had missing views; "
              "OpenWAM may black-fill them. Review camera configuration.")
    print("\nNOTE: Latent is theoretical payload only (no file/container overhead or compression).")
    print("NOTE: Train windows include padded tail windows, matching current OpenWAM main.")
    print("NOTE: Camera images are composed into ONE RGB canvas before VAE encoding.")


if __name__ == "__main__":
    main()
