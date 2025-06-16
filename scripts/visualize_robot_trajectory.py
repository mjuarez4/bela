import os
import cv2
import numpy as np
from collections import deque
import argparse

from bela.common.dataset import make_dataset
from flax.traverse_util import flatten_dict
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="robot_output.mp4")
    parser.add_argument("--window_size", type=int, default=30)
    args = parser.parse_args()

    episode_id = int(args.filename.split("_")[-1].split(".")[0])

    class DummyCfg:
        class dataset:
            repo_id = args.repo_id
            root = None
            revision = None
            episodes = [episode_id]
            video_backend = None
            use_imagenet_stats = False
            class image_transforms:
                enable = False
        class policy:
            delta_timestamps = False
            observation_delta_indices = None

    ds = make_dataset(DummyCfg)

    print(f"Loading episode index {episode_id} from {args.repo_id}")
    episode_frames = [ex for ex in ds if ex["episode_index"] == episode_id]

    print("Available keys in one episode:")
    for key in flatten_dict(episode_frames[0]).keys():
        print(".".join(key))

    if len(episode_frames) == 0:
        raise ValueError(f"No frames found for episode {episode_id}")

    frames = []
    ee_seq = []
    for ex in episode_frames:
        img = ex["observation.image.low"]
        img_np = img.numpy()
        img_np = np.clip(img_np, 0, 1) * 255.0
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        else:
            img_np = np.transpose(img_np, (1, 2, 0))
        img_np = img_np.astype(np.uint8)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        ee_seq.append(ex["observation.state.position"].numpy()[:3])  # x, y, z

    print("Projecting and drawing end-effector trails...")
    height, width, _ = frames[0].shape
    # Normalize X/Y over the full trajectory to project into screen space
    all_xyz = np.array(ee_seq)
    x_min, y_min = all_xyz[:, 0].min(), all_xyz[:, 1].min()
    x_max, y_max = all_xyz[:, 0].max(), all_xyz[:, 1].max()

    # Avoid divide-by-zero
    x_range = x_max - x_min + 1e-6
    y_range = y_max - y_min + 1e-6


    points2d = []
    for i, pt in enumerate(ee_seq):
        if pt is None or np.any(np.isnan(pt)) or pt[2] == 0:
            points2d.append(None)
            continue

        uv_x = ((pt[0] - x_min) / x_range) * width
        uv_y = ((pt[1] - y_min) / y_range) * height
        uv = np.array([uv_x, uv_y])
        uv_rounded = np.round(uv).astype(int)
        if not (0 <= uv_rounded[0] < width and 0 <= uv_rounded[1] < height):
            print(f"[Frame {i}] ⚠️ Skipping out-of-bounds UV: {uv_rounded}")
            points2d.append(None)
            continue
        points2d.append(uv_rounded)

    output_frames = []
    for t in range(len(frames) - args.window_size):
        base = frames[t].copy()
        traj = deque()

        for i in range(1, args.window_size + 1):
            pt = points2d[t + i]
            if pt is not None:
                traj.append(pt)

        for i in range(1, len(traj)):
            p1, p2 = tuple(traj[i - 1]), tuple(traj[i])
            if all(0 <= v < width for v in [p1[0], p2[0]]) and all(0 <= v < height for v in [p1[1], p2[1]]):
                cv2.line(base, p1, p2, (0, 255, 0), 2)
                cv2.circle(base, p2, 4, (0, 255, 0), -1)

        # Red dot for current EE position
        pt = points2d[t + args.window_size]
        if pt is not None:
            cv2.circle(base, tuple(pt), 5, (0, 0, 255), -1)

        cv2.putText(base, f"Frame {t}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        output_frames.append(base)

    print("Writing video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save_path, fourcc, 15, (width, height))
    for f in output_frames:
        out.write(f)
    out.release()
    print(f"Saved video to: {args.save_path}")

if __name__ == "__main__":
    main()
