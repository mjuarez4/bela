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
    parser.add_argument("--save_path", type=str, default="output.mp4")
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
    if len(episode_frames) == 0:
        raise ValueError(f"No frames found for episode {episode_id}")

    frames = []
    kp3d_seq = []
    for ex in episode_frames:
        img = ex["observation.image.low"]
        img_np = img.numpy()

        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        else:
            img_np = np.transpose(img_np, (1, 2, 0))  # C,H,W -> H,W,C

        if img_np.max() <= 1.0:
            img_np = (img_np * 255).clip(0, 255)
        img_np = img_np.astype(np.uint8)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        frames.append(img_bgr)
        kp3d_seq.append(ex["observation.state.kp3d"].numpy())

    print("Projecting and drawing palm trails...")
    height, width, _ = frames[0].shape
    center = np.array([width / 2, height / 2])
    focal_val = episode_frames[0]["observation.state.scaled_focal_length"]
    focal_val = focal_val.item() if isinstance(focal_val, torch.Tensor) else focal_val
    focal = np.array([focal_val, focal_val])

    points2d = []
    palm_traj = []
    print(f"Num frames: {len(frames)} | Num kp3d entries: {len(kp3d_seq)}")
    for i, kp3d_frame in enumerate(kp3d_seq):
        try:
            print(f"[Frame {i}] kp3d shape: {kp3d_frame.shape}")
            if kp3d_frame.shape == (21, 3):
                kp3d = kp3d_frame[0]
            elif kp3d_frame.shape == (3,):
                kp3d = kp3d_frame
            else:
                print(f"[Frame {i}] Unexpected kp3d shape: {kp3d_frame.shape}")
                kp3d = None
        except Exception as e:
            print(f"[Frame {i}] Error parsing kp3d: {e}")
            kp3d = None

        if kp3d is None or np.any(np.isnan(kp3d)) or kp3d[2] == 0:
            points2d.append(None)
            continue

        uv = focal * (np.array([kp3d[0], kp3d[1]]) / kp3d[2]) + center
        points2d.append(np.round(uv).astype(int))
        palm_traj.append(kp3d)

    entered_left_half = any(pt is not None and pt[0] < width / 2 for pt in points2d)
    print("Palm entered left half of screen." if entered_left_half else "Warning: Palm never entered left half.")

    print("Computing palm movement...")
    palm_traj = np.array(palm_traj)
    if len(palm_traj) >= 2:
        diffs = np.linalg.norm(palm_traj[1:] - palm_traj[:-1], axis=1)
        print(f"Average palm movement: {diffs.mean():.5f}")
        print(f"Max frame-to-frame movement: {diffs.max():.5f}")
    else:
        print("Not enough palm points to compute movement.")

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
                cv2.line(base, p1, p2, (0, 100 + i * 5, 255 - i * 5), 2)
                cv2.circle(base, p2, 3, (255, 255, 255), -1)

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
