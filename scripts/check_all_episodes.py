import os
import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "Jgold90/sweep_mano"
SAVE_PATH = "bad_episodes.txt"
CHUNK_PATH = "data/chunk-000"

def palm_entered_left_half(points2d, width):
    for pt in points2d:
        if pt is not None and pt[0] < width / 2:
            return True
    return False

def check_episode(filename):
    try:
        local_path = hf_hub_download(repo_id=REPO_ID, filename=f"{CHUNK_PATH}/{filename}", repo_type="dataset")
        df = pd.read_parquet(local_path)

        frames = []
        for item in df["observation.image.low"]:
            img_bytes = item["bytes"]
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        height, width, _ = frames[0].shape
        focal = df["observation.state.scaled_focal_length"].iloc[0]
        if isinstance(focal, float) or isinstance(focal, np.float32):
            focal = np.array([focal, focal])
        center = np.array([width / 2, height / 2])

        points2d = []
        for i in range(len(frames)):
            kp3d_raw = df["observation.state.kp3d"].iloc[i]
            try:
                kp3d_frame = np.stack(kp3d_raw)
                if kp3d_frame.shape == (21, 3):
                    kp3d = kp3d_frame[0]
                else:
                    kp3d = None
            except:
                kp3d = None

            if kp3d is None or np.any(np.isnan(kp3d)) or kp3d[2] == 0:
                points2d.append(None)
                continue

            uv = focal * (np.array([kp3d[0], kp3d[1]]) / kp3d[2]) + center
            points2d.append(np.round(uv).astype(int))

        entered = palm_entered_left_half(points2d, width)
        return not entered  # Return True if BAD (never entered left)

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return False

def main():
    bad_episodes = []
    for i in range(500):
        filename = f"episode_{i:06d}.parquet"
        is_bad = check_episode(filename)
        status = "BAD" if is_bad else "GOOD"
        print(f"{filename}: {status}")
        if is_bad:
            bad_episodes.append(filename)

    with open(SAVE_PATH, "w") as f:
        for ep in bad_episodes:
            f.write(ep + "\n")

    print(f"\nScan complete. Found {len(bad_episodes)} bad episodes.")
    print(f"Saved results to {SAVE_PATH}")

if __name__ == "__main__":
    main()
