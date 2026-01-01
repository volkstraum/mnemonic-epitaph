import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans


ROOT = Path("C:/QCC")
FRAMES_DIR = ROOT / "frames"
META_DIR = ROOT / "metadata"
META_DIR.mkdir(exist_ok=True)

# Number of clusters for k-means
K = 5

# Downsampling to reduce computation
# (e.g., sample every Nth pixel)
PIXEL_SAMPLE_RATE = 10  # keep 1/10th of pixels


def extract_pixels_from_frame(frame_path):
    """Load image and return sampled pixels in Lab color space."""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    # convert BGR (OpenCV default) → Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # reshape to Nx3
    pixels = lab.reshape(-1, 3)

    # sample pixels to speed up k-means
    pixels = pixels[::PIXEL_SAMPLE_RATE]

    return pixels


def dominant_colors_lab(pixels, k=K):
    """Run k-means clustering to get k dominant Lab colors."""
    if pixels is None or len(pixels) == 0:
        return None

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_  # shape: (k, 3)
    return centers


def lab_to_rgb(lab_array):
    """Convert Lab values to RGB using OpenCV."""
    lab_array = np.array(lab_array, dtype=np.uint8).reshape(1, -1, 3)
    rgb = cv2.cvtColor(lab_array, cv2.COLOR_Lab2RGB)
    return rgb.reshape(-1, 3)


def process_clip(clip_id):
    """Extract k dominant colors from all frames of a clip."""

    frame_dir = FRAMES_DIR / clip_id
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        return None

    # collect pixels from multiple frames
    all_pixels = []

    for fp in frame_paths:
        px = extract_pixels_from_frame(fp)
        if px is not None:
            all_pixels.append(px)

    if not all_pixels:
        return None

    all_pixels = np.concatenate(all_pixels, axis=0)

    # run kmeans
    lab_centers = dominant_colors_lab(all_pixels, k=K)
    if lab_centers is None:
        return None

    # convert Lab → RGB
    rgb_centers = lab_to_rgb(lab_centers)

    # compute palette summary
    L = lab_centers[:, 0]        # lightness
    a = lab_centers[:, 1]
    b = lab_centers[:, 2]
    mean_L = float(np.mean(L))
    mean_sat = float(np.mean(np.sqrt(a*a + b*b)))

    return {
        "clip_id": clip_id,
        "lab_colors": lab_centers.tolist(),
        "rgb_colors": rgb_centers.tolist(),
        "mean_lightness": mean_L,
        "mean_saturation": mean_sat,
    }


def main():
    rows = []

    clip_ids = [p.name for p in FRAMES_DIR.iterdir() if p.is_dir()]
    clip_ids = sorted(clip_ids)

    print("Processing clips...")
    for cid in tqdm(clip_ids):
        result = process_clip(cid)
        if result:
            rows.append(result)

    # Convert nested lists to strings for CSV storage
    df = pd.DataFrame(rows)
    df["lab_colors"] = df["lab_colors"].apply(lambda x: str(x))
    df["rgb_colors"] = df["rgb_colors"].apply(lambda x: str(x))

    out_path = META_DIR / "clip_color_features.csv"
    df.to_csv(out_path, index=False)

    print("Saved dominant color features to:", out_path)


if __name__ == "__main__":
    main()
