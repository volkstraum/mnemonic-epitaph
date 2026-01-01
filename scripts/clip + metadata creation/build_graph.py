"""
Compute a 2D layout (PCA) for all clips and store it in master_clips_table.csv
as columns graph_x, graph_y.

Run from anywhere; paths are absolute (ROOT = D:\\QCC).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

ROOT = Path(r"C:\QCC")
EMB_DIR = ROOT / "embeddings" / "per_clip"
META_DIR = ROOT / "metadata"

EMB_PATH = EMB_DIR / "clip_embeddings.npy"
IDS_PATH = EMB_DIR / "clip_ids.npy"
MASTER_CSV = META_DIR / "master_clips_table.csv"

OUT_CSV = MASTER_CSV  # overwrite in-place; change if you want a copy instead

PCA_DIM = 2


def main():
    print("[info] loading embeddings + ids...")
    embs = np.load(EMB_PATH)            # (N, D)
    ids = np.load(IDS_PATH).astype(str) # (N,)

    if embs.shape[0] != len(ids):
        raise ValueError(f"embeddings rows ({embs.shape[0]}) != ids length ({len(ids)})")

    print("[info] computing PCA layout...")
    pca = PCA(n_components=PCA_DIM, random_state=0)
    coords = pca.fit_transform(embs)    # (N, 2)

    layout_df = pd.DataFrame({
        "clip_id": ids,
        "graph_x": coords[:, 0],
        "graph_y": coords[:, 1],
    })

    print("[info] loading master_clips_table...")
    master = pd.read_csv(MASTER_CSV)
    master["clip_id"] = master["clip_id"].astype(str)

    print("[info] merging layout into master table...")
    master = master.merge(layout_df, on="clip_id", how="left")

    missing = master["graph_x"].isna().sum()
    if missing > 0:
        print(f"[warn] {missing} rows did not get graph coords (no embedding?)")

    print(f"[info] writing updated master table to {OUT_CSV} ...")
    master.to_csv(OUT_CSV, index=False)
    print("[info] done.")


if __name__ == "__main__":
    main()
