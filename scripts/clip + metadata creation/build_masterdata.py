from pathlib import Path
import pandas as pd
import numpy as np

# ========= CONFIG =========

ROOT = Path(r"C:\QCC")
META = ROOT / "metadata"
EMB_DIR = ROOT / "embeddings" / "per_clip"

# Input files (change names here if yours differ)
CLIPS_CSV          = META / "clips.csv"
COLORS_CSV         = META / "clip_color_features.csv"       # optional
MOTIFS_CSV         = META / "clip_motif_tags.csv"           # optional
MANUAL_CSV         = META / "clip_manual_annotations.csv"   # you just made this

CLIP_EMB_NPY       = EMB_DIR / "clip_embeddings.npy"        # optional check
CLIP_IDS_NPY       = EMB_DIR / "clip_ids.npy"

# Output
MASTER_CSV         = META / "master_clips_table.csv"
MASTER_PARQUET     = META / "master_clips_table.parquet"


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV if it exists, else return empty DataFrame."""
    if not path.exists():
        print(f"[warn] {path} not found, skipping.")
        return pd.DataFrame()
    print(f"[info] Loading {path}")
    return pd.read_csv(path, **kwargs)


def main():
    # ---- 1. Base table: clips.csv ----
    clips = safe_read_csv(CLIPS_CSV)
    if clips.empty:
        raise SystemExit(f"Base file {CLIPS_CSV} missing or empty; can't build master table.")
    if "clip_id" not in clips.columns:
        raise SystemExit("clips.csv must contain a 'clip_id' column.")

    # Enforce string clip_id
    clips["clip_id"] = clips["clip_id"].astype(str)

    master = clips.copy()
    print(f"[info] Base clips rows: {len(master)}")

    # ---- 2. Merge color features ----
    colors = safe_read_csv(COLORS_CSV)
    if not colors.empty:
        colors["clip_id"] = colors["clip_id"].astype(str)
        master = master.merge(colors, on="clip_id", how="left", suffixes=("", "_color"))
        print(f"[info] After merging colors: {len(master)} rows")

    # ---- 3. Merge motif tags ----
    motifs = safe_read_csv(MOTIFS_CSV)
    if not motifs.empty:
        motifs["clip_id"] = motifs["clip_id"].astype(str)
        master = master.merge(motifs, on="clip_id", how="left", suffixes=("", "_motif"))
        print(f"[info] After merging motifs: {len(master)} rows")

    # ---- 4. Merge manual annotations ----
    manual = safe_read_csv(MANUAL_CSV)
    if not manual.empty:
        manual["clip_id"] = manual["clip_id"].astype(str)
        master = master.merge(manual, on="clip_id", how="left", suffixes=("", "_manual"))
        print(f"[info] After merging manual annotations: {len(master)} rows")

    # ---- 5. (Optional) Sanity-check against embeddings ----
    if CLIP_IDS_NPY.exists():
        clip_ids = np.load(CLIP_IDS_NPY)
        clip_ids = pd.Series(clip_ids.astype(str), name="clip_id")
        print(f"[info] Embedding clip_ids count: {len(clip_ids)}")

        # Which master clip_ids have embeddings?
        master["has_embedding"] = master["clip_id"].isin(clip_ids.values).astype(int)
        missing_emb = master.loc[master["has_embedding"] == 0, "clip_id"]
        if not missing_emb.empty:
            print(f"[warn] {len(missing_emb)} clip_ids in master table have no embedding.")
        else:
            print("[info] All master clip_ids have embeddings.")

    # ---- 6. Save outputs ----
    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_CSV, index=False)
    print(f"[info] Saved master CSV to: {MASTER_CSV}")

    # Parquet is nice to have if you start doing heavier analysis
    try:
        master.to_parquet(MASTER_PARQUET, index=False)
        print(f"[info] Saved master Parquet to: {MASTER_PARQUET}")
    except Exception as e:
        print(f"[warn] Could not write parquet: {e}")


if __name__ == "__main__":
    main()
