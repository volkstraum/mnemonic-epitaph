"""
Build a unified memory index for Quarter Century Crisis.

Inputs (expected structure):
QCC/
  embeddings/
    per_clip/
      clip_embeddings.npy
      clip_ids.npy
      clip_notes_embeddings.npy       (optional, unused for now)
    per_frame/                        (unused for now)
  metadata/
    clip_color_features.csv
    clip_motif_tags.csv
    clip_manual_annotations.csv       (optional)
    master_clips_table.parquet        (preferred if exists)
    master_clips_table.csv            (fallback)
    clips.csv                         (unused for now)
  raw_videos/                         (video files, paths may be in master_clips_table)
  scripts/
    build_memory_index.py  <-- you are here

Output:
  QCC/memory_index.json
"""

import json
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# PATH CONFIG
# =========================

# project root = parent of this script's directory
ROOT_DIR = Path(__file__).resolve().parents[1]

EMB_DIR = ROOT_DIR / "embeddings" / "per_clip"
META_DIR = ROOT_DIR / "metadata"

EMBEDDINGS_PATH = EMB_DIR / "clip_embeddings.npy"
CLIP_IDS_PATH = EMB_DIR / "clip_ids.npy"

COLOR_FEATURES_PATH = META_DIR / "clip_color_features.csv"
MOTIF_TAGS_PATH = META_DIR / "clip_motif_tags.csv"
MANUAL_ANN_PATH = META_DIR / "clip_manual_annotations.csv"

MASTER_PARQUET_PATH = META_DIR / "master_clips_table.parquet"
MASTER_CSV_PATH = META_DIR / "master_clips_table.csv"

OUTPUT_INDEX_PATH = ROOT_DIR / "memory_index.json"
VIDEO_PATH_TEMPLATE = "clips/{clip_id}.mp4"

# Graph construction params
PCA_DIM = 2
GRAPH_TOP_K = 8


# =========================
# HELPERS
# =========================
def parse_maybe_list(value):
    """
    If value is a string that looks like a Python/JSON list (e.g. "[[...], [...]]"),
    parse it into a Python object using ast.literal_eval.
    Otherwise, return it unchanged.
    """
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return ast.literal_eval(s)
            except Exception:
                # If parsing fails, just return original string
                return value
        else:
            return value
    return value

def load_clip_ids(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"clip_ids file not found: {path}")
    clip_ids = np.load(path)
    return clip_ids.astype(str)


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"clip_embeddings file not found: {path}")
    embs = np.load(path)
    if embs.ndim != 2:
        raise ValueError(f"Expected embeddings shape (N, D), got {embs.shape}")
    return embs

def to_jsonable(x):
    """
    Recursively convert numpy / pandas types to plain Python types
    so json.dump doesn't choke.
    """
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    # pandas NA handling
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x

def load_csv_aligned(path: Path, clip_ids: np.ndarray, name: str) -> pd.DataFrame:
    """
    Load a CSV and align it to clip_ids.

    If 'clip_id' column exists:
        - use it to reindex in the order of clip_ids.
    Else:
        - assume row order matches clip_ids.
    """
    if not path.exists():
        raise FileNotFoundError(f"{name} CSV not found at {path}")

    df = pd.read_csv(path)

    if "clip_id" in df.columns:
        df["clip_id"] = df["clip_id"].astype(str)
        missing = set(clip_ids) - set(df["clip_id"])
        if missing:
            print(f"WARNING: {len(missing)} clip_ids missing from {name}: {list(missing)[:10]}...")
        df = df.set_index("clip_id")
        df = df.reindex(clip_ids)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "clip_id"}, inplace=True)
    else:
        if len(df) != len(clip_ids):
            raise ValueError(
                f"{name} has {len(df)} rows but clip_ids has {len(clip_ids)} entries "
                "(add a clip_id column or fix alignment)."
            )
        df = df.copy()
        df.insert(0, "clip_id", clip_ids)

    return df


def load_master_table() -> pd.DataFrame | None:
    """
    Try to load master_clips_table (parquet preferred, CSV fallback).
    Expected to contain at least 'clip_id'. Ideally also something like
    'video_path' or 'filename' that we can use.
    """
    if MASTER_PARQUET_PATH.exists():
        df = pd.read_parquet(MASTER_PARQUET_PATH)
        print(f"Loaded master_clips_table from {MASTER_PARQUET_PATH.name}")
    elif MASTER_CSV_PATH.exists():
        df = pd.read_csv(MASTER_CSV_PATH)
        print(f"Loaded master_clips_table from {MASTER_CSV_PATH.name}")
    else:
        print("No master_clips_table found (parquet or csv). "
              "Will fall back to naive video path template.")
        return None

    if "clip_id" not in df.columns:
        raise ValueError("master_clips_table must contain a 'clip_id' column.")

    df["clip_id"] = df["clip_id"].astype(str)
    return df


def compute_pca_layout(embeddings: np.ndarray, dim: int = 2) -> np.ndarray:
    if embeddings.shape[1] < dim:
        raise ValueError(f"Embedding dim {embeddings.shape[1]} < requested PCA dim {dim}")
    pca = PCA(n_components=dim, random_state=0)
    coords = pca.fit_transform(embeddings)
    return coords


def build_similarity_edges(embeddings: np.ndarray, clip_ids: np.ndarray, top_k: int) -> list[dict]:
    print("Computing cosine similarity matrix...")
    sim = cosine_similarity(embeddings)
    N = sim.shape[0]
    edges: list[dict] = []

    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self
        neighbor_idx = np.argsort(-row)[:top_k]  # top_k largest

        for j in neighbor_idx:
            weight = float(sim[i, j])
            edges.append({
                "source": str(clip_ids[i]),
                "target": str(clip_ids[j]),
                "weight": weight,
            })

    return edges


def get_video_path_for_clip(
    clip_id: str,
    master_df: pd.DataFrame | None,
    raw_videos_dir: Path,
) -> str:
    """
    Prefer using master_clips_table if it has an explicit path.
    Otherwise, fall back to raw_videos/{clip_id}.mp4
    or something similar you can adjust later.
    """
    # Try master table
    if master_df is not None:
        row = master_df.loc[master_df["clip_id"] == clip_id]
        if len(row) == 1:
            row = row.iloc[0]
            for col in ["video_path", "filepath", "file_path", "filename"]:
                if col in row and isinstance(row[col], str):
                    return row[col]

    # Fallback: simple template; adjust if your naming is different
    candidate = raw_videos_dir / f"{clip_id}.mp4"
    return str(candidate)


# =========================
# MAIN
# =========================

def main():
    print(f"ROOT_DIR: {ROOT_DIR}")

    # 1) Load base data
    print("Loading clip_ids...")
    clip_ids = load_clip_ids(CLIP_IDS_PATH)

    print("Loading embeddings...")
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    if embeddings.shape[0] != len(clip_ids):
        raise ValueError(
            f"Embeddings rows ({embeddings.shape[0]}) != clip_ids length ({len(clip_ids)})"
        )

    # 2) Load color + motif features
    print("Loading color features...")
    df_color = load_csv_aligned(COLOR_FEATURES_PATH, clip_ids, name="clip_color_features")

    print("Loading motif tags...")
    df_motif = load_csv_aligned(MOTIF_TAGS_PATH, clip_ids, name="clip_motif_tags")

    # 3) Optional: manual annotations (e.g. emotion labels, notes)
    df_manual = None
    if MANUAL_ANN_PATH.exists():
        print("Loading manual annotations...")
        df_manual = load_csv_aligned(MANUAL_ANN_PATH, clip_ids, name="clip_manual_annotations")
    else:
        print("No clip_manual_annotations.csv found; skipping manual annotations.")

    # 4) Master table (for video paths, dates, etc.)
    master_df = load_master_table()

    # 5) 2D layout + graph edges
    print("Computing 2D PCA layout...")
    layout_2d = compute_pca_layout(embeddings, dim=PCA_DIM)

    print("Building similarity edges...")
    edges = build_similarity_edges(embeddings, clip_ids, GRAPH_TOP_K)

    # 6) Assemble clip entries
    clips_dict: dict[str, dict] = {}

    color_feature_cols = [c for c in df_color.columns if c != "clip_id"]
    motif_feature_cols = [c for c in df_motif.columns if c != "clip_id"]
    manual_feature_cols = [c for c in df_manual.columns if c != "clip_id"]

    raw_videos_dir = ROOT_DIR / "raw_videos"

    color_is_numeric = {
        col: pd.api.types.is_numeric_dtype(df_color[col])
        for col in color_feature_cols
    }
    manual_is_numeric = {
        col: pd.api.types.is_numeric_dtype(df_manual[col])
        for col in manual_feature_cols
    }
    for idx, clip_id in enumerate(clip_ids):
        color_row = df_color.iloc[idx]
        motif_row = df_motif.iloc[idx]
        manual_row = df_manual.iloc[idx]

        # -------- color_features: scalars vs palettes ----------
        color_features = {}
        for col in color_feature_cols:
            val = color_row[col]

            if pd.isna(val):
                color_features[col] = None
                continue

            if color_is_numeric[col]:
                color_features[col] = float(val)
            else:
                parsed = parse_maybe_list(val)
                color_features[col] = to_jsonable(parsed)

        # -------- motif_features: tags / one-hot / text ----------
        motif_features = {}
        for col in motif_feature_cols:
            val = motif_row[col]
            if pd.isna(val):
                motif_features[col] = None
            else:
                motif_features[col] = to_jsonable(val)

        # -------- manual_features: your hand-annotated stuff ----------
        manual_features = {}
        for col in manual_feature_cols:
            val = manual_row[col]
            if pd.isna(val):
                manual_features[col] = None
                continue

            if manual_is_numeric[col]:
                manual_features[col] = float(val)
            else:
                # allow list-like strings here too if you ever stored e.g. ["sad", "blue", "inside"]
                parsed = parse_maybe_list(val)
                manual_features[col] = to_jsonable(parsed)

        # 2D graph coordinates
        x, y = layout_2d[idx].tolist()

        clips_dict[str(clip_id)] = {
            "clip_id": str(clip_id),
            "video_path": VIDEO_PATH_TEMPLATE.format(clip_id=clip_id),
            "embedding": to_jsonable(embeddings[idx]),
            "color_features": color_features,
            "motif_features": motif_features,
            "manual_features": manual_features,
            "graph_pos": {"x": float(x), "y": float(y)},
            # "edge_frame_paths": [],
            # "pointcloud_paths": [],
        }


        # 7) Dump full index
        memory_index = {
            "clips": clips_dict,
            "graph_edges": edges,
            "meta": {
                "embedding_dim": int(embeddings.shape[1]),
                "num_clips": int(len(clip_ids)),
                "graph_top_k": int(GRAPH_TOP_K),
                "root_dir": str(ROOT_DIR),
            },
        }

    print(f"Saving memory index to {OUTPUT_INDEX_PATH} ...")
    with open(OUTPUT_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(memory_index, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
