import numpy as np
import pandas as pd
from pathlib import Path

import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity


# ====== PATHS / CONFIG ======

ROOT = Path(r"C:\QCC")
EMB_DIR = ROOT / "embeddings" / "per_clip"
META_DIR = ROOT / "metadata"

EMB_PATH = EMB_DIR / "clip_embeddings.npy"
IDS_PATH = EMB_DIR / "clip_ids.npy"
MASTER_CSV = META_DIR / "master_clips_table.csv"

TOP_K_DEFAULT = 10


# ====== CLIP LOADING ======

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)


def encode_text(query: str) -> np.ndarray:
    """Encode a text query into a normalized CLIP embedding (1D np array)."""
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


# ====== MEMORY CLASS ======

class QCCMemory:
    def __init__(self,
                 emb_path: Path = EMB_PATH,
                 ids_path: Path = IDS_PATH,
                 master_csv: Path = MASTER_CSV,
                 notes_emb_path: Path | None = None):

        # embeddings (visual)
        self.embs = np.load(emb_path)   # (N, D)
        self.ids = np.load(ids_path).astype(str)

        norms = np.linalg.norm(self.embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embs = self.embs / norms

        # metadata
        self.meta = pd.read_csv(master_csv)
        self.meta["clip_id"] = self.meta["clip_id"].astype(str)
        self.meta["has_embedding"] = self.meta["clip_id"].isin(self.ids)

        # optional notes embeddings
        if notes_emb_path is None:
            notes_emb_path = EMB_DIR / "clip_notes_embeddings.npy"

        self.note_embs = None
        if notes_emb_path.exists():
            note_embs = np.load(notes_emb_path)
            if note_embs.shape[0] != len(self.ids):
                print("[warn] note_embs length mismatch; ignoring notes embeddings.")
            else:
                # normalize non-zero rows
                norms = np.linalg.norm(note_embs, axis=1, keepdims=True)
                zero_mask = norms == 0
                norms[zero_mask] = 1.0
                note_embs = note_embs / norms
                note_embs[zero_mask[:, 0]] = 0.0  # keep pure zeros as "no notes"
                self.note_embs = note_embs
                print("[info] Loaded notes embeddings.")

        # id mapping
        self.id_to_idx = {cid: i for i, cid in enumerate(self.ids)}

        print(f"[info] Loaded {len(self.ids)} visual embeddings.")
        print(f"[info] Master table rows: {len(self.meta)}")

    def get_row(self, clip_id: str) -> pd.Series:
        """Return metadata row for a given clip_id."""
        return self.meta.loc[self.meta["clip_id"] == clip_id].iloc[0]

    def search(self,
               query: str,
               top_k: int = TOP_K_DEFAULT,
               include_excluded: bool = False,
               mood_filter: list[str] | None = None,
               min_self_focus: int | None = None,
               max_self_focus: int | None = None,
               alpha_visual: float = 0.8   # weight on visual vs notes
               ) -> list[dict]:

        q_emb = encode_text(query)  # (D,)
        q_emb = q_emb.reshape(1, -1)

        # visual similarity
        sims_visual = cosine_similarity(self.embs, q_emb).flatten()

        # notes similarity (if available)
        if self.note_embs is not None:
            sims_notes = cosine_similarity(self.note_embs, q_emb).flatten()
        else:
            sims_notes = np.zeros_like(sims_visual)

        # blended similarity
        alpha = float(alpha_visual)
        sims = alpha * sims_visual + (1.0 - alpha) * sims_notes

        df = pd.DataFrame({
            "clip_id": self.ids,
            "similarity_visual": sims_visual,
            "similarity_notes": sims_notes,
            "similarity": sims,  # combined
        })

        df = df.merge(self.meta, on="clip_id", how="left", suffixes=("", "_meta"))

        # --- keep your existing filters ---
        if not include_excluded and "exclude_from_recall" in df.columns:
            df = df[(df["exclude_from_recall"].fillna(0) == 0)]

        if mood_filter is not None and "primary_mood" in df.columns:
            df = df[df["primary_mood"].isin(mood_filter)]

        if "self_focus" in df.columns:
            if min_self_focus is not None:
                df = df[df["self_focus"].fillna(0) >= min_self_focus]
            if max_self_focus is not None:
                df = df[df["self_focus"].fillna(5) <= max_self_focus]

        df = df.sort_values("similarity", ascending=False).head(top_k)
        # Take a wider candidate pool…
        candidates = df.head(top_k * 5).copy()

        # Add a small random jitter based on similarity
        noise = np.random.uniform(-0.02, 0.02, size=len(candidates))
        candidates["score_jittered"] = candidates["similarity"] + noise

        candidates = candidates.sort_values("score_jittered", ascending=False)
        df = candidates.head(top_k)
        results = []
        for _, row in df.iterrows():
            results.append({
                "clip_id": row["clip_id"],
                "similarity": float(row["similarity"]),
                "sim_visual": float(row["similarity_visual"]),
                "sim_notes": float(row["similarity_notes"]),
                "datetime": row.get("datetime", None),
                "short_title": row.get("short_title", None),
                "primary_mood": row.get("primary_mood", None),
                "energy_level": row.get("energy_level", None),
                "social_context": row.get("social_context", None),
                "self_focus": row.get("self_focus", None),
                "tags": row.get("tags", None),
                "graph_x": row.get("graph_x", None),
                "graph_y": row.get("graph_y", None),
            })
        return results


# ====== SIMPLE CLI FOR TESTING ======

def main():
    mem = QCCMemory()

    print("\nType a query (or just press Enter to quit):")
    while True:
        try:
            q = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            break

        results = mem.search(q, top_k=5)

        print(f"\nTop results for: '{q}'")
        for r in results:
            print(
                f"- {r['clip_id']} | sim={r['similarity']:.3f} | "
                f"{r.get('datetime', '')} | "
                f"title={r.get('short_title', '')} | "
                f"mood={r.get('primary_mood', '')} | "
                f"self_focus={r.get('self_focus', '')}"
            )

    print("\n[info] Done.")


if __name__ == "__main__":
    main()
