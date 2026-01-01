import numpy as np
import pandas as pd
from pathlib import Path

import torch
import clip

# ====== PATHS / CONFIG ======

ROOT = Path(r"C:\QCC")
EMB_DIR = ROOT / "embeddings" / "per_clip"
META_DIR = ROOT / "metadata"

IDS_PATH = EMB_DIR / "clip_ids.npy"
MASTER_CSV = META_DIR / "master_clips_table.csv"
NOTES_EMB_PATH = EMB_DIR / "clip_notes_embeddings.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)


def encode_text_batch(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into normalized CLIP text embeddings.
    Returns shape (len(texts), D).
    """
    if not texts:
        return np.empty((0, 512), dtype=np.float32)  # D will be overwritten anyway

    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)  # (B, D)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


def main():
    ids = np.load(IDS_PATH).astype(str)
    print(f"[info] Loaded {len(ids)} clip ids.")

    meta = pd.read_csv(MASTER_CSV)
    meta["clip_id"] = meta["clip_id"].astype(str)

    # Build dict: clip_id -> notes string (or None)
    # You can adjust this to concatenate other fields if you want
    id_to_notes = {}
    for _, row in meta.iterrows():
        cid = row["clip_id"]
        notes = str(row.get("notes", "") or "").strip()
        # optionally, add title/mood/etc into the text
        # title = str(row.get("short_title", "") or "").strip()
        # primary_mood = str(row.get("primary_mood", "") or "").strip()
        # combined = " | ".join(x for x in [title, primary_mood, notes] if x)
        combined = notes

        id_to_notes[cid] = combined if combined else None

    # To get D, do a dummy encode
    dummy = encode_text_batch(["dummy"])
    D = dummy.shape[1]
    print(f"[info] Using CLIP text embedding dim = {D}")

    note_embs = np.zeros((len(ids), D), dtype=np.float32)

    texts_to_encode = []
    indices = []

    for i, cid in enumerate(ids):
        txt = id_to_notes.get(cid)
        if txt is not None:
            texts_to_encode.append(txt)
            indices.append(i)
        # if txt is None, we leave the row as zeros

    print(f"[info] Encoding notes for {len(texts_to_encode)} clips (out of {len(ids)}).")

    # Encode in reasonably sized batches
    batch_size = 128
    for start in range(0, len(texts_to_encode), batch_size):
        end = start + batch_size
        batch_texts = texts_to_encode[start:end]
        batch_idxs = indices[start:end]

        batch_embs = encode_text_batch(batch_texts)  # (B, D)
        for idx, emb in zip(batch_idxs, batch_embs):
            note_embs[idx] = emb

    # Zero rows stay zero; they mean "no notes embedding"
    np.save(NOTES_EMB_PATH, note_embs)
    print(f"[info] Saved notes embeddings to: {NOTES_EMB_PATH}")


if __name__ == "__main__":
    main()
