import numpy as np
import pandas as pd
import torch
import clip
from pathlib import Path
from tqdm import tqdm

# ====== CONFIG ======

ROOT = Path("C:/QCCx")

EMB_DIR = ROOT / "embeddings" / "per_clip"
META_DIR = ROOT / "metadata"
META_DIR.mkdir(exist_ok=True)

CLIP_EMB_PATH = EMB_DIR / "clip_embeddings.npy"
CLIP_IDS_PATH = EMB_DIR / "clip_ids.npy"

OUT_CSV = META_DIR / "clip_motif_tags.csv"

# Your vocab list
VOCAB = [
    "Bedroom", "Self", "Night", "Day", "Spring", "Winter", "Summer", "Fall",
    "Rain", "Snow", "Alone", "Friends", "Fun", "Atmosphere", "Thoughts",
    "Nature", "Window", "Animals", "Close", "Far", "Now", "Past"
]

TOP_K = 5  # number of tags to keep per clip

# Optional: wrap vocab in prompts for CLIP (you can tweak these later)
def make_prompt(word: str) -> str:
    # A simple prompt template that usually works decently with CLIP
    return f"a photo about {word.lower()}"


# ====== LOAD CLIP & VOCAB EMBEDDINGS ======

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

print("Encoding vocabulary with CLIP text encoder...")
prompts = [make_prompt(w) for w in VOCAB]
text_tokens = clip.tokenize(prompts).to(device)

with torch.no_grad():
    text_embs = model.encode_text(text_tokens)
# normalize to unit length
text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
text_embs = text_embs.cpu().numpy()  # shape (V, D)

V = len(VOCAB)
print(f"Encoded {V} vocab prompts.")


# ====== LOAD CLIP EMBEDDINGS ======

print("Loading clip embeddings...")
clip_embs = np.load(CLIP_EMB_PATH)     # shape (N, D)
clip_ids = np.load(CLIP_IDS_PATH)      # shape (N,)

# Re-normalize just in case
norms = np.linalg.norm(clip_embs, axis=1, keepdims=True)
norms[norms == 0] = 1.0
clip_embs = clip_embs / norms

N, D = clip_embs.shape
print(f"Loaded {N} clip embeddings of dimension {D}.")


# ====== COMPUTE SIMILARITIES & TAGS ======

print("Computing similarities and top tags for each clip...")

# clip_embs: (N, D)
# text_embs: (V, D)
# similarity (cosine) = clip_embs @ text_embs.T  -> (N, V)
sims = clip_embs @ text_embs.T

rows = []

for i in tqdm(range(N)):
    cid = clip_ids[i]
    clip_sims = sims[i]  # shape (V,)

    # top-k vocab indices
    top_idx = np.argsort(clip_sims)[::-1][:TOP_K]

    top_words = [VOCAB[j] for j in top_idx]
    top_scores = [float(clip_sims[j]) for j in top_idx]

    row = {
        "clip_id": cid,
        "tags": ", ".join(top_words)
    }

    # also store individual tags + scores for more control later
    for rank, (w, s) in enumerate(zip(top_words, top_scores), start=1):
        row[f"tag_{rank}"] = w
        row[f"tag_{rank}_score"] = s

    rows.append(row)

df = pd.DataFrame(rows).sort_values("clip_id")

df.to_csv(OUT_CSV, index=False)
print("Saved motif tags to:", OUT_CSV)
