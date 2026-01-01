import numpy as np
from pathlib import Path
import pandas as pd

ROOT = Path("C:/QCC")
PER_FRAME = ROOT / "embeddings/per_frame"
PER_CLIP = ROOT / "embeddings/per_clip"
PER_CLIP.mkdir(exist_ok=True)

embs = []
ids = []

for npz in sorted(PER_FRAME.glob("*.npz")):
    cid = npz.stem
    arr = np.load(npz)["embeddings"]
    vec = arr.mean(axis=0)
    vec = vec / np.linalg.norm(vec)

    embs.append(vec)
    ids.append(cid)

embs = np.stack(embs)
np.save(PER_CLIP / "clip_embeddings.npy", embs)
np.save(PER_CLIP / "clip_ids.npy", np.array(ids))
