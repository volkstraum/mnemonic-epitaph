import os
from pathlib import Path
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm

ROOT = Path("C:/QCC")
FRAMES = ROOT / "frames"
OUT = ROOT / "embeddings" / "per_frame"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed_clip(cid):
    frame_dir = FRAMES / cid
    out_path = OUT / f"{cid}.npz"

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        return

    embs = []
    for fp in frames:
        img = preprocess(Image.open(fp).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            e = model.encode_image(img)
        e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.cpu().numpy())

    embs = np.concatenate(embs, axis=0)
    np.savez(out_path, embeddings=embs)

for cid in os.listdir(FRAMES):
    if (FRAMES / cid).is_dir():
        embed_clip(cid)
