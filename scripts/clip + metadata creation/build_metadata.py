from pathlib import Path
import pandas as pd
import re
from datetime import datetime
import subprocess
import json

ROOT = Path("C:/QCC")
RAW = ROOT / "raw_videos"
META = ROOT / "metadata"
META.mkdir(exist_ok=True)

name_pattern = re.compile(r"(\d{8})_(\d{6})")

def parse_location_tag(tag: str):
    """
    Parse strings like '+41.7845-087.6030/' → (41.7845, -87.6030)
    """
    if not tag:
        return None, None
    tag = tag.strip().rstrip("/")
    m = re.match(r'([+-]\d+\.\d+)([+-]\d+\.\d+)', tag)
    if not m:
        return None, None
    lat_str, lon_str = m.groups()
    return float(lat_str), float(lon_str)

def get_location_from_ffprobe(video_path: Path):
    """
    Use ffprobe to read format tags.location (if present).
    Returns (lat, lon) or (None, None).
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "format_tags=location",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        info = json.loads(result.stdout or "{}")
        tags = info.get("format", {}).get("tags", {})
        tag = tags.get("location") or tags.get("location-eng")
        return parse_location_tag(tag)
    except Exception:
        # if ffprobe missing or tag not found, just return None
        return None, None

def main():
    rows = []
    for f in RAW.glob("*.mp4"):
        name = f.stem
        m = name_pattern.match(name)
        if not m:
            # skip weirdly named files
            continue

        date_str, time_str = m.groups()
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

        lat, lon = get_location_from_ffprobe(f)

        rows.append({
            "clip_id": name,
            "video_file": str(f),
            "datetime": dt.isoformat(timespec="seconds"),
            "date": dt.date().isoformat(),
            "time": dt.time().isoformat(timespec="seconds"),
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "latitude": lat,
            "longitude": lon,
            "location_label": "",   # e.g. "Hyde Park", filled later by you
            "emotion_tags": "",     # to be filled manually
        })

    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df["order_index"] = df.index

    out_path = META / "clips.csv"
    df.to_csv(out_path, index=False)
    print("Wrote", out_path, "with", len(df), "rows")

if __name__ == "__main__":
    main()
