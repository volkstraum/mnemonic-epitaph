import json
from pathlib import Path
from scripts.memory_retrieval import QCCMemory  # adjust as needed

ROOT = Path(r"C:\QCC")
OUT_PATH = ROOT / "last_recall_packet.json"

def main():
    mem = QCCMemory()

    try:
        while True:
            query = input("Query> ").strip()
            if not query:
                continue

            results = mem.search(query, top_k=8)

            packet = {
                "query": query,
                "results": results,
            }

            with OUT_PATH.open("w", encoding="utf-8") as f:
                json.dump(packet, f, indent=2)

            print(f"[info] wrote {OUT_PATH}")

    except KeyboardInterrupt:
        print("\n[info] Interrupted. Exiting.")

if __name__ == "__main__":
    main()
