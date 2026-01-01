import json
from pathlib import Path
from scripts.memory_retrieval import QCCMemory  # adjust as needed

import speech_recognition as sr  # pip install SpeechRecognition pyaudio

ROOT = Path(r"C:\QCC")
OUT_PATH = ROOT / "last_recall_packet.json"


def init_recognizer():
    """Initialize recognizer and microphone once."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    return recognizer, mic


def listen_for_query(recognizer, mic):
    """
    Record one utterance from the mic and return transcribed text.
    Returns None if nothing usable was captured.
    """
    with mic as source:
        print("\n[voice] Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("[voice] Speak your query (Ctrl+C to quit)...")
        audio = recognizer.listen(source)

    try:
        # Uses Google's free Web Speech API via SpeechRecognition.
        text = recognizer.recognize_google(audio, language="en-US")
        text = text.strip()
        if not text:
            print("[voice] Heard silence / empty text.")
            return None

        print(f"[voice] You said: {text}")
        return text

    except sr.UnknownValueError:
        print("[voice] Couldn't understand audio, try again.")
        return None
    except sr.RequestError as e:
        print(f"[voice] Speech API error: {e}")
        return None


def main():
    mem = QCCMemory()
    recognizer, mic = init_recognizer()

    print("[info] Voice-driven recall loop running. Ctrl+C to exit.")

    try:
        while True:
            query = listen_for_query(recognizer, mic)
            if not query:
                # recognition failed / silence; skip this round
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
