
# Mnemonic Epitaph
![Capture](https://github.com/user-attachments/assets/8d6da50e-1038-485a-b6f0-637eff36c828)


Mnemonic Epitaph is an exploration in the artificial reconstruction of memory with machine learning.  
Here, memory as a digital construct loses its perfect replayability, instead becoming approximate, associative, and lossy, shaped by interpretation rather than retrieval.

At its core, the system uses semantic embeddings to approximate how fragments of lived experience might be recalled in response to vague, spoken prompts—producing imperfect, shifting recollections rather than exact matches.

## What this repository contains

This repository holds the **Python engine** behind *Mnemonic Epitaph*.  
It is responsible for:

- Generating CLIP embeddings from a dataset of personal video fragments  
- Accepting live voice input via speech-to-text  
- Translating spoken language into semantic queries  
- Retrieving related video memories based on similarity rather than identity  
- Emitting structured “recall packets” consumed by an external visualization system (TouchDesigner)

## System overview

The system operates as a pipeline:

1. **Dataset ingestion**  
   A collection of short video clips is paired with metadata describing time, affect, and context.

2. **Embedding generation**  
   Each video is embedded using CLIP, producing a semantic representation that abstracts away from the original footage.

3. **Voice input**  
   Spoken input is captured and transcribed using the Google Web Speech API.

4. **Semantic matching**  
   The transcribed text is embedded and compared against the dataset embeddings using similarity metrics.

5. **Recall packet construction**  
   Instead of returning a single “correct” memory, the system assembles a weighted recall packet describing a cluster of related fragments.

6. **External visualization**  
   The recall packet is streamed to TouchDesigner, where metadata is mapped onto a responsive, animated visual system that mimics the instability of memory.

## Repository structure

A high-level overview of the project layout:

Some files in this repository are **runtime artifacts** rather than source code.  
In practice, embeddings and recall outputs are often excluded from version control and regenerated as needed.

## Statement

Mnemonic Epitaph is a thought which articulates the paralells of memory re-construction in humans and machine. The dataset of the project was a series of 15 second videos recorded daily during 2025. These moments exist simultaneously as lived experience and infutable proof of its existence. Mnemonic Epitaph deconstructs the perfect reconstruction of these memories, these existences and artificially replicates the reconstruction process of human memory through sound, neural connections, mood, and emotion. It acts as an alternative expression to what could be "artificial" in the tension between human and technology.

## Requirements

- Python 3.x  
- PyTorch  
- CLIP  
- SpeechRecognition (Google Web Speech API backend)  
- NumPy  
- A compatible GPU is strongly recommended for embedding generation

(Exact dependency versions are intentionally left flexible, as this project evolved through experimentation rather than fixed deployment.)

## Notes

- This project processes voice input and personal media. Dataset ownership, consent, and privacy are critical considerations.
- The system is designed for **small, personal datasets**, not large-scale archival retrieval.
- Outputs are intentionally unstable; repeated queries may yield different recall structures.

## Status

This repository reflects a **completed exploratory system** developed as part of an art and machine intelligence course.  
Future work may include modularizing the embedding and recall logic further, or extending the recall packet format for alternative visualization strategies.

