# embedding_index.py

import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load embedder
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

def embed_row(row: pd.Series) -> str:
    """Convert a row into a single string representation."""
    return " | ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])

def build_faiss_index(df: pd.DataFrame, index_save_path="faiss_index"):
    """
    Build a FAISS index from row-level data and save metadata.

    Args:
        df (pd.DataFrame): Input DataFrame
        index_save_path (str): Folder to save FAISS index and metadata
    """
    os.makedirs(index_save_path, exist_ok=True)

    texts = df.apply(embed_row, axis=1).tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, os.path.join(index_save_path, "index.faiss"))

    # Save metadata
    metadata = [{"row": i, "text": texts[i]} for i in range(len(texts))]
    with open(os.path.join(index_save_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ FAISS index built with {len(texts)} rows.")


def load_faiss_index(index_path="faiss_index"):
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata
