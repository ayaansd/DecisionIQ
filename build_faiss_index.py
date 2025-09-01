# build_faiss_index.py

import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL)


def preprocess_row(row: pd.Series) -> str:
    """
    Converts a row into a readable text format for semantic embedding.
    """
    return "; ".join([f"{col}: {val}" for col, val in row.items()])


def build_faiss_index(df: pd.DataFrame, index_path=INDEX_DIR):
    """
    Builds and saves a FAISS index from DataFrame rows.
    """
    print("🔍 Building FAISS index...")

    # Step 1: Convert rows to text
    row_texts = df.astype(str).apply(preprocess_row, axis=1).tolist()

    # Step 2: Generate embeddings
    embeddings = embedder.encode(row_texts, convert_to_numpy=True)

    # Step 3: Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Step 4: Save index
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))

    # Step 5: Save metadata
    metadata = [{"text": row_texts[i]} for i in range(len(row_texts))]
    with open(os.path.join(index_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Index built and saved to '{index_path}/'")


# Optional CLI trigger
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    build_faiss_index(df)
