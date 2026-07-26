"""
Stage 5 — Create Chroma Vector Store
=====================================
Populates a persistent Chroma collection with the chunk embeddings
built in Stage 4, so the retrieval stage can query it directly instead
of recomputing embeddings on every run.
"""

import os

import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "first_aid_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def create_chroma_store(chunks_df, embedding_matrix, chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):

    client = chromadb.PersistentClient(path=chroma_path)

    print("=" * 60)
    print("CHROMA DATABASE")
    print("=" * 60)
    print("Database Path :", chroma_path)

    collection = client.get_or_create_collection(name=collection_name)
    print("Collection    :", collection_name)

    # Remove previous data (fresh build)
    existing = collection.count()
    if existing > 0:
        print(f"Existing Documents : {existing}")
        client.delete_collection(collection_name)
        collection = client.get_or_create_collection(name=collection_name)
        print("Old collection removed.")

    collection.add(
        ids=chunks_df["chunk_id"].astype(str).tolist(),
        documents=chunks_df["chunk_text"].tolist(),
        embeddings=embedding_matrix.tolist(),
        metadatas=[
            {"chunk_id": str(row.chunk_id)}
            for row in chunks_df.itertuples()
        ],
    )

    print("=" * 60)
    print("CHROMA STORE CREATED")
    print("=" * 60)
    print("Stored Documents :", collection.count())

    return client, collection


if __name__ == "__main__":

    chunks_df = pd.read_csv(os.path.join(DATA_DIR, "chunks.csv"), encoding="utf-8-sig")

    embedding_path = os.path.join(DATA_DIR, "embedding_matrix.npy")

    if os.path.exists(embedding_path):
        embedding_matrix = np.load(embedding_path)
    else:
        # Fallback: recompute embeddings if Stage 4's .npy isn't available
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embedding_matrix = model.encode(
            chunks_df["chunk_text"].tolist(),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    create_chroma_store(chunks_df, embedding_matrix)
