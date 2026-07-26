"""
Stage 4 — Vector Representation
================================
Builds the three retrieval indexes used by the hybrid retriever:
a TF-IDF index, a BM25 index, and a dense semantic embedding matrix
(SentenceTransformer). Each index is persisted to disk so later
stages (Chroma store creation, retrieval) can load them without
recomputation.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from bm25_utils import simple_tokenize, MiniBM25

DATA_DIR = "data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ==================================================
# TF-IDF Index
# ==================================================

def build_tfidf_index(texts):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
        dtype="float32",
    )

    matrix = vectorizer.fit_transform(texts)

    print("=" * 60)
    print("TF-IDF INDEX SUMMARY")
    print("=" * 60)
    print(f"Documents       : {len(texts)}")
    print(f"Vocabulary Size : {len(vectorizer.vocabulary_):,}")
    print(f"Matrix Shape    : {matrix.shape}")
    print(f"Non-zero Terms  : {matrix.nnz:,}")

    sparsity = (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
    print(f"Sparsity        : {sparsity:.2f}%")

    return vectorizer, matrix


# ==================================================
# Semantic Embedding Index
# ==================================================

def build_embedding_index(chunks_df, model_name=EMBEDDING_MODEL_NAME):

    print("=" * 60)
    print("BUILDING EMBEDDING INDEX")
    print("=" * 60)

    model = SentenceTransformer(model_name)

    texts = chunks_df["chunk_text"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Embedding Model : {model_name}")
    print(f"Documents       : {len(texts)}")
    print(f"Embedding Shape : {embeddings.shape}")

    return model, embeddings


# ==================================================
# Run Vector Representation
# ==================================================

if __name__ == "__main__":

    chunks_df = pd.read_csv(os.path.join(DATA_DIR, "chunks.csv"), encoding="utf-8-sig")
    texts = chunks_df["chunk_text"].tolist()

    # --- TF-IDF ---
    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(texts)

    with open(os.path.join(DATA_DIR, "tfidf_index.pkl"), "wb") as f:
        pickle.dump({"vectorizer": tfidf_vectorizer, "matrix": tfidf_matrix}, f)
    print("tfidf_index.pkl saved")

    # --- BM25 ---
    tokenized_docs = [simple_tokenize(text) for text in texts]
    bm25 = MiniBM25(tokenized_docs)

    print("=" * 60)
    print("BM25 INDEX SUMMARY")
    print("=" * 60)
    print(f"Documents      : {bm25.N}")
    print(f"Vocabulary     : {len(bm25.df):,}")
    print(f"Average Length : {bm25.avgdl:.1f} words")

    with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    print("bm25_index.pkl saved")

    # --- Semantic Embeddings ---
    embedding_model, embedding_matrix = build_embedding_index(chunks_df)

    np.save(os.path.join(DATA_DIR, "embedding_matrix.npy"), embedding_matrix)
    print("embedding_matrix.npy saved")

    print("\nSaved indexes:", sorted(os.listdir(DATA_DIR)))
