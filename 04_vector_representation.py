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
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ==================================================
# Shared Tokenizer
# ==================================================

def simple_tokenize(text):
    """Simple tokenizer for BM25 (also reused at retrieval time)."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


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
# BM25 Index
# ==================================================

class MiniBM25:

    def __init__(self, tokenized_docs, k1=1.5, b=0.75):

        self.k1 = k1
        self.b = b

        self.docs = tokenized_docs
        self.N = len(tokenized_docs)

        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = np.mean(self.doc_lens)

        self.term_freqs = [Counter(doc) for doc in tokenized_docs]

        self.df = Counter()
        for doc in tokenized_docs:
            self.df.update(set(doc))

        self.idf = {
            term: np.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

        print("=" * 60)
        print("BM25 INDEX SUMMARY")
        print("=" * 60)
        print(f"Documents      : {self.N}")
        print(f"Vocabulary     : {len(self.df):,}")
        print(f"Average Length : {self.avgdl:.1f} words")

    def get_scores(self, query_tokens):

        scores = np.zeros(self.N, dtype=np.float32)

        for term in query_tokens:

            if term not in self.idf:
                continue

            idf = self.idf[term]

            for i, tf_dict in enumerate(self.term_freqs):

                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue

                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)
                scores[i] += (idf * tf * (self.k1 + 1)) / denom

        return scores


def min_max_normalize(scores):

    scores = np.asarray(scores, dtype=np.float32)

    if scores.size == 0:
        return scores

    lo = scores.min()
    hi = scores.max()

    if hi == lo:
        return np.zeros_like(scores)

    return (scores - lo) / (hi - lo)


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

    with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    print("bm25_index.pkl saved")

    # --- Semantic Embeddings ---
    embedding_model, embedding_matrix = build_embedding_index(chunks_df)

    np.save(os.path.join(DATA_DIR, "embedding_matrix.npy"), embedding_matrix)
    print("embedding_matrix.npy saved")

    print("\nSaved indexes:", sorted(os.listdir(DATA_DIR)))
