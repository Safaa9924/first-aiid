"""
Stage 6 — Retrieve Context
===========================
Everything needed to turn a raw user question into a ranked, deduped,
budget-aware context package:

  question -> language detection -> translation to English ->
  query expansion -> hybrid retrieval (TF-IDF + BM25 + embeddings) ->
  cross-encoder reranking -> context package
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import CrossEncoder

# ==================================================
# Retrieval / Context Configuration
# ==================================================

TOP_K = 40
TOP_N_RERANK = 10

TFIDF_WEIGHT = 0.1
BM25_WEIGHT = 0.1
SEMANTIC_WEIGHT = 0.8

MAX_CONTEXT_CHUNKS = 8
WORD_BUDGET = 1500
MAX_CHUNK_WORDS = 180

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"

QUERY_EXPANSION = {
    "burn": ["thermal burn", "chemical burn", "critical burn", "burn dressing", "cool water"],
    "fracture": ["splint", "immobilization", "broken bone"],
    "stroke": ["FAST", "facial drooping", "speech difficulty"],
    "choking": ["back blows", "abdominal thrusts", "airway obstruction"],
}


def simple_tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


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
# Query Processing
# ==================================================

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"


def translate_to_english(text):
    return GoogleTranslator(source="auto", target="en").translate(text)


def translate_to_arabic(text):
    return GoogleTranslator(source="auto", target="ar").translate(text)


def expand_query(query):
    """Expand the translated English retrieval query with domain keywords."""

    expanded_query = query
    lower_query = query.lower()

    for keyword, synonyms in QUERY_EXPANSION.items():
        if keyword in lower_query:
            expanded_query += " " + " ".join(synonyms)

    return expanded_query


def process_user_question(user_question):
    """Detects language, translates to English if needed, and expands the query."""

    language = detect_language(user_question)

    if language == "ar":
        retrieval_query = translate_to_english(user_question)
    else:
        retrieval_query = user_question

    expanded = expand_query(retrieval_query)

    return {
        "language": language,
        "retrieval_query": retrieval_query,
        "expanded_query": expanded,
    }


# ==================================================
# Retrieval Functions (TF-IDF, BM25, Semantic, Hybrid)
# ==================================================

def retrieve_top_k_tfidf(query, tfidf_vectorizer, tfidf_matrix, chunks_df, k=40):

    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    ranking = np.argsort(scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "TF-IDF"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


def retrieve_top_k_bm25(query, bm25, chunks_df, k=40):

    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranking = np.argsort(scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = np.array(scores)[ranking]
    results["retriever"] = "BM25"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


def retrieve_top_k_semantic(query, embedding_model, embedding_matrix, chunks_df, k=40):

    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embedding_matrix).flatten()

    ranking = np.argsort(scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "Embeddings"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


def retrieve_top_k_hybrid(
    query,
    tfidf_vectorizer, tfidf_matrix,
    bm25,
    embedding_model, embedding_matrix,
    chunks_df,
    tfidf_weight=TFIDF_WEIGHT,
    bm25_weight=BM25_WEIGHT,
    semantic_weight=SEMANTIC_WEIGHT,
    k=TOP_K,
):

    q_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = min_max_normalize(cosine_similarity(q_vec, tfidf_matrix).flatten())

    bm25_scores = min_max_normalize(bm25.get_scores(simple_tokenize(query)))

    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    semantic_scores = min_max_normalize(cosine_similarity(query_embedding, embedding_matrix).flatten())

    combined = (
        tfidf_weight * tfidf_scores
        + bm25_weight * bm25_scores
        + semantic_weight * semantic_scores
    )

    ranking = np.argsort(combined)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["tfidf_score"] = tfidf_scores[ranking]
    results["bm25_score"] = bm25_scores[ranking]
    results["semantic_score"] = semantic_scores[ranking]
    results["score"] = combined[ranking]
    results["retriever"] = "Hybrid"

    return results[
        ["retriever", "chunk_id", "tfidf_score", "bm25_score", "semantic_score", "score", "chunk_text"]
    ].reset_index(drop=True)


# ==================================================
# Cross-Encoder Reranking
# ==================================================

_reranker = None


def get_reranker(model_name=RERANKER_MODEL_NAME):
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(model_name)
    return _reranker


def rerank_candidates(query, candidates_df, top_n=TOP_N_RERANK):

    reranker = get_reranker()

    df = candidates_df.copy().reset_index(drop=True)
    df["original_rank"] = range(1, len(df) + 1)

    pairs = [(query, text) for text in df["chunk_text"]]
    df["rerank_score"] = reranker.predict(pairs)

    df = df.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    df["new_rank"] = range(1, len(df) + 1)

    return df.head(top_n)


# ==================================================
# Context Construction
# ==================================================

def build_context_package(
    query,
    reranked_df,
    max_context_chunks=MAX_CONTEXT_CHUNKS,
    word_budget=WORD_BUDGET,
    max_chunk_words=MAX_CHUNK_WORDS,
):
    """
    Build the final context package for the LLM.

    - Removes duplicate chunks.
    - Limits each chunk length.
    - Respects total word budget.
    - Produces clean context (no scores or metadata) plus the
      selected_df used for grounding / citations.
    """

    candidates = reranked_df.sort_values("rerank_score", ascending=False).reset_index(drop=True)

    if candidates.empty:
        return {
            "query": query,
            "selected_df": pd.DataFrame(),
            "context_text": "",
            "num_sources": 0,
            "used_words": 0,
        }

    selected_rows = []
    seen_texts = set()
    used_words = 0

    for _, row in candidates.iterrows():

        text = row["chunk_text"].strip()
        normalized = re.sub(r"\s+", " ", text).lower()

        if normalized in seen_texts:
            continue

        words = text.split()
        if len(words) > max_chunk_words:
            text = " ".join(words[:max_chunk_words])

        chunk_words = len(text.split())
        if used_words + chunk_words > word_budget:
            break

        row = row.copy()
        row["chunk_text"] = text

        selected_rows.append(row)
        seen_texts.add(normalized)
        used_words += chunk_words

        if len(selected_rows) >= max_context_chunks:
            break

    selected_df = pd.DataFrame(selected_rows)

    context_blocks = []
    for i, row in selected_df.iterrows():
        context_blocks.append(f"[Source {i + 1}]\n{row['chunk_text']}")

    context_text = ("\n\n" + "=" * 80 + "\n\n").join(context_blocks)

    return {
        "query": query,
        "selected_df": selected_df,
        "context_text": context_text,
        "num_sources": len(selected_df),
        "used_words": used_words,
    }


# ==================================================
# End-to-End Retrieval Pipeline
# ==================================================

def retrieve_context(
    user_question,
    tfidf_vectorizer, tfidf_matrix,
    bm25,
    embedding_model, embedding_matrix,
    chunks_df,
):
    """Runs the full question -> context pipeline used by the app."""

    query_info = process_user_question(user_question)

    hybrid_results = retrieve_top_k_hybrid(
        query=query_info["expanded_query"],
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        bm25=bm25,
        embedding_model=embedding_model,
        embedding_matrix=embedding_matrix,
        chunks_df=chunks_df,
        k=TOP_K,
    )

    reranked = rerank_candidates(
        query=query_info["expanded_query"],
        candidates_df=hybrid_results,
        top_n=TOP_N_RERANK,
    )

    context = build_context_package(query=query_info["expanded_query"], reranked_df=reranked)
    context["language"] = query_info["language"]

    return context
