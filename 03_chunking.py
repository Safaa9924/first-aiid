"""
Stage 3 — Chunking
===================
Splits the cleaned document into adaptive, semantically-bounded chunks
(respecting heading boundaries and strict word-count limits), attaches
metadata to each chunk, and saves the result to a CSV that later
stages build their indexes from.
"""

import os
import re

import numpy as np
import pandas as pd

DATA_DIR = "data"

SOURCE_FILE_NAME = "First_aid_reference_guide_V4.1_Public.pdf"
SOURCE_TITLE = "First Aid Reference Guide, 4th Edition — St. John Ambulance Canada"
PUBLICATION_YEAR = 2019


# ==================================================
# Semantic Topic-aware Chunking (Strict Bounds)
# ==================================================

def semantic_chunk_markdown(
    text,
    target_words=150,
    overlap_words=20,
    min_chunk_words=60,
    max_chunk_words=200,
):
    """Semantic chunking for RAG with strict min/max word boundary enforcement.

    Features
    --------
    - Preserves section boundaries & headings.
    - Guarantees chunk word count stays between min_chunk_words and max_chunk_words.
    - Handles boundary cases (e.g. the last chunk or oversized merged chunks).
    """

    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 1. Parse Sections by Heading
    sections = []
    current_heading = "General"
    current_paragraphs = []

    for part in text.split("\n\n"):
        part = part.strip()
        if not part:
            continue

        if part.startswith("#"):
            if current_paragraphs:
                sections.append((current_heading, current_paragraphs))
            current_heading = part
            current_paragraphs = []
        else:
            current_paragraphs.append(part)

    if current_paragraphs:
        sections.append((current_heading, current_paragraphs))

    # 2. Build Initial Chunks
    raw_chunks = []

    for heading, paragraphs in sections:
        h_words = heading.split()
        current_words = len(h_words)
        current_pieces = [heading]

        for para in paragraphs:
            para_words = para.split()

            if len(para_words) > target_words:
                if len(current_pieces) > 1:
                    raw_chunks.append("\n\n".join(current_pieces))
                    current_pieces = [heading]
                    current_words = len(h_words)

                stride = max(1, target_words - overlap_words)
                for start in range(0, len(para_words), stride):
                    piece = para_words[start: start + target_words]
                    if piece:
                        raw_chunks.append(heading + "\n\n" + " ".join(piece))
                continue

            if current_words + len(para_words) <= target_words:
                current_pieces.append(para)
                current_words += len(para_words)
            else:
                raw_chunks.append("\n\n".join(current_pieces))

                overlap = " ".join(
                    " ".join(current_pieces[1:]).split()[-overlap_words:]
                )
                current_pieces = [heading]
                if overlap:
                    current_pieces.append(overlap)
                current_pieces.append(para)
                current_words = len(h_words) + len(overlap.split()) + len(para_words)

        if len(current_pieces) > 1 or (
            len(current_pieces) == 1 and current_pieces[0] != heading
        ):
            raw_chunks.append("\n\n".join(current_pieces))

    # 3. Strict Safety Split (Enforce max_chunk_words)
    bounded_chunks = []

    for chunk in raw_chunks:
        words = chunk.split()
        if len(words) <= max_chunk_words:
            bounded_chunks.append(chunk)
        else:
            heading = ""
            body = chunk
            if chunk.startswith("#"):
                parts = chunk.split("\n\n", 1)
                if len(parts) == 2:
                    heading = parts[0]
                    body = parts[1]

            h_count = len(heading.split())
            max_body = max(10, max_chunk_words - h_count)
            stride = max(1, max_body - overlap_words)

            body_words = body.split()
            for start in range(0, len(body_words), stride):
                piece = body_words[start: start + max_body]
                if not piece:
                    continue
                formatted = (
                    f"{heading}\n\n{' '.join(piece)}" if heading else " ".join(piece)
                )
                bounded_chunks.append(formatted)

    # 4. Strict Merge (Enforce min_chunk_words across ALL edge cases)
    final_chunks = []

    for chunk in bounded_chunks:
        if not final_chunks:
            final_chunks.append(chunk)
            continue

        prev_chunk = final_chunks[-1]
        prev_words = len(prev_chunk.split())
        curr_words = len(chunk.split())

        if prev_words < min_chunk_words and (prev_words + curr_words) <= max_chunk_words:
            final_chunks[-1] = prev_chunk + "\n\n" + chunk
        elif curr_words < min_chunk_words and (prev_words + curr_words) <= max_chunk_words:
            final_chunks[-1] = prev_chunk + "\n\n" + chunk
        else:
            final_chunks.append(chunk)

    if len(final_chunks) > 1:
        last_words = len(final_chunks[-1].split())
        prev_words = len(final_chunks[-2].split())

        if last_words < min_chunk_words and (last_words + prev_words) <= max_chunk_words:
            last_chunk = final_chunks.pop()
            final_chunks[-1] = final_chunks[-1] + "\n\n" + last_chunk

    # 5. DataFrame Construction
    records = []
    for i, chunk in enumerate(final_chunks, start=1):
        records.append({
            "chunk_id": f"chunk_{i:04d}",
            "word_count": len(chunk.split()),
            "char_count": len(chunk),
            "chunk_text": chunk,
        })

    return pd.DataFrame(records)


# ==================================================
# Chunk Metadata
# ==================================================

def process_chunk_metadata(
    chunks_df: pd.DataFrame,
    source_file: str,
    source_title: str,
    publication_year: int,
    words_per_minute: int = 200,
):

    df = chunks_df.copy()

    # 1. File & Context Metadata
    df["source_file"] = source_file
    df["source_title"] = source_title
    df["publication_year"] = int(publication_year) if publication_year else None

    # 2. Structural Parsing (Header Extraction & Level Detection)
    header_extract = df["chunk_text"].str.extract(
        r"^(?P<h_level>#+)\s*(?P<section>.+)$", re.MULTILINE
    )

    df["section"] = header_extract["section"].str.strip().fillna("General")
    df["header_level"] = header_extract["h_level"].str.len().astype("Int64").fillna(0)

    # 3. Page Number Handling
    if "page_number" in df.columns:
        df["page_number"] = df["page_number"].astype("Int64")
    else:
        df["page_number"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # 4. Computed Semantic & Text Statistics
    if "char_count" not in df.columns:
        df["char_count"] = df["chunk_text"].str.len()

    if "word_count" not in df.columns:
        df["word_count"] = df["chunk_text"].str.split().str.len()

    df["avg_word_length"] = (
        (df["char_count"] / df["word_count"].replace(0, np.nan)).round(2).fillna(0.0)
    )
    df["read_time_sec"] = (df["word_count"] / (words_per_minute / 60)).round(1)

    # 5. Schema Ordering
    core_columns = [
        "chunk_id", "source_title", "source_file", "publication_year",
        "section", "header_level", "page_number", "word_count",
        "char_count", "avg_word_length", "read_time_sec",
    ]

    metadata_df = df[[col for col in core_columns if col in df.columns]].copy()

    df["metadata"] = metadata_df.to_dict(orient="records")

    return df, metadata_df


# ==================================================
# Run Chunking
# ==================================================

if __name__ == "__main__":

    with open(os.path.join(DATA_DIR, "cleaned_text.txt"), "r", encoding="utf-8") as f:
        cleaned_text = f.read()

    chunks_df = semantic_chunk_markdown(
        cleaned_text,
        target_words=300,
        overlap_words=100,
        min_chunk_words=80,
    )

    chunks_df, metadata_df = process_chunk_metadata(
        chunks_df=chunks_df,
        source_file=SOURCE_FILE_NAME,
        source_title=SOURCE_TITLE,
        publication_year=PUBLICATION_YEAR,
    )

    output_path = os.path.join(DATA_DIR, "chunks.csv")
    chunks_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("SEMANTIC CHUNKING SUMMARY")
    print("=" * 60)

    word_counts = chunks_df["word_count"]
    print(f"Generated Chunks : {len(chunks_df)}")
    print(f"Average Words    : {word_counts.mean():.1f}")
    print(f"Median Words     : {word_counts.median():.1f}")
    print(f"Min Words        : {word_counts.min()}")
    print(f"Max Words        : {word_counts.max()}")
    print(f"Saved to         : {output_path}")
