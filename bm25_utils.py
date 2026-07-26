"""
Shared BM25 utilities
======================
`MiniBM25` used to be defined inline inside `04_vector_representation.py`.
When that script is run directly (`python 04_vector_representation.py`),
Python records the class as belonging to the `__main__` module — so a
pickled `MiniBM25` instance can only be unpickled by another script
that *also* defines that exact class inside its own `__main__`.

Moving the class here, into a normal importable module, fixes that:
both the index-building script and the Streamlit app import the same
`bm25_utils.MiniBM25`, so pickle can always resolve it.
"""

import re
from collections import Counter

import numpy as np


def simple_tokenize(text):
    """Simple tokenizer for BM25 (also reused at retrieval time)."""
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
