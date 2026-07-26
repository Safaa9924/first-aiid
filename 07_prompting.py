"""
Stage 7 — Prompting & Generation
=================================
Builds the grounded prompt from the retrieved context, calls a local
Ollama LLM to generate the English answer, translates it to Arabic,
and produces a grounding/confidence report with source citations.
"""

import time

import requests
import pandas as pd

from importlib import import_module

retrieve_context_module = import_module("06_retrieve_context")

# ==================================================
# Configuration
# ==================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

TEMPERATURE = 0.1
MAX_TOKENS = 2400
SEED = 42

# Minimum top rerank score required to trust the retrieved evidence.
CONFIDENCE_THRESHOLD = 1.5


# ==================================================
# Prompt Builder
# ==================================================

def build_chat_prompt(condition: str, context: str):

    if not context or not context.strip():
        raise ValueError("Retrieved context is empty.")

    prompt = f"""You are an expert Evidence-Based First Aid Assistant.
Answer the user question strictly in ENGLISH using ONLY the provided context. Never add outside knowledge.
If the answer is not in the context, output: "I couldn't find this information in the retrieved first aid reference."

CORE RULES:
1. Concise & Direct: Max 5 bullet points per section. No introductions or summaries.
2. No Repetition: Mention each piece of advice only once.
3. Omit Empty Sections: If a section has no context data, skip its header completely.
4. Clean Markdown: Strictly follow the structure below.

STRUCTURE TO FOLLOW:
## First Aid: {condition}

## Immediate Actions
- [Essential steps, max 5 items]

## Avoid
- [Warnings directly related, max 5 items]

## When to Call Emergency Services
- [Specific situations]

## Additional Notes
- [Extra crucial info, if any]

## Evidence Source
- [Source/Organization name from context, or 'Retrieved first aid reference document']

============================
USER QUESTION: {condition}
============================
RETRIEVED CONTEXT:
{context}
"""
    return prompt


# ==================================================
# LLM Generation (Ollama)
# ==================================================

def generate_answer(prompt, model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": seed,
            "top_k": 20,
            "top_p": 0.8,
            "repeat_penalty": 1.1,
        },
    }

    start = time.time()

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()

        elapsed = time.time() - start
        result = response.json()
        answer = result.get("response", "").strip()

        if not answer:
            answer = "The language model returned an empty response."

        print("=" * 60)
        print("LLM GENERATION STATS")
        print("=" * 60)
        print("Model            :", result.get("model"))
        print(f"Inference Time   : {elapsed:.2f} sec")
        print("Generated Tokens :", result.get("eval_count", "N/A"))

        if result.get("done_reason") == "length":
            print("Warning: Maximum token limit reached.")

        return answer

    except Exception as e:
        print("=" * 60)
        print("OLLAMA ERROR")
        print("=" * 60)
        print(e)
        return None


# ==================================================
# Confidence Classification & Grounding Report
# ==================================================

def confidence_label(score, best):

    if best <= 0:
        return "Low"

    ratio = score / best

    if ratio >= 0.80:
        return "High"
    elif ratio >= 0.50:
        return "Medium"
    else:
        return "Low"


def build_grounding_report(selected_df: pd.DataFrame):

    if selected_df.empty:
        return {
            "num_sources": 0,
            "top_score": 0.0,
            "grounded_sources": pd.DataFrame(),
        }

    best_score = selected_df["rerank_score"].max()

    grounded_sources = (
        selected_df[["chunk_id", "rerank_score"]]
        .copy()
        .sort_values("rerank_score", ascending=False)
        .reset_index(drop=True)
    )

    grounded_sources.insert(0, "Rank", range(1, len(grounded_sources) + 1))
    grounded_sources["confidence"] = grounded_sources["rerank_score"].apply(
        lambda x: confidence_label(x, best_score)
    )

    return {
        "num_sources": len(grounded_sources),
        "top_score": float(best_score),
        "grounded_sources": grounded_sources,
    }


# ==================================================
# Full Grounded-Answer Pipeline
# ==================================================

def generate_grounded_answer(user_question, context):
    """
    Takes the context package produced by `retrieve_context` in Stage 6
    and produces the final bilingual, grounded answer.
    """

    selected_df = context["selected_df"]

    if selected_df.empty:
        english_answer = "I couldn't find this information in the retrieved first aid reference."
        arabic_answer = retrieve_context_module.translate_to_arabic(english_answer)
        return {
            "english_answer": english_answer,
            "arabic_answer": arabic_answer,
            "grounding": build_grounding_report(selected_df),
        }

    best_score = float(selected_df["rerank_score"].max())

    if best_score >= CONFIDENCE_THRESHOLD:
        prompt = build_chat_prompt(condition=user_question, context=context["context_text"])
        english_answer = generate_answer(prompt)

        if english_answer and "I couldn't find this information" not in english_answer:
            arabic_answer = retrieve_context_module.translate_to_arabic(english_answer)
        else:
            arabic_answer = "لم أتمكن من العثور على هذه المعلومات في مرجع الإسعافات الأولية المسترجع."
    else:
        english_answer = (
            "Low-confidence retrieval. The retrieved evidence is insufficient to generate a reliable answer."
        )
        arabic_answer = "نتيجة الاسترجاع ضعيفة. الأدلة المسترجعة غير كافية لتقديم إجابة موثوقة."

    return {
        "english_answer": english_answer,
        "arabic_answer": arabic_answer,
        "grounding": build_grounding_report(selected_df),
    }
