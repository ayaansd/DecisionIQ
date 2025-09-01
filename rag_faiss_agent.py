# rag_faiss_agent.py

import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from agents.qa_agent import call_llm  # Reuse cloud/local toggle logic

EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)


def retrieve_similar_rows(question, index_path="faiss_index", top_k=5):
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_vector = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(query_vector, top_k)
    results = [metadata[i]["text"] for i in I[0] if i < len(metadata)]
    return results


def generate_rag_prompt(question, retrieved_rows):
    context = "\n".join([f"- {row}" for row in retrieved_rows])
    return f"""
You are a helpful analyst. Use the following context rows to answer the user's question in one line.

Context rows:
{context}

Question: {question}

Answer:
"""


def generate_explanation_prompt(question, rows, answer):
    context = "\n".join([f"- {row}" for row in rows])
    return f"""
Explain how the following context helps answer the question.

Question: {question}
Context rows:
{context}
Answer:
{answer}

Explanation:
"""


def generate_cot_prompt(question, rows, answer):
    context = "\n".join([f"- {row}" for row in rows])
    return f"""
Think through the logic step-by-step as an expert data analyst.

1. What is the user asking?
2. What clues do the retrieved rows provide?
3. How do those rows relate to the question?
4. Finally, explain why the answer is correct.

Question: {question}
Context rows:
{context}
Answer: {answer}

Chain-of-Thought:
"""


def run_faiss_rag_agent(question, model_mode="cloud", index_path="faiss_index", top_k=5):
    retrieved_rows = retrieve_similar_rows(question, index_path, top_k)
    if not retrieved_rows:
        return {
            "answer": "⚠️ No relevant data rows found.",
            "explanation": "N/A",
            "cot_reasoning": "N/A",
            "context_rows": [],
        }

    prompt = generate_rag_prompt(question, retrieved_rows)
    answer = call_llm(prompt, model_mode=model_mode, max_new_tokens=300)

    explanation_prompt = generate_explanation_prompt(question, retrieved_rows, answer)
    explanation = call_llm(explanation_prompt, model_mode=model_mode, max_new_tokens=300)

    cot_prompt = generate_cot_prompt(question, retrieved_rows, answer)
    cot = call_llm(cot_prompt, model_mode=model_mode, max_new_tokens=300)

    return {
        "answer": answer,
        "explanation": explanation,
        "cot_reasoning": cot,
        "context_rows": retrieved_rows
    }
##  stil lcan t see charts on the display it only downlaods ,erro with summary , no q?a is displayed 