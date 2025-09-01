# agents/alert_summary.py

import os
from agents.llm_utils import call_llm_model  # Existing helper
from transformers import pipeline  # Only used if model_mode = "local"

def generate_alert_summary(signals: str, df, model_mode: str = "cloud") -> str:
    """
    Generates a business-style summary of the proactive signals detected in the dataset.
    
    Args:
        signals (str): Raw alerts from `detect_proactive_signals(df)`
        df (pd.DataFrame): Full DataFrame for statistical reference if needed
        model_mode (str): "cloud" or "local"

    Returns:
        str: Clean, readable alert summary.
    """

    if not signals.strip():
        return "✅ No significant anomalies or issues detected in the dataset."

    # Build a structured prompt
    prompt = f"""
    You are a business analyst. Write a concise alert summary (2–4 bullet points max) based on the anomalies below.

    Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns

    Detected anomalies or signals:
    {signals}

    Focus on what changed, why it matters, and suggest 1 next step if applicable.
    Format: clean bullet points or short paragraph.
    """

    # Cloud or local model logic
    if model_mode == "cloud":
        return call_llm_model(prompt)
    
    elif model_mode == "local":
        # Load local model (Phi-2 or TinyLlama)
        local_llm = pipeline("text-generation", model="microsoft/phi-2", tokenizer="microsoft/phi-2")
        output = local_llm(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]
        return output.split(prompt)[-1].strip()
    
    else:
        return "⚠️ Unknown model mode selected."
