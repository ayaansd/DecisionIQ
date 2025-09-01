# agents/llm_utils.py

import os
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ------------------ Constants ------------------

HF_TOKEN = os.getenv("HF_TOKEN", "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")
CLOUD_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct:together"
LOCAL_MODEL_NAME = "microsoft/phi-2"

# For Hugging Face Inference API
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ------------------ Load Local Model ------------------

try:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME)
    local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    print("✅ Loaded Phi-2 for local inference.")
except Exception as e:
    local_pipeline = None
    print(f"⚠️ Failed to load local model: {e}")

# ------------------ Main Function ------------------

def call_llm_model(prompt: str, model_name: str = CLOUD_MODEL_NAME) -> str:
    """
    Calls local (phi-2) or cloud model based on name.
    """
    if model_name == "local":
        if not local_pipeline:
            return "❌ Local model not available."
        try:
            result = local_pipeline(prompt, max_new_tokens=300, do_sample=False)[0]['generated_text']
            return result.strip()
        except Exception as e:
            return f"❌ Local model error: {e}"

    # Cloud mode (HF inference endpoint)
    try:
        response = requests.post(API_URL, headers=HEADERS, json={
            "model": CLOUD_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}]
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error calling model {model_name}: {e}"
