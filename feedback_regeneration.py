import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
# Constants
HF_TOKEN = os.getenv("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")  # From .env or terminal
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Load local model once
local_tokenizer = None
local_model = None
local_pipe = None

def load_local_model():
    global local_tokenizer, local_model, local_pipe
    if not local_pipe:
        model_id = "microsoft/phi-2"
        local_tokenizer = AutoTokenizer.from_pretrained(model_id)
        local_model = AutoModelForCausalLM.from_pretrained(model_id)
        local_pipe = pipeline("text-generation", model=local_model, tokenizer=local_tokenizer)
    return local_pipe

def summarize_schema(df: pd.DataFrame) -> str:
    schema = "\n".join([f"- {col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
    sample_rows = df.sample(n=min(2, len(df))).to_string(index=False)
    return f"Dataset schema:\n{schema}\n\nSample rows:\n{sample_rows}"

def call_cloud_summary(prompt: str) -> str:
    try:
        response = requests.post(API_URL, headers=HEADERS, json={
            "model": "meta-llama/Llama-3.2-3B-Instruct:together",
            "messages": [{"role": "user", "content": prompt}]
        })
        if response.status_code != 200:
            return f"❌ API Error {response.status_code}: {response.text}"
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ API call failed: {e}"

def call_local_summary(prompt: str) -> str:
    try:
        pipe = load_local_model()
        output = pipe(prompt, max_new_tokens=300, temperature=0.7)[0]['generated_text']
        return output.strip().split("Improved Summary:")[-1].strip()
    except Exception as e:
        return f"❌ Local generation failed: {e}"

def regenerate_summary_from_feedback(df: pd.DataFrame, original: str, feedback: str, model_mode="cloud") -> str:
    prompt = f"""
You are a business analyst AI. You previously generated the following insight summary:

--- Original Summary ---
{original}
------------------------

Now the user gave you this feedback:
"{feedback}"

Using this feedback and the dataset below, regenerate an improved summary.

{summarize_schema(df)}

Improved Summary:
"""
    if model_mode == "local":
        return call_local_summary(prompt)
    else:
        return call_cloud_summary(prompt)
