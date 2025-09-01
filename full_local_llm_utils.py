# -------------------------------------------------
#  agents/llm_utils.py   (offline‑only version)
# -------------------------------------------------
import os
import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# -----------------------------------------------------------------
# 1️⃣  OFFLINE / TELEMETRY SETTINGS  (set once, globally)
# -----------------------------------------------------------------
# These flags make 🤗‑Transformers read only from the local cache
# and suppress the tiny usage‑telemetry ping.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# -----------------------------------------------------------------
# 2️⃣  CONFIGURATION – pick the model you have stored locally
# -----------------------------------------------------------------
# Example paths – change them to the folder where you cached the model.
# You can use any of the 7‑B models listed in the table in the answer
# (Mistral‑7B, Falcon‑7B, Llama‑2‑7B‑Chat, Phi‑2, Gemma‑7B, …).
LOCAL_MODEL_DIR = Path(
    os.getenv(
        "LOCAL_MODEL_DIR",
        # default to a folder inside your home directory
        "~/local-llms/mistral-7b"
    )
).expanduser()

# -----------------------------------------------------------------
# 3️⃣  QUANTISATION SETTINGS (4‑bit is the sweet spot for 16 GB GPUs)
# -----------------------------------------------------------------
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,                     # switch to False for 8‑bit or full FP16
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",            # best quality for LLMs
)

# -----------------------------------------------------------------
# 4️⃣  LOAD TOKENIZER & MODEL (once, at import time)
# -----------------------------------------------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_DIR,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        device_map="auto",                 # puts layers on GPU if available, else CPU
        quantization_config=quant_cfg,
        trust_remote_code=False,
    )
    _LOCAL_MODEL_READY = True
    print("✅ Offline LLM loaded from:", LOCAL_MODEL_DIR)
except Exception as exc:                     # noqa: BLE001
    _LOCAL_MODEL_READY = False
    model = tokenizer = None
    print(f"⚠️ Could not load local model ({LOCAL_MODEL_DIR}): {exc}")

# -----------------------------------------------------------------
# 5️⃣  Helper that actually runs generation
# -----------------------------------------------------------------
def _generate_local(prompt: str, max_new: int = 200, temperature: float = 0.7) -> str:
    """
    Run the locally‑loaded model (4‑bit quantised) on *prompt*.
    Returns the generated text (no leading prompt echo).
    """
    if not _LOCAL_MODEL_READY:
        return "❌ Local model is not available."

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    # The pipeline will automatically move the tensors to the right device.
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

# -----------------------------------------------------------------
# 6️⃣  PUBLIC API – ONLY LOCAL, NO CLOUD
# -----------------------------------------------------------------
def call_llm_model(prompt: str, model_name: str = "local") -> str:
    """
    **Offline‑only** wrapper.

    Parameters
    ----------
    prompt : str
        The user message you want the LLM to answer.
    model_name : str, optional
        Only the literal string ``"local"`` is recognised.
        Any other value will raise an error – this function never makes
        an HTTP request.

    Returns
    -------
    str
        The model’s response (or an error message if the model could
        not be loaded).
    """
    if model_name != "local":
        raise ValueError(
            "Only the local model is available in this offline build. "
            "Pass model_name='local'."
        )
    return _generate_local(prompt)


# -----------------------------------------------------------------
# 7️⃣  Example usage (run this file directly to test)
# -----------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = (
        "Summarise the key insights from a quarterly sales table in two sentences."
    )
    print("\n--- Prompt ---------------------------------------------------")
    print(test_prompt)
    print("\n--- Model output --------------------------------------------")
    print(call_llm_model(test_prompt, model_name="local"))