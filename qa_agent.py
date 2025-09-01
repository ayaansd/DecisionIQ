import pandas as pd
import traceback
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import io
import matplotlib.pyplot as plt

# 🔐 Hugging Face API setup
HF_TOKEN = "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"
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# 🧠 Local model cache
_local_pipeline = None

def get_local_llm():
    global _local_pipeline
    if _local_pipeline is None:
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        _local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return _local_pipeline

# ---------------- Utility Fixes ---------------- #
def format_result_for_answer(result):
    """Format numbers cleanly before sending to LLM."""
    if isinstance(result, pd.Series):
        return {str(k): f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)
                for k, v in result.items()}
    elif isinstance(result, pd.DataFrame):
        formatted_df = result.copy()
        for col in formatted_df.select_dtypes(include=['float', 'int']).columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.2f}")
        return formatted_df.to_dict(orient="records")
    return str(result)

def strip_print_statements(code):
    """Remove print() lines so explanation is clean."""
    return "\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith("print(")
    )
# ---------------- Prompt Generators ---------------- #
def summarize_schema(df: pd.DataFrame) -> str:
    schema = "\n".join([f"- {col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
    sample = df.sample(n=min(2, len(df))).to_string(index=False)
    return f"Dataset schema:\n{schema}\n\nSample rows:\n{sample}"


def generate_code_prompt(df, question):
    return f"""
You are a data assistant. Generate Python code to answer the user's question using the DataFrame `df`.

- Respond with ONLY the code.
- Do NOT explain.
- Assign the result to a variable called `result`.

{summarize_schema(df)}

Question: {question}
Code:
"""


def generate_answer_prompt(question, result):
    return f"""
You are a helpful analyst. The user asked: "{question}"

Their question was answered with the following result: {result}

Now write a clear, one-line answer in plain English.
"""


def generate_explanation_prompt(question, code, answer):
    return f"""
Explain how the following code answers the user's question step-by-step.

Question: {question}
Code:
{code}
Answer:
{answer}
"""


def generate_cot_prompt(question, code, answer):
    return f"""
You are an expert data analyst.

Break down your reasoning step by step to explain how the code answers the user's question.

Use bullet points and clear, concise logic.

Include:
- What the question is asking
- What the code does
- How the logic connects to the answer
- Any assumptions made

Format your reasoning like:
• Step 1: ...
• Step 2: ...
• Step 3: ...

Question: {question}
Code:
{code}
Answer: {answer}
"""


def extract_code_block(raw):
    if "```python" in raw:
        return raw.split("```python")[1].split("```")[0].strip()
    elif "```" in raw:
        return raw.split("```")[1].strip()
    return raw.strip()

# ---------------- Execution ---------------- #

def execute_generated_code(df, code):
    # Only block truly unsafe patterns
    banned_exact = ['os.', 'sys.', 'subprocess', 'eval(', 'exec(']
    # Keep open() allowed for libraries, but block only if it's used on raw strings/paths
    banned_open_pattern = "open("

    if any(bad in code for bad in banned_exact):
        return None, None, "🚫 Blocked: unsafe code (system access)."

    # Extra layer — block open() only if not related to matplotlib, pandas, or temp files
    if banned_open_pattern in code and not any(lib in code for lib in ["plt", "pd", "np", "tempfile"]):
        return None, None, "🚫 Blocked: unsafe file access."

    local_vars = {"df": df.copy(), "pd": pd, "plt": plt}
    try:
        plt.close('all')  # clear previous figures
        exec(code, {}, local_vars)

        result = local_vars.get("result", None)

        # Detect chart
        fig = None
        if plt.get_fignums():
            fig = plt.gcf()

        return result, fig, None
    except Exception as e:
        return None, None, f"❌ Execution error: {str(e)}\n{traceback.format_exc()}"

# ---------------- LLM Call ---------------- #

def call_llm(prompt, model_mode="cloud", max_new_tokens=200):
    if model_mode == "local":
        llm = get_local_llm()
        return llm(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"].strip()
    else:
        response = requests.post(API_URL, headers=HEADERS, json={
            "model": "meta-llama/Llama-3.2-3B-Instruct:together",
            "messages": [{"role": "user", "content": prompt}]
        })
        return response.json()["choices"][0]["message"]["content"].strip()

# ---------------- Main Agent ---------------- #
def run_rag_qa_agent(df, question, model_mode="cloud"):
    # Step 1: Generate code
    code_prompt = generate_code_prompt(df, question)
    code_raw = call_llm(code_prompt, model_mode)
    code = extract_code_block(code_raw)
    code = "\n".join(
    line for line in code.splitlines()
    if line.strip() and "undefined" not in line.lower()
    )

    # Step 2: Execute code
    result, fig, error = execute_generated_code(df, code)
    if error:
        return {
            "answer": error,
            "code": code,
            "result": pd.DataFrame({"error": [error]}),
            "chart_image": None,
            "explanation": "Code execution failed.",
            "cot_reasoning": "N/A"
        }

    # Step 3: Generate concise answer
    clean_result = format_result_for_answer(result)
    answer_prompt = generate_answer_prompt(question, clean_result)

    answer = call_llm(answer_prompt, model_mode)

    # Step 4: Detailed explanation
    clean_code = strip_print_statements(code)
    explanation_prompt = generate_explanation_prompt(question, clean_code, answer)
    explanation = call_llm(explanation_prompt, model_mode, max_new_tokens=300)

    # Step 5: Chain-of-Thought reasoning
    cot_prompt = generate_cot_prompt(question, clean_code, answer)
    cot_reasoning = call_llm(cot_prompt, model_mode, max_new_tokens=300)

    return {
        "answer": answer,
        "explanation": explanation,
        "cot_reasoning": cot_reasoning,
        "code": code,
        "result": result if isinstance(result, pd.DataFrame) else pd.DataFrame({"value": [result]}),
        "chart_image": fig
    }
