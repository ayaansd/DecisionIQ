import pandas as pd

def summarize_schema(df: pd.DataFrame) -> str:
    summary = []
    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].dropna().unique()[:3]
        summary.append(f"{col} ({dtype}): {sample}")
    return "\n".join(summary)


def generate_prompt_from_schema(schema: str, question: str) -> str:
    return f"""You are a Python data analyst. Write Python code to answer the question using a DataFrame `df`.

DataFrame Schema:
{schema}

Question: {question}

Instructions:
- Use pandas only.
- Assign the answer to a variable called `result`.
- Do NOT print anything.
- Do not use any external libraries.

Code:"""


def execute_answer_code(code: str, df: pd.DataFrame):
    local_vars = {"df": df.copy()}
    try:
        exec(code, {}, local_vars)
        return local_vars.get("result", "⚠️ Code ran, but no result was returned.")
    except Exception as e:
        return f"❌ Error while executing code: {e}"


def infer_schema(df: pd.DataFrame, sample_rows: int = 3) -> str:
    schema_lines = []
    for col in df.columns:
        dtype = df[col].dtype
        sample_vals = df[col].dropna().unique()[:sample_rows]
        sample_str = ", ".join(map(str, sample_vals))
        schema_lines.append(f"- `{col}` ({dtype}): e.g., {sample_str}")
    schema = "\n".join(schema_lines)
    return f"📊 The dataset has the following columns:\n{schema}"
