# agents/alert_summarizer.py

from agents.llm_utils import call_llm_model

def generate_alert_summary(alert_text: str, model_mode: str = "cloud") -> str:
    """
    Converts detected alert signals into a natural language summary using LLM.
    """
    if not alert_text.strip():
        return "No major alerts found."

    prompt = f"""
You are a business analyst reviewing a dataset. Based on the following detected alerts or signals, write a concise, professional summary of the **top data quality or modeling issues** that could impact decision-making.

Your summary should:
- Focus only on meaningful issues
- Avoid repeating obvious metadata (e.g., column types)
- Use a formal, concise tone
- Include potential impact and next steps if relevant

Here are the raw alerts:
{alert_text}

Write your summary below:
"""
    return call_llm_model(prompt, model_name="local" if model_mode == "local" else None)
