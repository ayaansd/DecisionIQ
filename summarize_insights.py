import os
import pandas as pd
from datetime import datetime
from colorama import Fore, Style, init
from agents.memory_logger import load_recent_sessions
from agents.proactive_agent import detect_proactive_signals
from agents.llm_utils import call_llm_model

init(autoreset=True)

# ---------------------- Helper: Stats Formatter ----------------------

def format_stats_for_llm(df: pd.DataFrame) -> str:
    lines = [f"Total Rows: {len(df)}", "\nColumn Types:"]
    for col in df.columns:
        lines.append(f" - {col}: {df[col].dtype}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        lines.append("\nMissing Values:")
        for col, count in missing.items():
            lines.append(f"  - {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        lines.append("\nNo Missing Values")

    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        summary_df = numeric_df.describe().loc[["mean", "std", "min", "max"]].round(2)
        lines.append("\nKey Stats:\n")
        for col in summary_df.columns:
            lines.append(f"• {col}: Mean={summary_df[col]['mean']}, Std={summary_df[col]['std']}, "
                         f"Min={summary_df[col]['min']}, Max={summary_df[col]['max']}")
    else:
        lines.append("\n⚠️ No numeric columns found.")

    categorical_cols = df.select_dtypes(include=['object', 'category'])
    if not categorical_cols.empty:
        lines.append("\nTop Categories:")
        for col in categorical_cols.columns[:3]:
            top_vals = df[col].value_counts().head(5)
            lines.append(f"  {col}")
            for val, cnt in top_vals.items():
                lines.append(f"   {val}: {cnt}")
    return "\n".join(lines)

# ---------------------- Main Summary Generator ----------------------

def generate_summary_from_df(df: pd.DataFrame, domain: str = "auto", output_dir="outputs", model_mode="local") -> str:
    if df.empty:
        return "⚠️ DataFrame is empty."

    def infer_domain(df):
        domain_keywords = {
            "finance": ["revenue", "profit", "cost", "margin"],
            "marketing": ["campaign", "click", "conversion", "impression"],
            "healthcare": ["patient", "diagnosis", "treatment", "medication"],
            "retail": ["product", "sales", "inventory", "price"],
            "hr": ["employee", "attrition", "satisfaction", "department"]
        }
        col_names = [c.lower() for c in df.columns]
        scores = {d: sum(any(kw in c for c in col_names) for kw in kws) for d, kws in domain_keywords.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    if domain == "auto":
        domain = infer_domain(df)
        print(Fore.MAGENTA + f"🧠 Auto-detected domain: {domain}" + Style.RESET_ALL)

    role_map = {
        "finance": "You are a senior financial analyst at a Fortune 500 firm.",
        "marketing": "You are a senior marketing analyst specializing in ROI and performance campaigns.",
        "healthcare": "You are a healthcare data analyst helping improve clinical outcomes.",
        "retail": "You are a retail business analyst focused on inventory, sales, and demand trends.",
        "hr": "You are an HR data specialist tracking performance, satisfaction, and retention.",
        "general": "You are a business analyst generating insights for strategic decisions."
    }

    # ✅ Fixed: Corrected to limit=3
    recent_sessions = load_recent_sessions(limit=3)
    if recent_sessions:
        memory_snippet = "\n".join(
            [f"[{s.get('timestamp','?')}] Goal: {s.get('goal','N/A')} | "
             f"Stages: {', '.join(s.get('completed_stages', [])) or 'None'} | "
             f"Feedback: {'; '.join(s.get('feedback_log', [])) or 'None'}"
             for s in recent_sessions]
        )
    else:
        memory_snippet = "No prior sessions found."

    prompt = f"""
{role_map.get(domain, role_map['general'])}

You are preparing a business intelligence summary for executives.  
Write the output in exactly **4 sections** with the following structure and rules:

**1. Top 3 KPIs to Monitor**  
- Each KPI must have: Name, Value (with units), and why it matters ("So What?") in 1 sentence.  
- Avoid repeating the same KPI in Trends.  
- Thresholds must be in plain English, no formulas.

**2. Key Trends or Anomalies**  
- Describe 3–4 major patterns, changes, or unusual points in the data.  
- Use numbers sparingly and only when they add impact.  
- No KPI repetition.

**3. Reasoning Process**  
- Briefly describe how you identified the KPIs and trends (1–2 sentences).  
- Mention the analysis approach (e.g., correlations, comparisons, grouping).

**4. Recommended Actions**  
- 3–5 specific, actionable steps directly tied to KPIs/trends.  
- Include measurable targets where possible (e.g., "Increase electronics SKUs by 15%" instead of "expand electronics offerings").  
- Keep them in parallel, action-oriented format.

---

📁 Context from past sessions:
{memory_snippet}

📊 Dataset Summary:
{format_stats_for_llm(df)}

🚨 Proactive Alerts:
{detect_proactive_signals(df)}

Now, generate the output strictly following the above structure.
"""


    print(Fore.BLUE + "\nPrompt Sent to LLM:\n" + prompt[:600] + "..." + Style.RESET_ALL)

    output = call_llm_model(prompt, model_name=model_mode)

    final_output = output.split("summary:")[-1].strip() if "summary:" in output.lower() else output.strip()

    if len(final_output) < 100:
        print(Fore.YELLOW + "⚠️ Short output — retrying..." + Style.RESET_ALL)
        prompt += "\nPlease expand with more detailed KPIs, trends, and actions."
        output = call_llm_model(prompt, model_name=model_mode)
        final_output = output.split("summary:")[-1].strip()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = os.path.join(output_dir, f"insight_summary_{timestamp}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(Fore.GREEN + f"✅ Summary saved to: {out_path}\n" + Style.RESET_ALL)
    print(final_output)

    return final_output
