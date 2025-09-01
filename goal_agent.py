import os
from eda.perform_eda import perform_eda
from kpi.extract_kpis import extract_kpis
from chart.generate_charts import smart_chart_agent
from agents.summarize_insights import generate_summary_from_df
from agents.qa_agent import run_rag_qa_agent
from agents.memory_logger import log_session


def run_goal_pipeline(df, goal: str, output_dir="outputs", user_feedback: str = None) -> str:
    steps = parse_goal(goal)
    results = [f"🎯 **Goal-Based Analysis Report**", f"📝 **Goal:** {goal}"]
    completed = []
    feedback = []

    print(f"🎯 Goal: {goal}")
    print(f"🧭 Planned steps: {steps}")

    for step in steps:
        try:
            if step == "eda":
                perform_eda(df)
                results.append("🧪 **EDA Completed**")
                completed.append("eda")

            elif step == "kpi":
                kpis = extract_kpis(df)
                results.append("📊 **KPI Summary:**")
                for k, v in kpis.items():
                    if isinstance(v, dict):
                        line = f"- {k}: mean = {v.get('mean', 'N/A'):.2f}, max = {v.get('max', 'N/A'):.2f}"
                    else:
                        line = f"- {k}: {v}"
                    results.append(line)
                completed.append("kpi")

            elif step == "charts":
                chart_paths = smart_chart_agent(df, output_dir=output_dir)
                results.append(f"📈 **Charts generated and saved to `{output_dir}`**")
                completed.append("charts")

            elif step == "summary":
                summary = generate_summary_from_df(df)
                results.append("📝 **Insight Summary:**\n" + summary)
                completed.append("summary")

            elif step == "qa":
                answer = run_rag_qa_agent(df, question=goal)
                results.append("💬 **Q&A Answer:**")
                results.append(f"- **Answer:** {answer.get('answer', '')}")
                results.append(f"- **Code:**\n```python\n{answer.get('code', '')}\n```")
                results.append("📊 **Result Preview:**")
                results.append(str(answer.get("result", "")))
                completed.append("qa")

        except Exception as e:
            feedback.append(f"⚠️ {step} failed: {e}")

    # Log session memory
    feedback_log = feedback.copy()
    if user_feedback:
        feedback_log.append(f"User said: {user_feedback}")

    log_session({
        "goal": goal,
        "completed_stages": completed,
        "feedback_log": feedback_log
    }, df_shape=df.shape)

    final_report = "\n\n".join(results)
    final_path = os.path.join(output_dir, "final_goal_report.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(final_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"✅ Goal pipeline completed. Report saved to: {final_path}")
    return final_report

def parse_goal(goal: str) -> list:
    """
    Parses a natural language goal and returns a list of analysis steps.
    Example: "Find top products and trends" → ['eda', 'kpi', 'charts', 'summary']
    """
    goal = goal.lower()
    steps = []

    if any(word in goal for word in ["trend", "pattern", "insight", "distribution", "correlation"]):
        steps.append("eda")

    if any(word in goal for word in ["visual", "chart", "graph", "plot", "top", "compare", "versus", "vs", "distribution", "region", "category"]):
        steps.append("charts")

    if any(word in goal for word in ["summary", "report", "insight", "recommend", "action", "overview"]):
        steps.append("summary")

    if any(word in goal for word in ["question", "how", "why", "what", "which", "does", "is", "can"]):
        steps.append("qa")

    if any(word in goal for word in ["top", "metric", "kpi", "measure", "growth", "performance", "revenue", "sales"]):
        steps.append("kpi")

    if "eda" not in steps:
        steps.insert(0, "eda")
    if "summary" not in steps:
        steps.append("summary")

    return list(dict.fromkeys(steps))  # remove duplicates, preserve order
