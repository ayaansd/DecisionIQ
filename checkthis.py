import os
import sys
import json
import time
import pandas as pd
import streamlit as st

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- MODULE IMPORTS ---
from eda.perform_eda import perform_eda
from kpi.extract_kpis import extract_kpis
from chart.generate_charts import smart_chart_agent
from agents.summarize_insights import generate_summary_from_df
from agents.feedback_regeneration import regenerate_summary_from_feedback
from agents.qa_agent import run_rag_qa_agent
from agents.goal_agent import run_goal_pipeline
from agents.agent_loop import insightgpt_agent_loop
from agents.proactive_agent import detect_proactive_signals
from agents.memory_logger import log_feedback
from exports.exports_utils import export_summary_to_pdf
from exports.exports_html import export_summary_to_html
from agents.alert_summarizer import generate_alert_summary
from agents.slack_utils import send_summary_to_slack
from agents.monitor.monitor_agent import detect_new_file
from logic.auto_pipeline import run_auto_pipeline
from agents.memory_logger import load_recent_sessions   


# --- PAGE CONFIG ---
st.set_page_config(page_title="InsightGPT | AI Business Analyst", layout="wide")

# --- HEADER ---
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 10px;">
  <div>
    <h1 style="margin-bottom: 0;">📊 InsightGPT</h1>
    <h4 style="margin-top: 0; color: gray;">Your Local AI Business Analyst</h4>
  </div>
  <div style="text-align: right;">
    <span style="color: #888;">Version 2.0 · Aug 2025</span>
  </div>
</div>
<hr style="margin-top: 0;">
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("⚙️ Settings")

model_mode = st.sidebar.radio(
    "Choose Model Mode:",
    ["🌐 Cloud Mode (Hugging Face)", "💻 Local Mode (Phi-2)"],
    index=0
)
model_mode_clean = "local" if model_mode.startswith("💻") else "cloud"

if model_mode_clean == "local":
    st.sidebar.warning("Local Mode runs entirely offline.\n\n- Requires 8GB+ RAM\n- May be slower.")
    use_local_confirmed = st.sidebar.checkbox("✅ I understand and want to proceed")
    if not use_local_confirmed:
        st.stop()
else:
    st.sidebar.info("Cloud Mode uses Hugging Face's LLaMA 3.2 and TinyLlama.")
    use_local_confirmed = True


# Memory Recall
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Memory Recall")
memory_path = "memory/session_memory.json"
if os.path.exists(memory_path):
    with open(memory_path, "r", encoding="utf-8") as f:
        memory_data = json.load(f)
    past_goals = [s["goal"] for s in memory_data]
    last_stage = memory_data[-1]["last_stage"] if memory_data else ""
    selected_past_goal = st.sidebar.selectbox("📌 Past Goals", ["None"] + past_goals[::-1])
    st.sidebar.markdown(f"✅ Last Completed Stage: `{last_stage}`")
    if selected_past_goal != "None":
        st.session_state["goal_input"] = selected_past_goal
        st.sidebar.success("💡 Goal prefilled from memory!")
else:
    st.sidebar.info("No past session memory found.")

# Slack Webhook
st.sidebar.markdown("---")
slack_webhook = st.sidebar.text_input(
    "🔗 Slack Webhook URL",
    type="password",
    help="Paste your Slack webhook URL to enable sharing summaries."
)

# Auto Analysis
enable_auto = st.sidebar.checkbox("🔁 Enable Auto Analysis (Watch /input_data)")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload your dataset", type=["csv", "json", "xlsx", "xls"])
df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded `{uploaded_file.name}`")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

elif enable_auto:
    st.markdown("⏳ Watching `/input_data` for new files...")
    max_wait_time = 300
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        new_file_path = detect_new_file()
        if new_file_path:
            st.success(f"📥 Detected new file: `{os.path.basename(new_file_path)}`")
            try:
                if new_file_path.endswith(".csv"):
                    df = pd.read_csv(new_file_path)
                elif new_file_path.endswith(".json"):
                    df = pd.read_json(new_file_path)
                else:
                    df = pd.read_excel(new_file_path)
            except Exception as e:
                st.error(f"❌ Failed to load file: {e}")
                break

            with st.spinner("🚀 Running full analysis..."):
                results = run_auto_pipeline(df, model_mode_clean)

            st.session_state["latest_summary"] = results["summary"]
            st.session_state["latest_kpis"] = results["kpis"]
            st.session_state["latest_alerts"] = results["alerts"]
            st.success("✅ Auto analysis complete!")
            break

        time.sleep(5)
    else:
        st.info("⏹️ No new files detected in 5 minutes.")

# --- MAIN TABS ---
if df is not None and use_local_confirmed:
    st.markdown("---")
    tabs = st.tabs(["🧪 EDA", "📌 KPIs", "📊 Charts", "🧠 Summary", "💬 Q&A", "🎯 Goal Agent", "📄 Export"])

    with tabs[0]:
        st.subheader("🔍 Exploratory Data Analysis")
        st.button("Run EDA", on_click=lambda: perform_eda(df))

    with tabs[1]:
        st.subheader("📌 KPI Extraction")
        if "latest_kpis" in st.session_state:
            st.json(st.session_state["latest_kpis"])
        elif st.button("Extract KPIs"):
            kpis = extract_kpis(df)
            st.json(kpis)

    with tabs[2]:
        st.subheader("📊 Smart Chart Generator")
        if st.button("Generate Charts"):
            smart_chart_agent(df)

    with tabs[3]:
        st.subheader("🧠 Insight Summary")

        if st.button("Generate Insight Summary"):
            with st.spinner("Generating insights..."):
                summary = generate_summary_from_df(df, domain="auto", model_mode=model_mode_clean)
                st.session_state["latest_summary"] = summary
                st.text_area("📋 AI Summary", summary, height=300)

        if "latest_summary" in st.session_state:
            st.markdown("---")
            st.text_area("📋 AI Summary", st.session_state["latest_summary"], height=300, key="summary_display")

            if slack_webhook:
                if st.button("📤 Send Summary to Slack"):
                    with st.spinner("Sending summary to Slack..."):
                        slack_response = send_summary_to_slack(
                            st.session_state["latest_summary"], webhook_url=slack_webhook
                        )
                    st.success(slack_response)

            with st.expander("✍️ Feedback-Based Summary Improvement"):
                st.markdown("Give feedback on the AI-generated summary, and we'll regenerate a better version using your selected model (cloud/local).")
                feedback_text = st.text_area("🔁 What would you like to improve or change?", key="summary_feedback_text")

                if feedback_text and st.button("Regenerate Summary", key="regenerate_button"):
                    with st.spinner("🧠 Improving the summary based on your feedback..."):
                        improved = regenerate_summary_from_feedback(
                            df, st.session_state["latest_summary"], feedback_text, model_mode_clean
                        )
                    st.markdown("### ✅ Improved Summary")
                    st.success(improved)

        if "improved" in locals():
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/regenerated_summary.txt", "w", encoding="utf-8") as f:
                f.write(improved)
    with tabs[4]:
        st.subheader("💬 RAG-Powered Q&A")
        question = st.text_input("Ask a question:")
        if question:
            with st.spinner("Thinking..."):
                output = run_rag_qa_agent(df, question, model_mode=model_mode_clean)
                result = output.get("result", "")
                answer = output.get("answer", "")
                explanation = output.get("explanation", "")
                code = output.get("code", "")

            st.markdown(f"**📍 Question:** {question}")

            if result is not None and not isinstance(result, pd.DataFrame):
                st.success(f"📊 Result: {result}")
            elif isinstance(result, pd.DataFrame):
                st.dataframe(result)
            else:
                st.info("No result returned.")


            with st.expander("🗣️ Answer"):
                st.markdown(answer or "_No answer provided._")

            with st.expander("📘 Explanation"):
                st.markdown(explanation or "_No explanation provided._")

            with st.expander("💻 Code Used"):
                st.code(code or "# No code returned", language="python")


    with tabs[5]:
        st.subheader("🎯 Goal-Oriented Analysis")
        goal_input = st.text_input("Enter your business goal", value=st.session_state.get("goal_input", ""))
        if goal_input and st.button("Run Goal Agent"):
            with st.spinner("Analyzing..."):
                report = run_goal_pipeline(df, goal_input)
                st.text_area("📄 Goal-Based Report", report, height=300)
                st.success("✅ Goal analysis complete")

    with tabs[6]:
        st.subheader("📄 Export Insight Report")
        if st.button("Download PDF"):
            try:
                with open("outputs/insight_summary.txt", "r", encoding="utf-8") as f:
                    summary = f.read()
                pdf_path = export_summary_to_pdf(summary)
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download PDF", f, file_name=os.path.basename(pdf_path))
            except Exception as e:
                st.error(f"❌ PDF export failed: {e}")
        if st.button("Download HTML Report"):
            try:
                with open("outputs/insight_summary.txt", "r", encoding="utf-8") as f:
                    summary = f.read()
                html_path = export_summary_to_html(summary, chart_dir="charts_output")
                with open(html_path, "r", encoding="utf-8") as f:
                    st.download_button("🌐 Download HTML", f, file_name="insight_report.html")
            except Exception as e:
                st.error(f"❌ HTML export failed: {e}")

# --- PROACTIVE ALERTS ---
st.markdown("---")
st.subheader("🚨 Proactive Alerts")

if df is not None:
    proactive_signals = detect_proactive_signals(df)
    with st.spinner("🧠 Summarizing key issues..."):
        alert_summary = generate_alert_summary(proactive_signals, model_mode=model_mode_clean)

    if "latest_alerts" in st.session_state:
        st.markdown("#### 🧠 AI-Generated Summary")
        st.success(st.session_state["latest_alerts"])
    else:
        st.markdown("#### 🧠 AI-Generated Summary")
        st.success(alert_summary)

    with st.expander("🛠️ Raw Alerts Detected"):
        if proactive_signals.strip():
            keywords = ["Q1", "Q2", "Q3", "Q4"] + [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            for line in proactive_signals.strip().split("\n"):
                for word in keywords:
                    if word in line:
                        line = line.replace(word, f"🟡 **{word}**")
                st.markdown(f"- {line}")
        else:
            st.info("✅ No anomalies or alerts detected.")


    with st.expander("🧠 Recent InsightGPT Sessions", expanded=False):
        recent_sessions = load_recent_sessions(limit=5)
        if not recent_sessions:
            st.info("No previous sessions found.")
        else:
            for session in reversed(recent_sessions):
                st.markdown(f"#### 📅 {session['timestamp'][:10]}")
                st.markdown(f"**Goal:** {session.get('goal', 'N/A')}")
                st.markdown(f"**Dataset Shape:** `{session.get('df_shape', 'unknown')}`")
                if session.get("feedback_log"):
                    st.markdown(f"**Feedback:** {' | '.join(session['feedback_log'])}")
                if session.get("completed_stages"):
                    st.markdown(f"**Completed Stages:** {', '.join(session['completed_stages'])}")
                st.markdown("---")

# --- FOOTER ---
st.markdown("""
---
<div style='text-align: center; color: gray; font-size: 0.85rem;'>
Made with ❤️ by InsightGPT · © 2025
</div>
""", unsafe_allow_html=True)
