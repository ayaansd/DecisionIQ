import os
import sys
import json
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from streamlit.components.v1 import html
import threading

# PATH SETUP
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# MODULE IMPORTS
from eda.perform_eda import perform_eda
from kpi.extract_kpis import extract_kpis
from chart.generate_charts import smart_chart_agent
from agents.summarize_insights import generate_summary_from_df
from agents.feedback_regeneration import regenerate_summary_from_feedback
from agents.qa_agent import run_rag_qa_agent
from agents.goal_agent import run_goal_pipeline
from agents.agent_loop import insightgpt_agent_loop
from agents.proactive_agent import detect_proactive_signals
from agents.memory_logger import log_feedback, load_recent_sessions
from exports.exports_utils import export_summary_to_pdf
from exports.exports_html import export_summary_to_html
from agents.alert_summarizer import generate_alert_summary
from agents.slack_utils import send_summary_to_slack
from logic.auto_pipeline import run_auto_pipeline
from monitor.continuous_monitor import watch_for_new_files
from agents.build_faiss_index import build_faiss_index
from agents.rag_faiss_agent import run_faiss_rag_agent

# PAGE CONFIG
st.set_page_config(page_title="InsightGPT", layout="wide")

if not os.path.exists("input_data"):
    os.makedirs("input_data")
if not os.path.exists("outputs"):
    os.makedirs("outputs")
 
# --- Smart Q&A Routing Logic ---
def smart_route_question(df, question, model_mode="cloud"):
    natural_keywords = ["who", "when", "which", "show", "list", "customer", "order", "message", "chat", "complain", "comment", "email", "review"]
    question_lower = question.lower()

    if any(kw in question_lower for kw in natural_keywords):
        return {
            "mode": "faiss",
            "answer": run_faiss_rag_agent(question, model_mode=model_mode)
        }
    else:
        return {
            "mode": "codegen",
            "output": run_rag_qa_agent(df, question, model_mode=model_mode)
        }

# --- SIDEBAR ---
st.sidebar.title("⚙️ Settings")

model_mode = st.sidebar.radio(
    "Choose Model Mode:",
    ["🌐 Cloud Mode (Hugging Face)", "💻 Local Mode (Phi-2)"],
    index=0
)
model_mode_clean = "local" if model_mode.startswith("💻") else "cloud"

if model_mode_clean == "local":
    st.sidebar.warning("Local Mode runs offline. Requires 8GB+ RAM.")
    if not st.sidebar.checkbox("✅ I understand and want to proceed"):
        st.stop()
else:
    st.sidebar.info("Cloud Mode uses Hugging Face's LLaMA 3.2 and TinyLlama.")

# --- Memory Recall ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Memory Recall")
memory_path = "memory/session_memory.json"
if os.path.exists(memory_path):
    with open(memory_path, "r", encoding="utf-8") as f:
        memory_data = json.load(f)
    past_goals = [s["goal"] for s in memory_data]
    last_stage = memory_data[-1]["last_stage"] if memory_data else ""
    selected_past_goal = st.sidebar.selectbox("📌 Past Goals", ["None"] + past_goals[::-1])
    st.sidebar.markdown(f"✅ Last Completed Stage: {last_stage}")
    if selected_past_goal != "None":
        st.session_state["goal_input"] = selected_past_goal
        st.sidebar.success("💡 Goal prefilled from memory!")
else:
    st.sidebar.info("No past session memory found.")

# --- Slack & Auto Trigger ---
st.sidebar.markdown("---")
slack_webhook = st.sidebar.text_input("🔗 Slack Webhook URL", type="password")
enable_auto = st.sidebar.checkbox("🔁 Enable Auto Analysis (Watch /input_data)")

if st.session_state.get("enable_auto", False):
    if "monitor_thread" not in st.session_state:
        st.success("🔁 Auto Analysis is ON — Watching /input_data folder")

        def start_monitor():
            watch_for_new_files(model_mode=model_mode_clean)

        thread = threading.Thread(target=start_monitor, daemon=True)
        thread.start()
        st.session_state["monitor_thread"] = thread

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
        st.success(f"✅ Loaded {uploaded_file.name}")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

# --- MAIN TABS ---
if df is not None:
    build_faiss_index(df)
    tabs = st.tabs(["🧪 EDA", "📌 KPIs", "📊 Charts", "🧠 Summary", "💬 Q&A", "🎯 Goal Agent", "📄 Export"])

    # EDA Tab
    with tabs[0]:
        st.subheader("🔍 Exploratory Data Analysis")
        if st.button("Run EDA"):
            perform_eda(df)

    # KPIs Tab
    with tabs[1]:
        st.subheader("📌 KPI Extraction")
        if "latest_kpis" in st.session_state:
            st.json(st.session_state["latest_kpis"])
        elif st.button("Extract KPIs"):
            kpis = extract_kpis(df)
            st.session_state["latest_kpis"] = kpis
            st.success("✅ KPIs extracted!")
            st.json(kpis)

    # Charts Tab
    with tabs[2]:
        st.subheader("📊 Smart Chart Generator")
        if st.button("Generate Charts"):
            chart_paths, chart_summaries = smart_chart_agent(df)  # must return both
            if chart_paths:
                st.session_state["chart_paths"] = chart_paths
                st.session_state["chart_summaries"] = chart_summaries

                st.success("✅ Charts generated!")
                for path, summary in zip(chart_paths[:3], chart_summaries[:3]):
                    st.image(path, use_column_width=True)
                    st.markdown(f"**Insight:** {summary}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"outputs/charts_{timestamp}.txt", "w") as f:
                    f.write("\n".join(chart_paths))
                st.toast("📊 Charts saved!", icon="✅")


    # Insight Summary Tab
    with tabs[3]:
        st.subheader("🧠 Insight Summary")

        if st.button("Generate Insight Summary"):
            with st.spinner("Generating insights..."):
                summary = generate_summary_from_df(df, domain="auto", model_mode=model_mode_clean)
                st.session_state["latest_summary"] = summary
                st.text_area("📋 AI Summary", summary, height=300)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("outputs", exist_ok=True)
                with open(f"outputs/insight_summary_{timestamp}.txt", "w", encoding="utf-8") as f:
                    f.write(summary)
                st.toast("🧠 Summary saved!", icon="✅")

        if "latest_summary" in st.session_state:
        # ✅ Show stored charts + insights if available
            if "chart_paths" in st.session_state and "chart_summaries" in st.session_state:
                st.markdown("### 📊 Key Charts")
                for path, summary in zip(st.session_state["chart_paths"][:2], st.session_state["chart_summaries"][:2]):
                    st.image(path, use_container_width=True, width=400)  # Compact preview
                    st.markdown(f"**Insight:** {summary}")
            st.markdown("---")
        
            summary = st.session_state["latest_summary"]
            summary_lines = summary.strip().split("\n")
            preview = "\n".join(summary_lines[:2]) if len(summary_lines) >= 2 else summary
            st.success("🧠 Insight summary generated.")
            st.markdown("#### 📝 Summary Preview")
            st.code(preview, language="markdown")
            with st.expander("📄 Click to view full AI Summary"):
                st.text_area("📋 Full Summary", summary, height=300, key="summary_display", disabled=True)

        # Feedback & Regeneration
            with st.expander("✍️ Feedback-Based Summary Improvement"):
                st.markdown("Give feedback on the AI-generated summary below.")
                feedback_text = st.text_area("🔁 What would you like to improve?", key="summary_feedback_text")
                if feedback_text and st.button("Regenerate Summary", key="regenerate_button"):
                    with st.spinner("Improving the summary..."):
                        improved = regenerate_summary_from_feedback(
                            df, st.session_state["latest_summary"], feedback_text, model_mode_clean
                        )
                        st.success("✅ Summary improved!")
                        st.text_area("📄 Improved Summary", improved, height=300)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        with open(f"outputs/regenerated_summary_{timestamp}.txt", "w", encoding="utf-8") as f:
                            f.write(improved)
                        st.toast("✍️ Improved Summary saved!", icon="✅")

    # Q&A Tab  ✅ FIXED POSITION
    with tabs[4]:
        st.subheader("💬 RAG-Powered Q&A")
        question = st.text_input("Ask a business question:")

        if question:
            with st.spinner("Thinking..."):
                response = smart_route_question(df, question, model_mode_clean)

            st.markdown(f"**📍 Question:** {question}")

            if response["mode"] == "faiss":
                st.success(response["answer"])
            else:
                output = response["output"]
                result = output.get("result", "")
                answer = output.get("answer", "")
                explanation = output.get("explanation", "")
                code = output.get("code", "")
                chart = output.get("chart_image", None)

            # 🔹 Show error first if present
                if isinstance(answer, str) and ("error" in answer.lower() or "🚫" in answer):
                    st.error(answer)
                else:
                # Show DataFrame result if available
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        st.dataframe(result, height=250, width=600)
                    elif result:
                        st.success(f"📊 Result: {result}")

                # Show chart if available
                    if chart is not None:
                        st.pyplot(chart, clear_figure=False)

                # Plain answer
                    with st.expander("🗣️ Answer", expanded=True):
                        st.markdown(answer or "_No answer provided._")

                # Step-by-step reasoning
                    with st.expander("🔍 Chain-of-Thought Explanation"):
                        if explanation:
                            formatted_explanation = "\n".join(
                                [
                                    f"- **{line.strip().split(':')[0]}:** {':'.join(line.strip().split(':')[1:])}"
                                    if ":" in line else f"- {line.strip()}"
                                    for line in explanation.strip().split("\n") if line.strip()
                                ]
                            )
                            st.markdown(formatted_explanation)

                # Generated code
                    with st.expander("💻 Code Used"):
                        st.code(code or "# No code provided", language="python")

    # Goal Agent Tab
    with tabs[5]:
        st.subheader("🎯 Goal-Oriented Report")
        goal_input = st.text_input("Enter your business goal:", value=st.session_state.get("goal_input", ""))
        if goal_input and st.button("Run Goal Agent"):
            with st.spinner("Generating goal-driven analysis..."):
                report = run_goal_pipeline(df, goal_input)
                st.text_area("📄 Goal-Based Report", report, height=300)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"outputs/goal_report_{timestamp}.txt", "w", encoding="utf-8") as f:
                    f.write(report)
                st.toast("🎯 Goal Report saved!", icon="✅")

    # Export Tab
    with tabs[6]:
        st.subheader("📄 Export Insight Report")
        if st.button("📥 Download PDF"):
            try:
                with open("outputs/insight_summary.txt", "r", encoding="utf-8") as f:
                    summary = f.read()
                pdf_path = export_summary_to_pdf(summary)
                with open(pdf_path, "rb") as f:
                    st.download_button("📩 Download PDF", f, file_name=os.path.basename(pdf_path))
                st.toast("📄 PDF exported!", icon="✅")
            except Exception as e:
                st.error(f"❌ PDF export failed: {e}")

        if st.button("🌐 Download HTML Report"):
            try:
                with open("outputs/insight_summary.txt", "r", encoding="utf-8") as f:
                    summary = f.read()
                html_path = export_summary_to_html(summary, chart_dir="charts_output")
                with open(html_path, "r", encoding="utf-8") as f:
                    st.download_button("🌍 Download HTML", f, file_name="insight_report.html")
                st.toast("🌐 HTML exported!", icon="✅")
            except Exception as e:
                st.error(f"❌ HTML export failed: {e}")
     
# --- Proactive Alerts ---
if df is not None:
    st.markdown("---")
    st.subheader("🚨 Proactive Alerts")

    proactive_signals = detect_proactive_signals(df)
    with st.spinner("🧠 Summarizing key issues..."):
        alert_summary = generate_alert_summary(proactive_signals, model_mode=model_mode_clean)
        st.session_state["latest_alerts"] = alert_summary

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
            st.info("✅ No anomalies detected.")

    with st.expander("🧠 Recent InsightGPT Sessions", expanded=False):
        recent_sessions = load_recent_sessions(limit=5)
        if not recent_sessions:
            st.info("No previous sessions found.")
        else:
            for session in reversed(recent_sessions):
                st.markdown(f"#### 📅 {session['timestamp'][:10]}")
                st.markdown(f"**Goal:** {session.get('goal', 'N/A')}")
                st.markdown(f"**Dataset Shape:** {session.get('df_shape', 'unknown')}")
                if session.get("feedback_log"):
                    st.markdown(f"**Feedback:** {' | '.join(session['feedback_log'])}")
                if session.get("completed_stages"):
                    st.markdown(f"**Completed Stages:** {', '.join(session['completed_stages'])}")
                st.markdown("---")

if enable_auto:
    st.info("🔁 Auto-analysis is running. Watching /input_data for new files...")
    run_auto_pipeline()

st.markdown("""
---
<div style='text-align: center; color: gray; font-size: 0.85rem; padding-top: 20px;'>
InsightGPT — The Only Analyst Who Works Offline and Never Sleeps · © 2025
</div>
""", unsafe_allow_html=True)