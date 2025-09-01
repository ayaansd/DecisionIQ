# agents/agent_loop.py (Streamlit-compatible version)

import pandas as pd
import streamlit as st
from agents.agent_state import initialize_state, STAGE_TRANSITIONS
from eda.perform_eda import perform_eda
from kpi.extract_kpis import extract_kpis
from chart.generate_charts import smart_chart_agent
from agents.summarize_insights import generate_summary_from_df
from agents.memory_logger import log_session
from exports.exports_utils import export_summary_to_pdf

def insightgpt_agent_loop(df: pd.DataFrame, goal_prompt: str = "auto", model_mode: str = "cloud") -> None:
    state = initialize_state(goal_prompt)
    st.write("🤖 **InsightGPT Agent: Starting Autonomous Analysis...**")

    while state['current_stage'] != "done":
        current = state['current_stage']
        st.info(f"🔄 Running Stage: `{current}`")

        try:
            if current == "eda":
                perform_eda(df)
                st.success("📊 EDA Completed.")
                state["completed_stages"].append(current)

            elif current == "kpi_extraction":
                kpis = extract_kpis(df)
                st.success("📈 KPIs Extracted.")
                st.json(kpis)
                state["completed_stages"].append(current)

            elif current == "charting":
                smart_chart_agent(df)
                st.success("📉 Charts Generated.")
                state["completed_stages"].append(current)

            elif current == "summary":
                summary = generate_summary_from_df(df, model_mode=model_mode, goal=state["goal"])
                st.success("🧠 Insight Summary Generated.")
                st.markdown(f"#### Summary Preview\n\n{summary}")
                state["completed_stages"].append(current)

                # Try PDF export here after summary
                try:
                    pdf_path = export_summary_to_pdf(summary)
                    st.success(f"📄 Report exported to PDF at: `{pdf_path}`")
                except Exception as e:
                    st.warning(f"⚠️ PDF export failed: {e}")

            elif current == "qna":
                st.write("💬 Q&A Stage (manual for now).")
                state["completed_stages"].append(current)

        except Exception as e:
            st.error(f"❌ Error during `{current}`: {e}")

        state["current_stage"] = STAGE_TRANSITIONS.get(current, "done")
        log_session(state, df_shape=df.shape)
        st.info("📝 Session logged to memory_log.json")

    st.success("✅ InsightGPT Agent: All Steps Completed.")
    st.markdown(f"**🧾 Completed Stages:** `{', '.join(state['completed_stages'])}`")




















