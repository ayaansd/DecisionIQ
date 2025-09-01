# pipeline_runner.py

from eda.perform_eda import perform_eda
from kpi.extract_kpis import extract_kpis
from chart.generate_charts import smart_chart_agent
from agents.summarize_insights import generate_summary_from_df
from agents.proactive_agent import detect_proactive_signals
from agents.alert_summarizer import generate_alert_summary

def run_full_analysis_pipeline(df, source_file="unknown.csv"):
    print(f"\n🔍 Running InsightGPT pipeline for {source_file}...")

    # EDA
    perform_eda(df)

    # KPI Extraction
    extract_kpis(df)

    # Charting
    generate_charts(df)

    # Summary
    summary = generate_summary_from_df(df, model_mode="local")

    # Proactive Alerts
    alerts = detect_proactive_signals(df)
    alert_summary = generate_alert_summary(alerts, model_mode="local")

    # Final Output
    print("\n🧠 Insight Summary:\n", summary)
    print("\n🚨 Alert Summary:\n", alert_summary)
