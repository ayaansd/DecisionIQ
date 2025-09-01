import os
from datetime import datetime

def export_summary_to_html(summary_text: str, kpis: dict = None, chart_dir: str = None) -> str:
    """
    Export a styled HTML report with summary, KPIs, and charts.
    Returns the path to the saved HTML file.
    """
    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join("exports", f"insight_report_{timestamp}.html")

    # HTML Header
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>InsightGPT Report - {timestamp}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f8f9fa;
            color: #212529;
        }}
        h1 {{
            color: #007bff;
        }}
        .summary {{
            background: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .kpi {{
            margin-bottom: 20px;
        }}
        .chart {{
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <h1>📊 InsightGPT Analysis Report</h1>
    <p><strong>Generated:</strong> {timestamp}</p>

    <div class="summary">
        <h2>📝 Summary</h2>
        <p>{summary_text.replace('\n', '<br>')}</p>
    </div>
"""

    # KPIs
    if kpis:
        html += '<div class="kpi"><h2>📌 Key Metrics</h2><ul>'
        for key, value in kpis.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul></div>"

    # Charts
    if chart_dir and os.path.exists(chart_dir):
        html += '<div><h2>📈 Charts</h2>'
        for filename in sorted(os.listdir(chart_dir)):
            if filename.endswith((".png", ".jpg")):
                chart_path = os.path.join(chart_dir, filename)
                rel_path = os.path.relpath(chart_path, "exports")
                html += f'<div class="chart"><img src="{rel_path}" width="600"></div>'
        html += "</div>"

    # Footer
    html += "</body></html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path
