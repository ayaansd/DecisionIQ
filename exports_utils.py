# exports/export_utils.py

import os
from fpdf import FPDF
import datetime

def export_summary_to_pdf(summary_text, output_dir="exports"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"InsightGPT_Report_{datetime.date.today()}.pdf")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "InsightGPT - Summary Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    for line in summary_text.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(file_path)
    return file_path
