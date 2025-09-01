import os 
import pandas as pd
import typer
from colorama import Fore, Style, init
from rich.console import Console
from rich.panel import Panel
from eda.perform_eda import perform_eda
from kpi.extract_kpis import extract_kpis
from chart.generate_charts import smart_chart_agent
from agents.summarize_insights import generate_summary_from_df
import requests
from io import StringIO, BytesIO
from urllib.parse import urlparse

init(autoreset=True)
app = typer.Typer(help="📊 InsightGPT - Your Local GenAI Business Analyst CLI", invoke_without_command=True)
console = Console()

def load_data_and_preview(path_or_url: str):
    """Load a file and display a preview, handling file existence and format errors"""
    console.print(Panel.fit(f"Loading File: [bold]{path_or_url}[/bold]", title="InsightGPT", subtitle="Data Preview"))
    df = None
    try:
        if urlparse(path_or_url).scheme in ('http', 'https'):
            console.print("[blue]🌐 Detected remote file URL, downloading...[/blue]")
            response = requests.get(path_or_url)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")

            if 'csv' in content_type or path_or_url.endswith(".csv"):
                df = pd.read_csv(StringIO(response.text))
            elif 'xlsx' in content_type or path_or_url.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(BytesIO(response.content))
            elif 'json' in content_type or path_or_url.endswith(".json"):
                df = pd.read_json(BytesIO(response.content))
            else:
                console.print(Panel.fit(f"[red] Unsupported file format: {content_type}[/red]"), style="red")
                raise typer.Exit(code=1)
        else:
            if not os.path.exists(path_or_url):
                console.print(Panel.fit(f"[red] File not found at '{path_or_url}'[/red]"), style="red")
                raise typer.Exit(code=1)

            if path_or_url.endswith(".csv"):
                df = pd.read_csv(path_or_url)
            elif path_or_url.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(path_or_url)
            elif path_or_url.endswith(".json"):
                df = pd.read_json(path_or_url)
            else:
                console.print(Panel.fit(f"[red] Unsupported file format: {path_or_url}[/red]"), style="red")
                raise typer.Exit(code=1)

        console.print(Panel.fit(" File Loaded Successfully!", title="Status", style="green"))
        console.print(df.head())
        return df

    except Exception as e:
        console.print(Panel.fit(f"[red]❌ Error loading file: {e}[/red]"), style="red")
    raise typer.Exit(code=1)

@app.callback()
def run(
    file: str = typer.Option(..., "--file", "-f", help="Path to input CSV/Excel/JSON file"),
    eda: bool = typer.Option(False, "--eda", help="Run Exploratory Data Analysis"),
    kpi: bool = typer.Option(False, "--kpi", help="Extract Key Performance Indicators"),
    charts: bool = typer.Option(False, "--charts", help="Generate charts from data"),
    summary: bool = typer.Option(False, "--summary", help="Generate AI-powered summary from data"),
    domain: str = typer.Option("auto", "--domain", help="Optional domain context (e.g., finance, marketing, auto)"),
    all_steps: bool = typer.Option(False, "--all", help="Run all steps in sequence"),
):
    """
    📊 InsightGPT: A Local GenAI-Powered Business Analyst CLI
    """
    df = load_data_and_preview(file)

    if all_steps or eda:
        console.rule("[bold cyan]🔍 EDA Summary")
        perform_eda(df)

    if all_steps or kpi:
        console.rule("[bold green]📈 KPI Extraction")
        extract_kpis(df)

    if all_steps or charts:
        console.rule("[bold magenta]📊 Chart Generation")
        smart_chart_agent(df)

    if all_steps or summary:
        console.rule("[bold yellow]🧠 Insight Summary")
        summary_text = generate_summary_from_df(df, domain=domain)
        console.print(summary_text)

if __name__ == "__main__":
    app()
