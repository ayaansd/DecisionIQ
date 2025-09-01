import os
import sys
import json
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from typing import Optional, List
import threading
import uuid
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from fastapi.responses import FileResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

MONGODB_URI = os.getenv("MONGODB_URI", "your-default-uri-if-needed")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
mongo_db = mongo_client["decisioniq"]
mongo_collection = mongo_db["analyses"]

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
from agents.proactive_agent import detect_proactive_signals
from agents.alert_summarizer import generate_alert_summary
from agents.build_faiss_index import build_faiss_index
from agents.rag_faiss_agent import run_faiss_rag_agent
from agents.memory_logger import log_feedback, load_recent_sessions
from agents.slack_utils import send_summary_to_slack
from monitor.continuous_monitor import watch_for_new_files
from logic.auto_pipeline import run_auto_pipeline
from agents.url_monitor_agent import start_url_monitoring
from agents.analysis_modes import (
    generate_swot_analysis,
    generate_financial_analysis,
    generate_market_research,
    generate_process_optimization
)

# --- FASTAPI SETUP ---
app = FastAPI(
    title="DecisionIQ API",
    description="Backend for the DecisionIQ AI Analyst Dashboard.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the outputs directory to serve charts as static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# --- GLOBAL STATE FOR BACKGROUND TASKS AND IN-MEMORY DATA ---
monitor_thread: Optional[threading.Thread] = None
in_memory_data: dict = {}

# --- UTILITY FUNCTIONS ---
def read_uploaded_file(file_id: str):
    """Loads a DataFrame from a temporary location using a file_id."""
    if file_id not in in_memory_data:
        raise HTTPException(status_code=404, detail="File ID not found.")

    file_content = in_memory_data[file_id]['content']
    filename = in_memory_data[file_id]['filename']

    try:
        if filename.endswith(".csv"):
            return pd.read_csv(BytesIO(file_content))
        elif filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(BytesIO(file_content))
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

# --- API ENDPOINTS ---

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a dataset, stores it, and returns a unique file_id."""
    file_id = str(uuid.uuid4())
    file_content = await file.read()

    in_memory_data[file_id] = {
        "content": file_content,
        "filename": file.filename,
    }

    df = read_uploaded_file(file_id)
    build_faiss_index(df)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "preview": df.head().to_dict(orient="records")
    }

@app.post("/api/eda")
async def run_eda(file_id: str = Query(...)):
    """Performs Exploratory Data Analysis on the uploaded dataset."""
    df = read_uploaded_file(file_id)
    eda_result = perform_eda(df)
    return JSONResponse(content={"eda_result": eda_result})

@app.post("/api/kpis")
async def get_kpis(file_id: str = Query(...)):
    df = read_uploaded_file(file_id)
    kpis = extract_kpis(df)
    return JSONResponse(content={"kpis": kpis})

@app.post("/api/charts")
async def generate_charts(file_id: str = Query(...)):
    df = read_uploaded_file(file_id)
    chart_paths, chart_summaries = smart_chart_agent(df)
    return JSONResponse(content={
        "charts": chart_paths,
        "summaries": chart_summaries
    })

@app.post("/api/summary")
async def generate_summary(file_id: str = Query(...), model_mode: str = "cloud"):
    df = read_uploaded_file(file_id)
    summary = generate_summary_from_df(df, domain="auto", model_mode=model_mode)
    return JSONResponse(content={"summary": summary})

@app.post("/api/regenerate_summary")
async def regenerate_summary_endpoint(
    file_id: str = Query(...),
    summary: str = Query(...),
    feedback: str = Query(...),
    model_mode: str = "cloud"
):
    df = read_uploaded_file(file_id)
    improved_summary = regenerate_summary_from_feedback(df, summary, feedback, model_mode)
    return JSONResponse(content={"improved_summary": improved_summary})

@app.post("/api/qa")
async def ask_question(
    file_id: str = Query(...),
    question: str = Query(...),
    model_mode: str = "cloud"
):
    df = read_uploaded_file(file_id)
    natural_keywords = ["who", "when", "which", "show", "list", "customer", "order", "message", "chat", "complain", "comment", "email", "review"]
    question_lower = question.lower()
    if any(kw in question_lower for kw in natural_keywords):
        answer = run_faiss_rag_agent(question, model_mode=model_mode)
        return JSONResponse(content={"mode": "faiss", "answer": answer})
    else:
        output = run_rag_qa_agent(df, question, model_mode=model_mode)
        return JSONResponse(content={"mode": "codegen", "output": output})

@app.post("/api/goal")
async def run_goal_agent(
    file_id: str = Query(...),
    goal: str = Query(...)
):
    df = read_uploaded_file(file_id)
    report = run_goal_pipeline(df, goal)
    return JSONResponse(content={"report": report})

@app.post("/api/alerts")
async def get_proactive_alerts(file_id: str = Query(...), model_mode: str = "cloud"):
    df = read_uploaded_file(file_id)
    proactive_signals = detect_proactive_signals(df)
    alert_summary = generate_alert_summary(proactive_signals, model_mode=model_mode)
    return JSONResponse(content={
        "summary": alert_summary,
        "raw_alerts": proactive_signals.strip().split("\n")
    })

@app.get("/api/sessions")
def get_recent_sessions():
    return JSONResponse(content={"sessions": load_recent_sessions(limit=5)})

@app.post("/api/log_feedback")
async def log_user_feedback(feedback_entry: dict):
    log_feedback(feedback_entry["session_id"], feedback_entry["feedback"])
    return {"status": "success"}

@app.post("/api/slack/send_summary")
async def send_to_slack(summary: str, webhook_url: str):
    send_summary_to_slack(summary, webhook_url)
    return {"status": "success"}

@app.post("/api/auto_analysis/start")
def start_auto_analysis(model_mode: str = "cloud"):
    global monitor_thread
    if monitor_thread and monitor_thread.is_alive():
        return {"status": "info", "message": "Auto-analysis monitor is already running."}
    def run_monitor_task():
        watch_for_new_files(model_mode=model_mode)
    monitor_thread = threading.Thread(target=run_monitor_task, daemon=True)
    monitor_thread.start()
    return {"status": "success", "message": "Auto-analysis monitor started."}


@app.post("/api/auto_analysis/url")
def start_url_based_monitoring(
    url: str = Query(...),
    interval_seconds: int = Query(60),
    model_mode: str = Query("cloud")
):
    try:
        start_url_monitoring(url, interval_seconds=interval_seconds, model_mode=model_mode)
        return {"status": "success", "message": f"Started monitoring URL: {url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auto_analysis/status")
def get_auto_analysis_status():
    global monitor_thread
    is_running = monitor_thread and monitor_thread.is_alive()
    return {"is_running": is_running}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/analyze")
async def analyze_text_or_file(
    text: Optional[str] = Query(None),
    file_id: Optional[str] = Query(None),
    analysis_type: str = Query("summary"),
    model_mode: str = Query("cloud")
):
    try:
        if file_id:
            df = read_uploaded_file(file_id)
            content = df.to_csv(index=False)
        elif text:
            df = pd.DataFrame({"text": [text]})
            content = text
        else:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'file_id'.")

        if analysis_type == "summary":
            result = generate_summary_from_df(df, model_mode=model_mode)

        elif analysis_type == "kpi":
            result = extract_kpis(df)

        elif analysis_type == "swot":
            result = generate_swot_analysis(df, model_mode=model_mode)

        elif analysis_type == "financial":
            result = generate_financial_analysis(df, model_mode=model_mode)

        elif analysis_type == "market":
            result = generate_market_research(df, model_mode=model_mode)

        elif analysis_type == "optimization":
            result = generate_process_optimization(df, model_mode=model_mode)

        else:
            result = "❌ Unsupported analysis type"


        # Save to MongoDB
        analysis_doc = {
            "input_type": "file" if file_id else "text",
            "input_text": content,
            "analysis_type": analysis_type,
            "model_mode": model_mode,
            "result": result,
            "timestamp": datetime.utcnow()
        }
        inserted = await mongo_collection.insert_one(analysis_doc)

        return JSONResponse(content={
            "analysis_id": str(inserted.inserted_id),
            "result": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyses")
async def list_analyses():
    """List all saved analyses in reverse chronological order."""
    try:
        cursor = mongo_collection.find().sort("timestamp", -1)
        analyses = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
            doc["timestamp"] = doc["timestamp"].isoformat() if "timestamp" in doc else None
            analyses.append(doc)
        return JSONResponse(content={"analyses": analyses})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve a specific analysis by its MongoDB ID."""
    try:
        obj_id = ObjectId(analysis_id)
        doc = await mongo_collection.find_one({"_id": obj_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        doc["_id"] = str(doc["_id"])
        doc["timestamp"] = doc["timestamp"].isoformat() if "timestamp" in doc else None
        return JSONResponse(content=doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

@app.post("/api/generate-report")
async def generate_report(analysis_id: str = Query(...)):
    try:
        obj_id = ObjectId(analysis_id)
        doc = await mongo_collection.find_one({"_id": obj_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Analysis not found.")

        # Extract details
        analysis_type = doc.get("analysis_type", "N/A")
        input_text = doc.get("input_text", "")[:3000]  # Truncate if too long
        result = doc.get("result", "")
        timestamp = doc.get("timestamp", "").isoformat() if "timestamp" in doc else ""

        # Generate PDF
        filename = f"{REPORTS_DIR}/report_{analysis_id}.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, height - 40, "📊 DecisionIQ - Business Report")

        c.setFont("Helvetica", 12)
        y = height - 80
        c.drawString(40, y, f"Analysis ID: {analysis_id}")
        y -= 20
        c.drawString(40, y, f"Timestamp: {timestamp}")
        y -= 20
        c.drawString(40, y, f"Analysis Type: {analysis_type}")
        y -= 30

        def draw_wrapped(text, start_y, max_width=500, line_height=14):
            from reportlab.lib.utils import simpleSplit
            lines = simpleSplit(text, "Helvetica", 12, max_width)
            for line in lines:
                nonlocal y
                if y < 40:
                    c.showPage()
                    y = height - 40
                    c.setFont("Helvetica", 12)
                c.drawString(40, y, line)
                y -= line_height

        c.setFont("Helvetica-Bold", 13)
        c.drawString(40, y, "📥 Input:")
        y -= 20
        draw_wrapped(input_text, y)

        c.setFont("Helvetica-Bold", 13)
        y -= 30
        c.drawString(40, y, "📈 Result:")
        y -= 20
        draw_wrapped(result, y)

        c.save()

        return FileResponse(path=filename, filename=os.path.basename(filename), media_type='application/pdf')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
