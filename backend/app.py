import os
import sys
import shutil
import subprocess
from typing import List
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.pipeline import run_pipeline

app = FastAPI(
    title="OmniBrain API",
    description="Agentic Multi-Modal RAG Orchestrator API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    route: str


@app.get("/")
def health_check():
    return {"status": "running", "service": "OmniBrain Multi-Agent RAG"}


@app.post("/query", response_model=QueryResponse)
def execute_query(payload: QueryRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = run_pipeline(payload.question)
        return {
            "answer": result.get("answer", "No answer generated."),
            "route": result.get("route", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Receives PDF files, saves them to data/, and updates vector store."""
    try:
        saved = []
        for file in files:
            file_path = os.path.join(DATA_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved.append(file.filename)

        # Trigger RAG ingest
        proc = subprocess.run(
            [sys.executable, "-m", "rag.ingest"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        if proc.returncode != 0:
            raise Exception(f"Ingestion failed: {proc.stderr}")

        return {"status": "success", "indexed_files": saved}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)