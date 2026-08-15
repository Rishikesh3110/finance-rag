"""
api/main.py
Optional FastAPI backend (bonus marks).

Run with:  uvicorn api.main:app --reload
Docs at:   http://localhost:8000/docs
"""

import sys
from pathlib import Path

# allow "import ingest, rag" from the project root when running from api/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import List, Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from ingest import ingest_files, collection_stats
from rag import answer_question

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Quarterly Financial Reports RAG API")


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    saved_paths = []
    for f in files:
        out_path = DATA_DIR / f.filename
        content = await f.read()
        with open(out_path, "wb") as fh:
            fh.write(content)
        saved_paths.append(out_path)

    result = ingest_files(saved_paths)
    return result  # {"files": 3, "chunks": 214}


@app.post("/ask")
def ask(req: AskRequest):
    result = answer_question(req.question, top_k=req.top_k)
    return result  # {"answer": "...", "sources": [{"file": "...", "page": 4}]}


@app.get("/stats")
def stats():
    return collection_stats()
