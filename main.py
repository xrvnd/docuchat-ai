from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os
import uuid
import json
from dotenv import load_dotenv
import pypdf
import docx

#my custom libs
from reranker import rerank
from vector_engine import VectorEngine



load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing")

#Vector Engine (Chroma + Embedding)
ve = VectorEngine()

#FASTAPI APP
app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Session Handling
SESSIONS_FILE = "sessions.json"
sessions_data = {}

def load_sessions():
    global sessions_data
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            sessions_data = json.load(f)
    else:
        sessions_data = {}

def save_sessions():
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions_data, f, indent=2)

load_sessions()

# File Text Extraction
def extract_text_from_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        reader = pypdf.PdfReader(file_path)
        return "".join([p.extract_text() for p in reader.pages])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format")

# rag searcging function with cosine reranking
def search_documents(query: str, n_results: int = 3):
    retrieved_docs = ve.search(query, n_results)
    if not retrieved_docs:
        return {"documents": [[]], "metadatas": [[]], "sources": []}

    docs = [d["document"] for d in retrieved_docs]
    metas = [d["metadata"] for d in retrieved_docs]
    filenames = [m.get("filename", "Unknown") for m in metas]

    # Rerank using cosine similarity
    embeddings = ve.embed_documents(docs)
    query_embedding = ve.embed_query(query)
    reranked_docs = rerank(query_embedding, docs, embeddings, top_k=n_results)

    return {
        "documents": [reranked_docs],
        "metadatas": [metas],
        "sources": filenames,
    }

# ---- FastAPI Endpoints ----
@app.get("/")
def home():
    return {"status": "RAG API running", "documents": ve.count()}

@app.post("/query")
async def run_query(body: dict):
    query = body.get("query")
    if not query:
        raise HTTPException(400, "Query missing")

    results = search_documents(query)
    docs = results["documents"][0]

    if not docs:
        return {"response": "I couldn't find relevant documents.", "sources": []}

    answer = ve.generate_answer(query, docs)
    return {"response": answer, "sources": results["sources"]}

@app.post("/admin/upload")
async def upload_document(file: UploadFile = File(...)):
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    file_path = uploads_dir / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_file(str(file_path))
    doc_id = f"doc_{uuid.uuid4()}"

    ve.add_document(doc_id, text, {"filename": file.filename})

    return {"message": f"{file.filename} uploaded", "total_docs": ve.count()}

if __name__ == "__main__":
    import uvicorn
    print("uvicorn running") #checking
    uvicorn.run(app, host="0.0.0.0", port=8000)
