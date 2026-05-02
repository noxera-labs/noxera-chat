"""
Noxera Labs – RAG Demo System v2
FastAPI + ChromaDB (persistent) + Claude Sonnet
"""

import io
import uuid
from pathlib import Path

import anthropic
import chromadb
import pdfplumber
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ──────────────────────────────────────────────
# Persistent data directory
# ──────────────────────────────────────────────
DATA_DIR = Path("./data/chroma")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(title="Noxera Labs RAG Demo", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# ChromaDB – persistent on disk
# ──────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=str(DATA_DIR))
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="noxera_documents",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

# ──────────────────────────────────────────────
# Anthropic
# ──────────────────────────────────────────────
anthropic_client = anthropic.Anthropic()

# ──────────────────────────────────────────────
# In-memory registry (rebuilt from ChromaDB on startup)
# ──────────────────────────────────────────────
doc_registry: dict[str, dict] = {}


def _rebuild_registry():
    """Rebuild doc registry from existing ChromaDB data on startup."""
    try:
        all_items = collection.get()
        for meta in all_items.get("metadatas", []):
            if not meta:
                continue
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in doc_registry:
                doc_registry[doc_id] = {
                    "id": doc_id,
                    "filename": meta.get("filename", "Unbekannt"),
                    "chunks": 0,
                    "chars": meta.get("chars", 0),
                    "preview": meta.get("preview", ""),
                }
        # Count chunks per doc
        for meta in all_items.get("metadatas", []):
            if not meta:
                continue
            doc_id = meta.get("doc_id")
            if doc_id in doc_registry:
                doc_registry[doc_id]["chunks"] += 1
    except Exception as e:
        print(f"Registry rebuild warning: {e}")


_rebuild_registry()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_pdf_text(content: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Nur PDF-Dateien werden unterstützt.")

    content = await file.read()
    text = extract_pdf_text(content)

    if not text:
        raise HTTPException(400, "Kein Text aus PDF extrahierbar.")

    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)

    if not chunks:
        raise HTTPException(400, "Dokument zu kurz.")

    preview = text[:300].replace("\n", " ")

    chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_index": i,
            "chars": len(text),
            "preview": preview,
        }
        for i in range(len(chunks))
    ]

    collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas)

    doc_registry[doc_id] = {
        "id": doc_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "chars": len(text),
        "preview": preview,
    }

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_indexed": len(chunks),
        "status": "success",
    }


class QueryRequest(BaseModel):
    question: str
    n_results: int = 5


@app.post("/query")
async def query_documents(req: QueryRequest):
    if not doc_registry:
        raise HTTPException(400, "Keine Dokumente vorhanden.")

    n = min(req.n_results, collection.count())
    results = collection.query(query_texts=[req.question], n_results=n)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not chunks:
        raise HTTPException(404, "Keine relevanten Abschnitte gefunden.")

    context = "\n\n---\n\n".join(
        f"[Quelle {i+1}: {m['filename']}, Abschnitt {m['chunk_index']+1}]\n{chunk}"
        for i, (chunk, m) in enumerate(zip(chunks, metadatas))
    )

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""Du bist ein präziser Dokumenten-Assistent.
Beantworte Fragen ausschließlich auf Basis der bereitgestellten Kontextabschnitte.
Wenn die Antwort nicht im Kontext enthalten ist, kommuniziere das klar.
Verweise auf Quellen wo sinnvoll (z.B. "Laut Quelle 2...").
Antworte in der gleichen Sprache wie die gestellte Frage.""",
        messages=[
            {
                "role": "user",
                "content": f"Kontext:\n{context}\n\nFrage: {req.question}",
            }
        ],
    )

    answer = message.content[0].text

    sources = []
    seen = set()
    for m, dist in zip(metadatas, distances):
        key = f"{m['filename']}_{m['chunk_index']}"
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "filename": m["filename"],
                    "chunk_index": m["chunk_index"] + 1,
                    "relevance_score": round(1 - dist, 4),
                }
            )

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
    }


@app.get("/documents")
async def list_documents():
    return list(doc_registry.values())


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in doc_registry:
        raise HTTPException(404, "Dokument nicht gefunden.")

    existing = collection.get(where={"doc_id": doc_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    del doc_registry[doc_id]
    return {"status": "deleted", "doc_id": doc_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "documents": len(doc_registry),
        "chunks_indexed": collection.count(),
        "version": "2.0.0",
    }


# Static files last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
