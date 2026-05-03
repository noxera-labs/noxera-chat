"""
Noxera Labs – RAG Demo System v3
FastAPI + ChromaDB (persistent) + Claude Sonnet
Documents are auto-ingested from ./docs/ on startup.
"""

import hashlib
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import anthropic
import chromadb
import pdfplumber
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_DIR = Path("./data/chroma")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = Path("./docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path("./static")

# ──────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=str(DATA_DIR))
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="noxera_documents",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

# ──────────────────────────────────────────────
# Anthropic — async client so streaming doesn't block FastAPI's event loop
# ──────────────────────────────────────────────
anthropic_client = anthropic.AsyncAnthropic()
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """Du bist ein präziser Assistent für die hinterlegten Unternehmens-Dokumente.
Beantworte Fragen ausschließlich auf Basis der bereitgestellten Kontextabschnitte.
Wenn die Antwort nicht im Kontext steht, sage das klar.
Verweise auf Quellen mit Kurzmarkern wie [1], [2] direkt im Text — passend zur Quellenliste, die der Anwender unter deiner Antwort sieht.
Antworte natürlich, in der Sprache der Frage, ohne unnötiges Vorgeplänkel."""
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
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


def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def index_pdf(path: Path) -> tuple[str, int]:
    """Index a PDF file. Returns (status, chunks_added)."""
    content = path.read_bytes()
    fhash = file_hash(content)
    doc_id = f"{path.stem}__{fhash}"

    existing = collection.get(where={"doc_id": doc_id}, limit=1)
    if existing.get("ids"):
        return ("skipped", 0)

    text = extract_pdf_text(content)
    if not text:
        return ("empty", 0)

    chunks = chunk_text(text)
    if not chunks:
        return ("too_short", 0)

    chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": path.name,
            "chunk_index": i,
            "source_path": str(path.relative_to(DOCS_DIR)),
        }
        for i in range(len(chunks))
    ]
    collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas)
    return ("indexed", len(chunks))


def sync_docs_folder():
    """Index all PDFs in ./docs/, drop chunks for files that no longer exist."""
    pdfs = sorted(DOCS_DIR.rglob("*.pdf"))
    seen_doc_ids: set[str] = set()
    summary = {"indexed": 0, "skipped": 0, "empty": 0, "too_short": 0, "removed": 0}

    for pdf in pdfs:
        try:
            content = pdf.read_bytes()
            doc_id = f"{pdf.stem}__{file_hash(content)}"
            seen_doc_ids.add(doc_id)
            status, _ = index_pdf(pdf)
            summary[status] = summary.get(status, 0) + 1
        except Exception as e:
            print(f"[ingest] {pdf.name}: error {e}")

    # Remove stale docs (not in folder anymore)
    try:
        existing = collection.get()
        stale_ids: set[str] = set()
        for meta in existing.get("metadatas", []) or []:
            if not meta:
                continue
            d = meta.get("doc_id")
            if d and d not in seen_doc_ids:
                stale_ids.add(d)
        for d in stale_ids:
            entries = collection.get(where={"doc_id": d})
            if entries.get("ids"):
                collection.delete(ids=entries["ids"])
                summary["removed"] += 1
    except Exception as e:
        print(f"[ingest] cleanup warning: {e}")

    print(f"[ingest] sync complete: {summary} (total chunks in store: {collection.count()})")
    return summary


# ──────────────────────────────────────────────
# Lifespan: ingest on startup
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    sync_docs_folder()
    yield


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(title="Noxera Labs RAG Demo", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    n_results: int = 5


# ──────────────────────────────────────────────
# Retrieval helper
# ──────────────────────────────────────────────
def retrieve(question: str, n_results: int) -> tuple[list[str], list[dict], list[float]]:
    """Returns chunks, metadatas, distances — already de-duplicated and aligned."""
    total = collection.count()
    if total == 0:
        return [], [], []
    n = min(n_results, total)
    results = collection.query(query_texts=[question], n_results=n)
    chunks = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    dists = results.get("distances", [[]])[0] or []

    # Drop duplicate (filename, chunk_index) hits so citation indices stay contiguous
    seen: set = set()
    out_chunks: list[str] = []
    out_metas: list[dict] = []
    out_dists: list[float] = []
    for chunk, m, d in zip(chunks, metas, dists):
        key = (m.get("filename"), m.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        out_chunks.append(chunk)
        out_metas.append(m)
        out_dists.append(d)
    return out_chunks, out_metas, out_dists


def build_messages(question: str, chunks: list[str], metadatas: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[{i+1}] Quelle: {m.get('filename', 'Unbekannt')}, Abschnitt {m.get('chunk_index', 0)+1}\n{chunk}"
        for i, (chunk, m) in enumerate(zip(chunks, metadatas))
    )
    return [{"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {question}"}]


def sources_payload(metadatas: list[dict], distances: list[float]) -> list[dict]:
    return [
        {
            "index": i + 1,
            "filename": m.get("filename", "Unbekannt"),
            "chunk_index": (m.get("chunk_index") or 0) + 1,
            "relevance_score": round(1 - dist, 4),
        }
        for i, (m, dist) in enumerate(zip(metadatas, distances))
    ]


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.post("/query")
async def query_documents(req: QueryRequest):
    """Non-streaming fallback (returns full answer + sources)."""
    chunks, metadatas, distances = retrieve(req.question, req.n_results)
    if not chunks:
        raise HTTPException(404, "Keine relevanten Abschnitte gefunden.")

    message = await anthropic_client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=build_messages(req.question, chunks, metadatas),
    )
    return {
        "answer": message.content[0].text,
        "sources": sources_payload(metadatas, distances),
    }


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming endpoint via Server-Sent Events.

    Event types:
      - sources: emitted once before the answer starts
      - token: incremental text chunks
      - done: end of stream
      - error: error message
    """
    chunks, metadatas, distances = retrieve(req.question, req.n_results)

    async def event_generator() -> AsyncIterator[bytes]:
        if not chunks:
            yield sse("error", {"message": "Keine relevanten Abschnitte gefunden."})
            return

        # Send sources first so the UI can render placeholders before the answer arrives
        yield sse("sources", {"sources": sources_payload(metadatas, distances)})

        try:
            async with anthropic_client.messages.stream(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=build_messages(req.question, chunks, metadatas),
            ) as stream:
                async for text in stream.text_stream:
                    yield sse("token", {"text": text})
            yield sse("done", {})
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chunks_indexed": collection.count(),
        "version": "3.0.0",
    }


@app.post("/admin/reindex")
async def reindex():
    """Re-scan ./docs/ and update the vector store. Idempotent."""
    summary = sync_docs_folder()
    return {"status": "ok", "summary": summary, "total_chunks": collection.count()}


# ──────────────────────────────────────────────
# Static files (must be last so API routes win)
# ──────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
