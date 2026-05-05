"""
Noxera Labs — RAG + General AI Chat
FastAPI + ChromaDB (persistent) + Claude
Documents are auto-ingested from ./docs/ on startup (PDF + TXT).
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
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
# Anthropic
# ──────────────────────────────────────────────
anthropic_client = anthropic.AsyncAnthropic()
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """Du bist der KI-Assistent von Noxera Labs — einem unabhängigen Software- und KI-Studio aus Hamburg, Deutschland, gegründet von Noah Wilm und Raphael Ghazaryan.

Wenn Kontext-Abschnitte aus Dokumenten bereitgestellt werden:
- Nutze sie als primäre Grundlage deiner Antwort
- Verweise im Fließtext mit [1], [2] etc. auf die Quellen

Wenn kein Dokumentkontext vorhanden ist:
- Beantworte Fragen aus deinem allgemeinen Wissen

Antworte stets präzise, hilfreich und in der Sprache der Frage.""",
)


# ──────────────────────────────────────────────
# Text helpers
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


def _index_text(filename: str, text: str, source_path: str = "") -> tuple[str, int]:
    """Index extracted text into ChromaDB. Returns (status, chunks_added)."""
    fhash = file_hash(text.encode())
    stem = Path(filename).stem
    doc_id = f"{stem}__{fhash}"

    existing = collection.get(where={"doc_id": doc_id}, limit=1)
    if existing.get("ids"):
        return ("skipped", 0)

    chunks = chunk_text(text)
    if not chunks:
        return ("too_short", 0)

    chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "source_path": source_path or filename,
        }
        for i in range(len(chunks))
    ]
    collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas)
    return ("indexed", len(chunks))


def index_pdf(path: Path) -> tuple[str, int]:
    text = extract_pdf_text(path.read_bytes())
    if not text:
        return ("empty", 0)
    return _index_text(path.name, text, str(path.relative_to(DOCS_DIR)))


def index_txt(path: Path) -> tuple[str, int]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return ("empty", 0)
    return _index_text(path.name, text, str(path.relative_to(DOCS_DIR)))


def sync_docs_folder():
    files = sorted(DOCS_DIR.rglob("*.pdf")) + sorted(DOCS_DIR.rglob("*.txt"))
    seen_doc_ids: set[str] = set()
    summary = {"indexed": 0, "skipped": 0, "empty": 0, "too_short": 0, "removed": 0}

    for f in files:
        try:
            if f.suffix == ".pdf":
                text = extract_pdf_text(f.read_bytes())
            else:
                text = f.read_text(encoding="utf-8", errors="ignore")
            doc_id = f"{f.stem}__{file_hash(text.encode())}"
            seen_doc_ids.add(doc_id)
            status, _ = (index_pdf if f.suffix == ".pdf" else index_txt)(f)
            summary[status] = summary.get(status, 0) + 1
        except Exception as e:
            print(f"[ingest] {f.name}: error {e}")

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

    print(f"[ingest] sync complete: {summary} (total chunks: {collection.count()})")
    return summary


# ──────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    sync_docs_folder()
    yield


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(title="Noxera Labs AI Chat", version="4.1.0", lifespan=lifespan)

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
    model: str | None = None
    web_search: bool = False


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
def retrieve(question: str, n_results: int) -> tuple[list[str], list[dict], list[float]]:
    total = collection.count()
    if total == 0:
        return [], [], []
    n = min(n_results, total)
    results = collection.query(query_texts=[question], n_results=n)
    chunks = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    dists = results.get("distances", [[]])[0] or []

    seen: set = set()
    out_chunks, out_metas, out_dists = [], [], []
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
    if not chunks:
        return [{"role": "user", "content": question}]
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


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    model = req.model or DEFAULT_MODEL
    chunks, metadatas, distances = retrieve(req.question, req.n_results)

    async def event_generator() -> AsyncIterator[bytes]:
        if chunks:
            yield sse("sources", {"sources": sources_payload(metadatas, distances)})

        try:
            stream_kwargs: dict = dict(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=build_messages(req.question, chunks, metadatas),
            )
            if req.web_search:
                stream_kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search"}]
                yield sse("status", {"text": "Web-Suche läuft…"})

            async with anthropic_client.messages.stream(**stream_kwargs) as stream:
                async for text in stream.text_stream:
                    yield sse("token", {"text": text})
            yield sse("done", {})
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".txt"):
        return JSONResponse(
            {"status": "error", "message": "Nur PDF und TXT werden unterstützt"},
            status_code=400,
        )
    content = await file.read()
    if suffix == ".pdf":
        text = extract_pdf_text(content)
        if not text:
            return {"status": "empty", "chunks": 0, "filename": filename}
    else:
        text = content.decode("utf-8", errors="ignore").strip()
        if not text:
            return {"status": "empty", "chunks": 0, "filename": filename}

    status, n = _index_text(filename, text)
    return {"status": status, "chunks": n, "filename": filename}


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": collection.count(), "version": "4.1.0"}


@app.post("/admin/reindex")
async def reindex():
    summary = sync_docs_folder()
    return {"status": "ok", "summary": summary, "total_chunks": collection.count()}


# ──────────────────────────────────────────────
# Static (must be last)
# ──────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
