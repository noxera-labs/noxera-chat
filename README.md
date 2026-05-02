# Noxera Labs – RAG Demo v2

Dokument-Intelligence Demo für Kundenpräsentationen.  
PDFs hochladen → indexieren → Fragen stellen → Antworten mit Quellenangaben.

## Stack

| Komponente | Technologie |
|---|---|
| Backend | FastAPI + Uvicorn |
| Vektorsuche | ChromaDB (persistent) |
| PDF-Extraktion | pdfplumber |
| LLM | Claude Sonnet 4 |
| Frontend | HTML/CSS/JS (kein Build-Step) |

## Lokal starten

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. API Key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Starten
uvicorn main:app --reload --port 8000
```

→ http://localhost:8000

## Deployment (Coolify / Hetzner)

1. Repo in Coolify als **Dockerfile**-App hinzufügen
2. Environment Variable setzen: `ANTHROPIC_API_KEY=sk-ant-...`
3. Volume mounten für Persistenz: `/app/data` → persistentes Volume
4. Port `8000` exponieren

## API

| Method | Pfad | Beschreibung |
|---|---|---|
| `POST` | `/upload` | PDF hochladen & indexieren |
| `POST` | `/query` | RAG-Anfrage stellen |
| `GET` | `/documents` | Alle Dokumente |
| `DELETE` | `/documents/{id}` | Dokument löschen |
| `GET` | `/health` | Status + Chunk-Count |

```bash
# Upload
curl -X POST http://localhost:8000/upload -F "file=@dokument.pdf"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Was sind die Hauptpunkte?"}'
```

## Persistenz

Daten werden in `./data/chroma/` gespeichert und überleben Neustarts.  
Beim nächsten Start werden alle Dokumente automatisch aus dem Index wiederhergestellt.
