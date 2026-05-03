# Noxera Labs – RAG Showcase

Vorzeige-Produkt für Kundenpräsentationen. Sieht aus wie ein normaler LLM-Chat —
unter der Haube läuft RAG über die Dokumente, die du im `docs/`-Ordner ablegst.

## Stack

| Komponente      | Technologie                    |
|-----------------|--------------------------------|
| Backend         | FastAPI + Uvicorn              |
| Vektorsuche     | ChromaDB (persistent)          |
| PDF-Extraktion  | pdfplumber                     |
| LLM             | Anthropic Claude Sonnet 4 (Streaming) |
| Frontend        | Vanilla HTML/CSS/JS, kein Build |

## Funktionsweise

1. **PDFs in `docs/` ablegen** (lokal oder via Volume-Mount in Coolify)
2. **Beim Start scannt das Backend** den Ordner und indexiert neue Dateien automatisch
3. **Idempotent:** schon indexierte Dateien werden übersprungen, gelöschte aus dem Index entfernt
4. **Frontend** zeigt nur den Chat — keine Upload-UI, keine Document-Liste
5. **Streaming-Antworten** via Server-Sent Events
6. **Inline-Quellen** im Antworttext (`[1]`, `[2]` …) — klickbar, scrollt zur Quelle

## Lokal starten

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. API Key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. PDFs in docs/ legen
cp ~/Downloads/firmenprofil.pdf docs/

# 4. Server starten
uvicorn main:app --reload --port 8000
```

→ http://localhost:8000

## Deployment (Coolify / Hetzner)

1. Repo in Coolify als **Dockerfile**-App hinzufügen
2. Environment Variables setzen:
   - `ANTHROPIC_API_KEY=sk-ant-...`
   - Optional: `ANTHROPIC_MODEL=claude-sonnet-4-20250514`
   - Optional: `SYSTEM_PROMPT="…"` für kundenspezifisches Verhalten
3. **Persistent Storage konfigurieren** (kritisch — sonst gehen Daten beim Redeploy verloren):
   - `/app/data` → persistentes Volume (ChromaDB Index)
   - `/app/docs` → persistentes Volume (PDF-Quellen)
4. Port `8000` exponieren

PDFs in das `docs/`-Volume kopieren (z.B. via Coolify-Terminal oder SCP), dann
`POST /admin/reindex` aufrufen — fertig.

## API

| Method | Pfad              | Beschreibung                              |
|--------|-------------------|-------------------------------------------|
| `POST` | `/query/stream`   | RAG-Anfrage mit SSE-Streaming             |
| `POST` | `/query`          | Synchroner Fallback (komplette Antwort)   |
| `POST` | `/admin/reindex`  | `docs/` neu scannen (idempotent)          |
| `GET`  | `/health`         | Status + Chunk-Count                      |

## Customization pro Kunde

- **System Prompt** via `SYSTEM_PROMPT` env var überschreiben
- **Modell** via `ANTHROPIC_MODEL` env var (z.B. `claude-haiku-4-5-20251001` für günstiger)
- **Branding** in `static/index.html` anpassen (Logo, Titel, Beispielfragen, Farben in `:root`)
