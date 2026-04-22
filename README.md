# aaptor-rag-service

FAISS-based semantic retrieval service for the Aaptor assessment platform.

## Supported Competencies
- `aiml` — AI/ML datasets (180 entries)
- `dsa` — DSA problems (LeetCode enriched)
- `devops` — DevOps scenarios (coming soon)
- `data_engineering` — Data engineering problems (coming soon)
- `design` — System design problems (coming soon)
- `prompt_engineering` — Prompt engineering tasks (coming soon)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your settings
uvicorn main:app --host 0.0.0.0 --port 7002
```

## Data Structure

```
data/
├── aiml/
│   ├── catalog.json      ← source of truth
│   ├── faiss.index       ← built from catalog
│   └── metadata.json     ← FAISS vector → catalog ID mapping
├── dsa/
│   ├── catalog.json
│   ├── faiss.index
│   └── metadata.json
└── ...
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check + index stats |
| POST | `/api/v1/retrieve` | Search for matching dataset/problem |
| POST | `/api/v1/ingest/{competency}` | Add new entries + rebuild index |
| POST | `/api/v1/rebuild/{competency}` | Rebuild index from catalog |

## Rebuild Index

```bash
curl -X POST http://localhost:7002/api/v1/rebuild/aiml \
  -H "X-Api-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"use_gpu_model": false}'
```

## Add New Data

```bash
curl -X POST http://localhost:7002/api/v1/ingest/aiml \
  -H "X-Api-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"entries": [{"id": "new-dataset", "name": "...", ...}]}'
```
