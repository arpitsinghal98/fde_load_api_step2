
# FDE Loads API (Step 2)

Minimal FastAPI service exposing `/loads/search` and `/loads/{id}` with simple API key auth.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set API_KEY (keep it secret)
```

## 2) Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 3) Test

### Health (no auth required)
```bash
curl http://localhost:8000/health
```

### Search (auth required)
```bash
curl -H "x-api-key: supersecret123" "http://localhost:8000/loads/search?origin=Chicago&min_rate=1500"
```

### Get by ID
```bash
curl -H "x-api-key: supersecret123" "http://localhost:8000/loads/FDE-1001"
```

## 4) Notes
- CSV path defaults to `/mnt/data/loads_sample.csv` but can be overridden with `LOADS_CSV` in `.env`.
- Sorting and pagination supported.
- Date filters accept ISO strings, e.g., `pickup_earliest=2025-09-09T00:00:00`.
