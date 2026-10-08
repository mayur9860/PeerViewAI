# PeerReview AI — Academic Writing Coach

An AI-powered academic writing coach that profiles researcher writing behavior, recommends sections to revise, provides RAG-powered craft guidance search, and runs a Socratic critique agent structurally forbidden from writing replacement prose.

## Features

- **Writer Profiling**: 15-feature analysis classifies writers into 8 archetypes (The Overclaimer, The Undersold Contributor, etc.)
- **Section Priority Ranking**: XGBoost-based recommendation of which section to revise next
- **Craft Guidance Search**: RAG pipeline (self-query → bi-encoder → filter → cross-encoder) over 40 curated writing guides
- **Socratic Critique Agent**: LangGraph pipeline (Critic → Coach → Reviewer) with structural guardrails preventing replacement prose

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy 2.x, Pydantic v2, Python 3.11+
- **Database**: SQLite (Postgres-compatible schema)
- **ML**: scikit-learn (GMM), XGBoost, sentence-transformers
- **LLM**: Google Gemini (configurable model)
- **Vector DB**: ChromaDB (local persistent)
- **Agentic**: LangGraph, LangChain
- **Frontend**: Vanilla HTML/CSS/JS, KaTeX for math rendering

## Quick Start

```bash
cd peerreview_ai/backend

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Generate ML model artifacts (fallback mode for first run)
python generate_artifacts_fallback.py

# Ingest craft guidance corpus into ChromaDB
python pipeline.py --ingest

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Project Structure

```
peerreview_ai/
├── backend/
│   ├── main.py                  # FastAPI app + all routes
│   ├── config.py                # Settings loader from .env
│   ├── models.py                # SQLAlchemy ORM models
│   ├── schemas.py               # Pydantic request/response models
│   ├── database.py              # Engine, session factory, get_db
│   ├── ml_engine.py             # 15 features + GMM + XGBoost
│   ├── persona_prompts.py       # Archetype labels + persona templates
│   ├── rag_engine.py            # ChromaDB search pipeline
│   ├── agent_graph.py           # LangGraph Socratic critique agent
│   ├── pipeline.py              # Corpus ingestion CLI
│   ├── generate_artifacts_fallback.py  # Pure-Python model stubs
│   ├── requirements.txt
│   ├── .env.example
│   ├── artifacts/               # Trained model pickles
│   ├── tests/
│   │   ├── test_ml_engine.py
│   │   ├── test_rag_engine.py
│   │   └── test_agent_graph.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── train/
│   ├── train_gmm.ipynb          # GMM training + BIC plot
│   └── train_xgboost.ipynb      # XGBoost ranker training
├── data/
│   ├── common_words_10k.txt     # Wordlist for jargon_ratio
│   └── craft_corpus/            # 40 craft guidance .md files
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/users` | Create user |
| POST | `/drafts` | Create draft |
| POST | `/drafts/{id}/sections` | Upsert a section |
| POST | `/drafts/{id}/analyze` | Run feature engineering + GMM |
| GET | `/drafts/{id}/recommend` | XGBoost section ranking |
| POST | `/search` | RAG craft guidance search |
| POST | `/sections/{id}/critique` | LangGraph Socratic critique |
| GET | `/drafts/{id}/sections/{id}/history` | Critique session history |

## Running Tests

```bash
cd peerreview_ai/backend
python -m pytest tests/ -v
```

## Notes

- **ML Artifacts**: The fallback artifacts (`generate_artifacts_fallback.py`) use pure-Python stubs. For production, train real models using the Jupyter notebooks in `train/`.
- **Gemini API Key**: Required for RAG search filter extraction, the critique agent, and the reviewer's LLM classifier. The ML features work without it.
- **Application Control**: If your machine blocks native DLLs (sklearn/xgboost), use the fallback artifact generator which doesn't require DLL-dependent libraries at runtime.
