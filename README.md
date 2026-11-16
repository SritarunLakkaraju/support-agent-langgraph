# Customer Support Agent for a Small Business (LangGraph)

A support bot for a fictional phone-repair shop. It answers FAQs (RAG), checks repair status from the shop's
order sheet, and books appointments over several turns (slot filling, with no double booking). It hands off to a
human when unsure or when the customer is upset. It is built on LangGraph with per-conversation memory and serves
a web widget plus Telegram and WhatsApp webhooks. A weekly report gives the % resolved without a human, CSAT and
the unanswered questions.

```bash
python -m fde6_support_agent.evaluate            # 40 scripted conversations + 15 held-out messages
uvicorn fde6_support_agent.api:app --port 8006   # chat widget at http://localhost:8006
```

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m fde6_support_agent.data
python -m fde6_support_agent.evaluate
uvicorn fde6_support_agent.api:app --port 8080
pytest -q
```

Everything runs **offline**: no API keys or cloud account are needed. The steps that would use an LLM have
deterministic offline implementations, so tests are reproducible and answers cannot be hallucinated.

## Results (synthetic, seeded data)

- 40 scripted conversations (60 turns): 100.0% turn accuracy, 77.5% resolved without a human, handoff precision/recall 100.0%/100.0%. These were used for tuning.
- **Held-out messages (never tuned on): 46.7%**; of the misses 7 were safe handoffs and 1 a wrong answer. Lexical matching cannot know that 'address' means 'location'; Vertex AI embeddings are the fix.
- No real customer deployment yet: these are scripted conversations.

## Switching to the real services

| Environment variable | Effect |
|---|---|
| `GEMINI_API_KEY` (+ `GEMINI_MODEL`) | use Gemini for the LLM steps |
| `GOOGLE_CLOUD_PROJECT` + `FDE_USE_VERTEX=1` | Vertex AI text-embedding-005 instead of the offline hashing embedder |
| `FDE_VECTOR_BACKEND=chroma` or `pgvector` (+ `PGVECTOR_DSN`) | Chroma or PostgreSQL/pgvector instead of the NumPy vector store |

The cloud adapters are written against the official SDKs but have not been run against live services yet.
Deploy to Cloud Run with the included `Dockerfile` (`gcloud run deploy --source .`).

## Layout

- `fde6_support_agent/`: the project (see the module docstrings; each file explains its role)
- `common/`: shared tokeniser + BM25, hashing embedder / Vertex embedder, vector stores (NumPy, Chroma, pgvector), Gemini client
- `docs/FDE_Projects_Guide.pdf`: a detailed write-up of this project and its five siblings

## Part of a series of six Forward Deployed Engineer projects

- [Agentic RAG Assistant on Google ADK](https://github.com/SritarunLakkaraju/agentic-rag-adk)
- [DealMind v2: Sales Agent on LangGraph](https://github.com/SritarunLakkaraju/dealmind-langgraph)
- [CrewAI Weekly Business Report on a BigQuery-style Warehouse](https://github.com/SritarunLakkaraju/crewai-bigquery-reports)
- [Event-driven Invoice Intelligence Pipeline](https://github.com/SritarunLakkaraju/invoice-intelligence-pipeline)
- [Vector Search from Scratch: HNSW vs FAISS vs Chroma](https://github.com/SritarunLakkaraju/hnsw-vector-search)
