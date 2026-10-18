# Mutual Funds Recommendation System

An AI-powered recommendation engine for Indian mutual funds. You describe your goal in plain English, and the system finds and ranks suitable funds with retrieval-augmented generation (RAG), now built on **LangChain**. Each pick comes with an explanation grounded in the fund's real numbers.

```
goal ──► LLM query rewrite (Phi-2, optional)
            │
            ▼
   Hybrid retrieval  ─ FAISS (MiniLM embeddings) ┐
   (LangChain)       ─ BM25 keywords             ├─► reciprocal rank fusion
                       filtered by category/risk ┘     over original + rewritten query
            │
            ▼
   Ranking: goal match + returns vs category peers + cost vs peers
            × horizon fit × risk fit
            │
            ▼
   Live NAV (mfapi.in, cached) ─► Explanations (Phi-2 + LoRA via LCEL,
                                   fact-checked, deterministic fallback)
            │
            ▼
   FastAPI  /recommend  /categories  /health  ──►  React frontend
```

## What changed in v2

### Where LangChain is used

| Piece | LangChain component |
|---|---|
| Fund records | `Document` with filterable metadata (category, risk) |
| Embeddings | `HuggingFaceEmbeddings` (`langchain-huggingface`), unit-normalised |
| Vector store | `FAISS` (`langchain-community`), persisted with `save_local` / `load_local`, rebuilt automatically when the data or model changes |
| Keyword search | `BM25Retriever` |
| Hybrid search | Custom `HybridFundRetriever(BaseRetriever)`: reciprocal rank fusion of FAISS and BM25 across several queries, with metadata filters |
| Prompts and chains | `PromptTemplate` with LCEL (`prompt \| llm \| StrOutputParser()`), `RunnableParallel`, `RunnableLambda`, `.with_fallbacks()`, `.batch()` |
| LLM | `HuggingFacePipeline` wrapping Phi-2 + your LoRA adapter. Or any provider through `init_chat_model` (`MF_LLM_BACKEND=chat`), with no code changes |

### Bugs fixed from the notebook version

- **Risk filter never matched.** `risk_level` is stored as 0.0–0.5 but was compared to "Low"/"Medium"/"High". The six values match the SEBI riskometer (Low … Very High), so Low = 0.0–0.1, Medium = 0.2–0.3 and High = 0.4–0.5. The API now returns the label (for example "Moderately High").
- **Similarity was scored backwards.** The FAISS L2 *distance* was added to the score, so less similar funds ranked higher. The new version uses cosine similarity.
- **Distances were attached to the wrong funds** after keyword merging and filtering. Scores now travel with each document.
- **The keyword filter almost never matched,** because the whole query had to be a substring of a fund name. It is replaced by BM25.
- **The embedded text was only "name – category".** Each fund's text now includes a plain-language category profile (purpose, holding period, lock-in), so "park my emergency cash" finds liquid funds.
- **NAV no longer counts as a quality signal.** It was 20% of the score, but a NAV of 900 vs 15 only reflects launch date and unit price.
- **Min-max scaling within the candidates was fragile:** outliers squashed every other score, a single candidate divided by zero, and one version divided by the max only. Funds are now scored against their **category peers** ("5-year return beats 77% of ELSS funds").
- **Impossible requests (e.g. Small Cap + Low risk) were silently ignored.** They now return the closest funds plus a `notice` explaining why.
- **One endpoint used Phi-2 to make search embeddings** (`model.encode`). Retrieval now always uses its own embedding model.
- **Explanations:** generation is batched, `return_full_text=False` replaces the `output.replace(prompt, "")` hack, and every explanation is checked: if the model cites a percentage that isn't in the fund's data, repeats itself or fails, a grounded template is used instead. `explanation_source` tells you which one you got.
- **Live NAV:** the old code fetched all 625 NAVs at startup and stored failures as 0.0 (25 funds). Now only the recommended funds are fetched, in parallel, cached for 6 hours, and the dataset value is used as a fallback.
- **Duplicate scheme names** no longer appear twice in results.
- **CORS** no longer combines `*` origins with credentials, which browsers reject.
- **Training** (`scripts/train_lora.py`):
  - Loss is computed on the answer only.
  - The EOS token is now learned. The old collator masked it, so the model never learned to stop.
  - Padding is dynamic.
  - Targets come from each fund's real data, not one fixed sentence.
  - Evaluation uses held-out funds, instead of BLEU on the training data.
  - The prompt format is shared with inference.

### Other improvements

- **"Any category" mode:** leave the category empty and the goal picks it (for example, "save tax" → ELSS).
- The **time horizon** is read from the goal ("6 months", "10 years", "retirement"). It weights 1Y/3Y/5Y returns and down-ranks unsuitable categories.
- **Score breakdown** for every fund (goal match, returns, cost, horizon fit, risk fit).
- Code moved from notebook cells into a tested Python package (71 backend tests, 3 frontend tests). A CLI was added and the Colab notebook is now a thin runner.
- Frontend:
  - Backend URL comes from `.env`.
  - Categories load from the API.
  - The ngrok browser warning is bypassed.
  - Server errors and notices are shown.
  - `FundCard` is reused instead of duplicated.
  - The meaningless one-bar NAV chart is removed.

## Project layout

```
backend/
  mf_recommender/     config, data, categories, embeddings, vectorstore, retrieval,
                      ranking, nav, prompts, chains, explanations, llm, service, api, cli
  scripts/train_lora.py
  tests/              pytest suite (offline: no model downloads needed)
  requirements*.txt   .env.example   pytest.ini
frontend/             React app (src/App.js, src/FundCard.js, ...)
data/                 updated_data_with_nav.csv (used), data_with_sub_categories.csv (raw)
notebooks/            MutualFundsRecommendation_LangChain.ipynb (Colab runner)
notebooks/legacy/     the original notebook, for reference
```

`updated_data_with_embeddings.csv` is gone; the FAISS index is built from the data on first run and saved in `artifacts/` (or `MF_INDEX_DIR`).

## Quick start

### Google Colab (GPU, with your fine-tuned Phi-2)

Upload the unzipped folder anywhere in My Drive (for example `My Drive/Colab_Notebooks/`), open `notebooks/MutualFundsRecommendation_LangChain.ipynb` and click *Run all*. There are no paths to edit: the notebook finds the project, its data and any Phi-2 LoRA adapter in your Drive by itself. It then installs the requirements, builds the index (saved in the project's `artifacts/` folder), runs a sample query and serves the API through ngrok.

### Locally

```bash
cd backend
pip install -r requirements-dev.txt            # install torch first if you don't have it

# Fast, no GPU: template explanations only
MF_LLM_BACKEND=none uvicorn mf_recommender.api:app --port 8000

# With Phi-2 (+ the newest LoRA adapter in artifacts/, found automatically)
uvicorn mf_recommender.api:app --port 8000

# Command line
python -m mf_recommender.cli "Grow my savings over 7 years" --risk High --llm none
python -m pytest -q
```

```bash
cd frontend
cp .env.example .env        # set REACT_APP_API_URL (localhost or your ngrok URL)
npm install
npm start
```

## Configuration

All settings are environment variables (see `backend/.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `MF_LLM_BACKEND` | `hf` | `hf` (local model), `chat` (any LangChain chat model), `none` (templates only) |
| `MF_BASE_MODEL` | `microsoft/phi-2` | Base model for `hf` |
| `MF_ADAPTER_PATH` | auto | LoRA adapter folder. If unset, the newest adapter in `artifacts/` is used |
| `MF_CHAT_MODEL` | – | For `chat`, e.g. `openai:<model>`. Needs `pip install langchain` plus the provider package |
| `MF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Or `hashing` for offline, keyword-only embeddings |
| `MF_DATA_PATH` / `MF_INDEX_DIR` | `data/…csv` / `artifacts/faiss_index` | Dataset and saved index |
| `MF_REWRITE_QUERY` | `true` | LLM query rewrite, fused with the original goal |
| `MF_LIVE_NAV` | `true` | Fetch today's NAV from api.mfapi.in |
| `MF_WEIGHT_RELEVANCE` / `_RETURNS` / `_COST` | 0.4 / 0.4 / 0.2 | Ranking weights |
| `MF_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## API

`POST /recommend`

```json
{ "goal": "Save tax and grow my money over 5 years", "category": null, "risk": "High", "top_k": 5, "explain": true }
```

`category` may be `null`, `""` or `"Any"`, and `risk` is case-insensitive. Response (abridged):

```json
{
  "funds": [{
    "scheme_code": 101742, "scheme_name": "DSP Tax Saver Fund", "category": "ELSS Mutual Funds",
    "NAV": 19.4774, "nav_date": null, "nav_source": "dataset",
    "returns_1yr": 1.8, "returns_3yr": 28.8, "returns_5yr": 13.3, "expense_ratio": 0.81,
    "risk_level": "Very High", "risk_score": 0.5, "score": 0.8219,
    "score_breakdown": { "relevance": 1.0, "returns": 0.659, "cost": 0.791, "horizon_fit": 1.0, "risk_fit": 1.0 },
    "explanation": "DSP Tax Saver Fund is an ELSS fund (equity) rated Very High risk on the riskometer. Its 5-year return of 13.3% beats 77% of the 43 other funds in its category, ... Note the 3-year lock-in.",
    "explanation_source": "template"
  }],
  "query": { "original": "Save tax and grow my money over 5 years", "rewritten": null, "horizon_years": 5.0 },
  "notice": null,
  "filters": { "category": null, "risk": "High", "risk_band": [4, 5] }
}
```

`GET /categories` → `{ "categories": [...], "risk_levels": ["Low", "Medium", "High"] }`  `GET /health` → status, fund count, active LLM backend.

## Retraining the adapter

```bash
cd backend
pip install -r requirements-train.txt
python scripts/train_lora.py --gradient-checkpointing   # saves to artifacts/fine_tuned_phi2_v2, used automatically
```

The script prints the held-out eval loss and how many generations pass the grounding check. Your existing adapter still works, but it was trained on a slightly different prompt layout, so retraining should give cleaner, better-stopping explanations.

## Notes and limitations

- Returns and expense ratios are a snapshot in the CSV; only NAV is live.
- The Low/Medium/High → riskometer mapping is inferred from the data (overnight funds are all 0.0, equity funds all 0.5).
- `hashing` embeddings match keywords only. They're for tests and offline demos; use a sentence-transformer for real semantic search.
- Recommendations are based on past performance and are not investment advice.
