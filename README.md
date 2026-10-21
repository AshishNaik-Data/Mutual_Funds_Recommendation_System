# Mutual Funds Recommendation System

An AI-powered recommendation engine for Indian mutual funds. You describe your goal in plain English, and the system finds and ranks suitable funds with retrieval-augmented generation (RAG), now built on **LangChain**. Each pick comes with an explanation grounded in the fund's real numbers.

```
goal ──► LLM query rewrite (Phi-3-mini, optional)
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
   Live NAV (mfapi.in, cached) ─► Explanations (Phi-3-mini + LoRA via LCEL,
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
| LLM | `HuggingFacePipeline` wrapping Phi-3-mini + its LoRA adapter, with prompts wrapped in the model's chat template. Or any provider through `init_chat_model` (`MF_LLM_BACKEND=chat`), with no code changes |

### The language model: Phi-3-mini

The LLM is **Microsoft Phi-3-mini** (`microsoft/Phi-3-mini-4k-instruct`, 3.8B parameters, about 7.6 GB in 16-bit), replacing the original Phi-2. It is instruction-tuned, so it should follow "explain using only these facts" much better than a base model, and it still fits a free Colab T4.

- **Chat format.** Prompts are sent as a user turn in Phi-3's chat template (`<|user|> … <|end|> <|assistant|>`), and training examples use exactly the same format.
- **LoRA layers chosen per architecture.** Phi-3 fuses q, k and v into `qkv_proj`; the training script picks `qkv_proj` and `o_proj` automatically (Phi-2 would get `q_proj`, `k_proj`, `v_proj`, `dense`).
- **Adapters are matched to their base model.** An adapter is only loaded onto the model it was trained on; a mismatched one is skipped with a warning instead of crashing.
- **Swappable.** Set `MF_BASE_MODEL` to another Hugging Face causal LM and retrain; nothing else changes.

### Bugs fixed from the notebook version

- **Risk filter never matched.** `risk_level` is stored as 0.0–0.5 but was compared to "Low"/"Medium"/"High". The six values match the SEBI riskometer (Low … Very High), so Low = 0.0–0.1, Medium = 0.2–0.3 and High = 0.4–0.5. The API now returns the label (for example "Moderately High").
- **Similarity was scored backwards.** The FAISS L2 *distance* was added to the score, so less similar funds ranked higher. The new version uses cosine similarity.
- **Distances were attached to the wrong funds** after keyword merging and filtering. Scores now travel with each document.
- **The keyword filter almost never matched,** because the whole query had to be a substring of a fund name. It is replaced by BM25.
- **The embedded text was only "name – category".** Each fund's text now includes a plain-language category profile (purpose, holding period, lock-in), so "park my emergency cash" finds liquid funds.
- **NAV no longer counts as a quality signal.** It was 20% of the score, but a NAV of 900 vs 15 only reflects launch date and unit price.
- **Min-max scaling within the candidates was fragile:** outliers squashed every other score, a single candidate divided by zero, and one version divided by the max only. Funds are now scored against their **category peers** ("5-year return beats 77% of ELSS funds").
- **Impossible requests (e.g. Small Cap + Low risk) were silently ignored.** They now return the closest funds plus a `notice` explaining why.
- **One endpoint used the language model (Phi-2 at the time) to make search embeddings** (`model.encode`). Retrieval now always uses its own embedding model.
- **Explanations:** generation is batched, `return_full_text=False` replaces the `output.replace(prompt, "")` hack, and every explanation passes a quality gate before it is shown (see below). If it fails, a grounded template built from the fund's numbers is used instead, and `explanation_source` (`llm` or `template`) tells you which one you got.
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
- Code moved from notebook cells into a tested Python package (93 backend tests, 4 frontend tests). A CLI was added and the Colab notebook is now a thin runner.
- Frontend:
  - Backend URL comes from `.env`.
  - Categories load from the API.
  - The ngrok browser warning is bypassed.
  - Server errors and notices are shown.
  - `FundCard` is reused instead of duplicated.
  - The meaningless one-bar NAV chart is removed.
  - Each explanation is tagged "AI-written" or "From fund data".
  - The footer shows which model is answering, e.g. "Model: Phi-3-mini-4k-instruct + LoRA fine-tune" or "(not fine-tuned)".

## Fixes after the first Phi-3 test run (v3)

The first end-to-end run (goal "I want a short term investment", Banking and PSU, High risk) worked, but showed these problems:

| Seen in the result | Cause | Fix |
|---|---|---|
| All five explanations were the same generic paragraph with no numbers | Fine-tuning had crashed (see Troubleshooting: torchao), and the notebook went on to serve untrained Phi-3 | The notebook now stops if training saves no adapter; `/health` and the UI footer show whether the fine-tuned model is serving |
| "Low expense ratio compared to peers" for funds cheaper than only 13–33% of their category | The quality gate only caught invented *numbers*, not false *claims* | The gate now also requires at least one real figure and rejects cost or return claims the peer comparison contradicts ("low cost" when cheaper than fewer than half of peers; "strong returns" when below the median in every period) |
| "An individual seeking moderate risk" when the user picked High | The prompt said the user wanted the fund's own risk level | The prompt now gives the user's risk choice and whether the fund matches it; the template adds "That is lower than the High risk you asked for." |
| Kotak's NAV shown as "live, 2013-08-13" | Some scheme codes in the dataset belong to plans that stopped publishing NAVs | A NAV more than 10 days old is returned as `"nav_source": "stale"` and shown as "last published <date>; this plan may be closed or merged" |
| Query rewritten as "Short-term growth equity mutual funds" | No guidance on matching asset class to horizon | The rewrite prompt now has examples steering horizons under 3 years to debt, liquid or money market funds |

Because the explanation prompt changed, an adapter trained before v3 should be retrained once (`RETRAIN = True` in the notebook).

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

### Google Colab (T4 GPU, Phi-3-mini)

Upload the unzipped folder anywhere in My Drive (for example `My Drive/Colab_Notebooks/`), open `notebooks/MutualFundsRecommendation_LangChain.ipynb` and click *Run all*. There are no paths to edit: the notebook finds the project, its data and any Phi-3-mini LoRA adapter in your Drive by itself. It installs the requirements, runs the tests and builds the index. On the first run, if no adapter exists yet, it fine-tunes one (about 1–1.5 hours on a T4); later runs reuse it. It then loads Phi-3-mini, runs a sample query and serves the API through ngrok. Everything it creates is saved in the project's `artifacts/` folder.

Before the first run, add your ngrok token in Colab's **Secrets** panel (key icon) as `NGROK_AUTHTOKEN`. Never paste it into a cell.

The fine-tuning cell has one setting:

| `RETRAIN` | What happens |
|---|---|
| `'auto'` (default) | Trains only if no adapter for Phi-3-mini exists yet |
| `True` | Always retrains. Use it once after changing the prompt, the training data or the base model, then set it back to `'auto'`, or every *Run all* will retrain |
| `False` | Never trains |

If training fails, the notebook stops with "Fine-tuning did not save an adapter" instead of quietly serving the untrained model; scroll up to the training output for the real error.

**Check that the fine-tuned model is serving.** The "Try it" cell prints `fine_tuned: True`; `GET /health` returns `"fine_tuned": true`; the page footer says "+ LoRA fine-tune"; and most cards say "AI-written". If cards say "From fund data", the model's answer failed the quality gate, and the Colab log says why.

**Run the frontend on your computer** once the notebook prints `REACT_APP_API_URL = https://….ngrok-free.dev`:

```bash
cd frontend
echo "REACT_APP_API_URL=https://<your-url>.ngrok-free.dev" > .env
npm install        # first time only
npm start          # opens http://localhost:3000
```

Free ngrok accounts get one fixed address, so the URL usually stays the same; if the notebook prints a different one, update `.env` and restart `npm start`. To stop: Control+C in the terminal, then set `STOP = True` in the notebook's last cell (or Runtime → Disconnect and delete runtime).

### What is saved, and what isn't

| Item | Where | Created |
|---|---|---|
| LoRA adapter (~40 MB: `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, `training_metrics.json`) | `artifacts/lora-phi-3-mini-4k-instruct/` in Drive | Once, by fine-tuning; reused on every later run |
| FAISS index | `artifacts/faiss_index/` in Drive | Once; rebuilt automatically when the data or embedding model changes |
| Phi-3-mini itself (~7.6 GB) | Colab's temporary disk | Downloaded from Hugging Face in each new session (a few minutes, not retraining) |

LoRA trains only about 9 million extra weights, against Phi-3-mini's 3.8 billion, so only those are saved. At startup they are merged into the base model in memory. After training, check the adapter files appear in Drive before deleting the runtime, since Drive can take a moment to sync.

### Locally

```bash
cd backend
pip install -r requirements-dev.txt            # install torch first if you don't have it

# Fast, no GPU: template explanations only
MF_LLM_BACKEND=none uvicorn mf_recommender.api:app --port 8000

# With Phi-3-mini (needs a GPU with ~8 GB free; the matching adapter in artifacts/ is found automatically)
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
| `MF_BASE_MODEL` | `microsoft/Phi-3-mini-4k-instruct` | Base model for `hf` (any Hugging Face causal LM) |
| `MF_ADAPTER_PATH` | auto | LoRA adapter folder. If unset, the newest adapter in `artifacts/` trained on `MF_BASE_MODEL` is used |
| `MF_CHAT_MODEL` | – | For `chat`, e.g. `openai:<model>`. Needs `pip install langchain` plus the provider package |
| `MF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Or `hashing` for offline, keyword-only embeddings |
| `MF_DATA_PATH` / `MF_INDEX_DIR` | `data/…csv` / `artifacts/faiss_index` | Dataset and saved index |
| `MF_REWRITE_QUERY` | `true` | LLM query rewrite, fused with the original goal |
| `MF_LIVE_NAV` | `true` | Fetch today's NAV from api.mfapi.in |
| `MF_NAV_STALE_DAYS` | `10` | A fetched NAV older than this is labelled stale (plan probably closed or merged) |
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

`GET /categories` → `{ "categories": [...], "risk_levels": ["Low", "Medium", "High"] }`  `GET /health` → status, fund count, and what writes the explanations: `llm_backend`, `base_model`, `adapter`, `fine_tuned`. If `fine_tuned` is `false` with the `hf` backend, Phi-3-mini is running without its LoRA adapter (the UI footer says "not fine-tuned"), and most explanations will fall back to the template.

## Fine-tuning Phi-3-mini (LoRA)

```bash
cd backend
pip install -r requirements-train.txt
python scripts/train_lora.py      # saves to artifacts/lora-phi-3-mini-4k-instruct, used automatically
```

The notebook runs this for you on the first run. It takes about 1–1.5 hours on a T4.

| Setting | Default |
|---|---|
| Base model | `MF_BASE_MODEL`, i.e. `microsoft/Phi-3-mini-4k-instruct` |
| Examples | 3 per fund (~1,875), 10% of funds held out for evaluation |
| Format | Phi-3 chat template; loss only on the assistant's answer and its `<\|end\|>` token |
| LoRA | r 16, alpha 32, dropout 0.05 on `qkv_proj`, `o_proj` (`--target-modules all-linear` to train every layer) |
| Batch | 2 per step × 8 accumulation steps = 16, gradient checkpointing on |
| Schedule | 2 epochs, learning rate 2e-4, cosine with warm-up |
| Precision | float16 on a T4; bfloat16 only on GPUs with native support (compute capability 8+, e.g. A100, L4). The LoRA weights themselves are kept in float32 |

It prints the held-out eval loss and how many held-out explanations pass the same fact check the API uses, and writes both to `training_metrics.json` beside the adapter. Training checkpoints are deleted at the end to save Drive space.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ImportError: Found an incompatible version of torchao` during training | Colab ships `torchao` 0.10, and recent `peft` refuses to set up LoRA while an old version is installed. This project doesn't use `torchao` (it is for quantization), so remove it: `!pip uninstall -y torchao`. The notebook's install cell now does this. Don't upgrade it instead: newer versions may need a newer PyTorch than Colab has |
| Training on a T4 shows about 60+ s per step (`68.63s/it`, ~4 hours) | The script picked bfloat16, which a T4 only emulates. Fixed: the script now uses float16 unless the GPU supports bfloat16 natively, and prints `Precision: float16 on Tesla T4` at the start |
| `RuntimeError: Fine-tuning did not save an adapter` | Training failed; the real error is in the training output above it. Out of GPU memory: Runtime → Restart session, then *Run all* |
| Explanations all generic, footer says "not fine-tuned" | No adapter was loaded. Retrain with `RETRAIN = True`, and check `artifacts/lora-phi-3-mini-4k-instruct/` exists in Drive |
| `FileNotFoundError` for the CSV, or the data shows as a Google Sheet | Drive converted the CSV on upload. The notebook detects this and asks you to upload the CSV again; turn off *Settings → Convert uploads* in Drive to avoid it |
| `ModuleNotFoundError: mf_recommender` | The notebook couldn't find the project folder. Check the whole unzipped folder (with `backend/`) is in My Drive, then re-run the first cell |
| `ERR_NGROK_3200` or the page says it can't reach the backend | The Colab server isn't running (session ended or the STOP cell ran). Re-run the server cell and put the new URL in `frontend/.env` |
| NAV shows "last published …; this plan may be closed or merged" | Not a bug: the dataset's scheme code points to a plan that stopped publishing NAVs. Returns and expense ratio still come from the dataset |

## Notes and limitations

- Returns and expense ratios are a snapshot in the CSV; only NAV is live.
- Some scheme codes in the dataset belong to older plans (dividend options, closed or merged plans), so their live NAV can be old or differ from the plan you would buy today.
- The Low/Medium/High → riskometer mapping is inferred from the data (overnight funds are all 0.0, equity funds all 0.5).
- `hashing` embeddings match keywords only. They're for tests and offline demos; use a sentence-transformer for real semantic search.
- Recommendations are based on past performance and are not investment advice.
