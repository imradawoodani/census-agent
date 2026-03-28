# US Census Data Agent

A production-quality chat agent that answers natural language questions about the US population using the 2020 SafeGraph Open Census dataset (ACS block-group level). Built on Snowflake, DigitalOcean Inference, and Cohere embeddings.

**Live demo:** [https://your-app.ondigitalocean.app](https://census-agent-o5o2c.ondigitalocean.app/)
(not live right now)

---

## Evaluating the Demo

No login required. Open the URL and try these questions:

| Category | Example question |
|---|---|
| Population | What is the population of California? |
| Income | Which 5 states have the highest median household income? |
| Race/ethnicity | What percentage of Florida residents are Hispanic? |
| Education | What percentage of California adults have a bachelor's degree? |
| Housing | What is the homeownership rate in Ohio? |
| Veterans | How many veterans live in Texas? |
| Employment | What is the unemployment rate in Michigan? |
| Qualitative | Why is poverty higher in rural areas? |
| Out-of-scope | What is the stock price of Apple? (should be rejected) |
| Greeting | Hi, how are you? (should respond conversationally) |

Multi-turn context is preserved within a session — try asking "What about Texas?" after any state-level question.

---

## Architecture

```
Browser (SSE)
    │
    ▼
FastAPI (app/main.py)
    │
    ▼
CensusAgent (app/agent.py) — pipeline orchestrator
    │
    ├── Guardrails (app/guardrails.py)
    │     Fast pre-LLM checks: injection detection, greeting routing
    │
    ├── Intent classifier (app/llm_client.py)
    │     Llama 3.1 8B via DO inference
    │     → QUANTITATIVE / QUALITATIVE / AMBIGUOUS / OFF_TOPIC / GREETING
    │
    ├── Schema retriever (app/schema_retriever.py)
    │     Cohere embeddings over the table descriptions
    │     Returns top-8 most relevant tables per question
    │
    ├── Field retriever (app/field_retriever.py)
    │     Cohere embeddings over the Census field descriptions
    │     from 2020_METADATA_CBG_FIELD_DESCRIPTIONS
    │     Returns top-15 most relevant columns in plain English
    │     Disk-cached after first build to preserve API cost
    │
    ├── Few-shot retriever (app/few_shot.py)
    │     15 verified question to SQL pairs embedded at startup
    │     Returns top-3 most similar examples per question
    │
    ├── SQL generator (app/llm_client.py)
    │     Llama 3.3 70B via DO inference, temperature=0
    │     Prompt = schema + field context + few-shot examples + history
    │
    ├── SQL validator (app/sql_validator.py)
    │     sqlglot AST validation
    │     Rejects non-SELECT statements structurally
    │
    ├── Snowflake executor (app/snowflake_client.py)
    │     Async wrapper with 55s timeout, 500-row cap
    │     Smart retry: classifies retryable vs fatal errors
    │
    ├── Plausibility check (app/llm_client.py)
    │     Llama 3.1 8B sanity-checks results before synthesis
    │     Catches outrageous numbers
    │
    └── Answer synthesizer (app/llm_client.py)
          Llama 3.3 70B streams tokens via SSE
          Converts FIPS codes to state names and formats numbers
```

### Key design decisions

**FIPS-based filtering instead of geometry joins.** The first 2 digits of `CENSUS_BLOCK_GROUP` are the state FIPS code. Filtering with `SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '06'` is faster and simpler than joining `2020_CBG_GEOMETRY_WKT`. Learned this from examining how Snowflake Cortex queries the same dataset.

**Dynamic field retrieval over hardcoded column mappings.** At startup we embed all field descriptions from `2020_METADATA_CBG_FIELD_DESCRIPTIONS`. For each question, the top-15 most semantically relevant fields are retrieved and injected into the SQL generation prompt. This means the LLM is told "B19013e1 = Median household income" rather than guessing. No column codes are hardcoded anywhere.

**Single inference provider.** All LLM calls (classification, SQL generation, synthesis, plausibility check) go through DigitalOcean's serverless inference endpoint. One API key, one base URL, OpenAI SDK. Fast model (Llama 3.1 8B) for cheap tasks and smart model (Llama 3.3 70B) for quality tasks.

**Streaming via SSE.** The `/api/chat/stream` endpoint sends status events (`Classifying...`, `Querying database...`) followed by answer tokens as they stream from the LLM. The browser receives words as they're generated.

**Two intent paths.** Not every question needs SQL. `QUALITATIVE`, `AMBIGUOUS`, and `GREETING` intents route directly to a conversational LLM response. Only `QUANTITATIVE` goes through the full text-to-SQL pipeline. This prevents the system from returning gibberish for "why is poverty high in Mississippi?" or "hi how are you".

---

## Data

The [US Open Census Data](https://app.snowflake.com/marketplace/listing/GZT0ZEQPJ0/safegraph-us-open-census-data) accessed via Snowflake Marketplace.

Key tables used:
- `2020_CBG_B01` — Population and age
- `2020_CBG_B02` — Race categories
- `2020_CBG_B03` — Hispanic/Latino
- `2020_CBG_B05` — Place of birth
- `2020_CBG_B15` — Educational attainment
- `2020_CBG_B17` — Poverty
- `2020_CBG_B19` — Household income
- `2020_CBG_B21` — Veteran status
- `2020_CBG_B23` — Employment
- `2020_CBG_B25` — Housing tenure
- `2020_METADATA_CBG_FIELD_DESCRIPTIONS` — Plain-English column definitions

---

## Running Locally

**Requirements:** Python 3.11+, Snowflake trial account with US Open Census Data installed, DigitalOcean account, Cohere API key.

```bash
git clone https://github.com/imradawoodani/census-agent.git
cd census-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your credentials (see .env.example)
uvicorn app.main:app --reload
```

Open http://localhost:8000. Watch startup logs for `agent_ready`. This confirms Snowflake connected and embeddings were built. The first startup builds the field embedding cache (about a minute) and subsequent startups load from disk instantly.

**Running tests:**
```bash
pytest tests/ -v           # unit tests (no credentials needed)
pytest tests/ --live       # includes live Snowflake + LLM tests
```

---

## Deploying to DigitalOcean

1. **Edit `.do/app.yaml`** to rename with your respective GitHub repository and push
3. DO Dashboard → Apps → Create App → connect your repo
4. Add env vars as encrypted secrets (see `.env.example` for the full list)
5. Deploy and URL appears in about 4 minutes
