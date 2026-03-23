# Reflection

## Development Process and Key Architectural Decisions

### The core problem

A user types plain English. The answer is in a Snowflake table. The gap between those two things is text-to-SQL, and text-to-SQL fails in ways that are hard to detect. For example, bad SQL triggers an error (catchable), but semantically wrong SQL returns a plausible-looking number (not catchable without ground truth).

Every significant decision in this codebase is a response to that failure mode.

### Why not RAG?

The Census data is structured tabular data, not abstract, text-heavy documents. RAG retrieves approximate passages whileSQL retrieves exact numbers. "What is the poverty rate in Mississippi?" has one correct answer. A RAG system might return a passage mentioning Mississippi's poverty, but it would not compute the SUM of the appropriate columns. Retrieval still matters, but at the schema and field level, not the data level.

### Dynamic field retrieval instead of hardcoded column mappings

The Census dataset uses opaque column codes (`B19013e1`, `B03002e12`) that the LLM has no prior knowledge of. When beginning deployment, I embed all field descriptions from `2020_METADATA_CBG_FIELD_DESCRIPTIONS`, which maps each code to its plain-English meaning. For any question then, the top-15 most semantically relevant fields are retrieved and injected into the SQL generation prompt. The LLM is told the exact verified column names from the official Census metadata rather than guessing. It means the system handles topics we never explicitly anticipated, because the metadata knows about them even if we didn't write an example.

### FIPS-based state filtering

I was playing around with Snowflake Cortex queries the same dataset and picked the idea to use the first 2 digits of `CENSUS_BLOCK_GROUP` (same as the state FIPS code). Filtering with `SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '06'` is both faster and simpler than joining `2020_CBG_GEOMETRY_WKT`. This single change eliminated a category of join errors and improved query performance.

### Intent routing before SQL

Not every question needs SQL. "Why is poverty high in Mississippi?" cannot be answered by a database. "Hi, how are you?" should get a friendly response, not a rejection. A single-path system that tries to generate SQL for everything produces nonsense for qualitative questions and frustrating responses for greetings.

The solution is an intent classifier that routes before any expensive operations. Five intents: `QUANTITATIVE` (SQL path), `QUALITATIVE` (LLM knowledge path), `AMBIGUOUS` (clarification request), `OFF_TOPIC` (polite redirect), `GREETING` (conversational response). The guardrail layer handles obvious greetings and injections without even reaching the LLM.

### Two-tier model routing

DigitalOcean's serverless inference provides access to all models through a single endpoint and API key (Llama 3.1 8B and Llama 3.3 70B). Simple tasks (intent classification, plausibility checking) use 8B for cheap, fast, single-token or short responses. Quality tasks (SQL generation, answer synthesis) use 70B for more capable on structured reasoning and natural language fluency.

### sqlglot over regex for SQL validation

regex gave many false positives like `SELECT 'DROP TABLE is fine' AS note FROM t` would incorrectly fail. sqlglot parses SQL into an Abstract Syntax Tree and inspects node types. `DROP` in a string literal is a leaf node of type `Literal`, not a statement of type `Drop`.

### Plausibility checking

The retry loop catches SQL that Snowflake rejects. It does not catch SQL that Snowflake accepts but is semantically wrong. A plausibility check runs after every successful query and asks, is this result plausible for this question? "Louisiana median household income: $338,831" fails immediately. This is not foolproof since it catches extreme failures but not subtle ones, but it prevents the worst class of confident wrong answers from reaching users.

---

## What I Would Improve With More Time

**Metadata-lookup step before SQL generation.** Snowflake Cortex queries `METADATA_CBG_FIELD_DESCRIPTIONS` at query time to find the exact columns, then generates SQL using those verified names. We do this at startup (embedding all fields) and at retrieval time (semantic search). A hybrid approach that runs a targeted metadata query at question time to confirm the top candidates would increase precision, at the cost of one extra Snowflake round trip per query.

**Eval harness against live data.** The test suite validates the structure of generated SQL (correct table, correct aggregation, correct FIPS code). It does not validate the numeric accuracy of results. A proper eval harness would store ~50 canonical questions with verified ground-truth answers, run them against Snowflake, and compare. This is the only reliable way to catch semantic regressions when prompts change.

**Redis for persistent sessions.** The current session store is in-memory and resets on server restart. Redis is wired in and falls back gracefully — setting `REDIS_URL` enables it but it hasn't been fully tested due to cost constraints. For production it's required, especially when Digital Ocean may restart instances.

**User feedback loop.** The thumbs up/down buttons in the UI log nothing currently. In production, negative feedback is the most valuable signal for finding systematic failures. It questions where the plausibility check didn't fire but the answer was wrong.

---

## Edge Cases Identified But Not Fully Addressed

**Semantically wrong but structurally valid SQL.** If the LLM uses `B19001e1` (total households by income bracket) instead of `B19013e1` (median household income estimate), Snowflake returns a large number, the plausibility check sees a number in range, and the user gets a confidently wrong answer, often with caution stating something might be wrong but there's not retry. The fix is verified ground truth in the eval harness, not a runtime check.

**Qualitative questions are not data-grounded.** "Why is poverty higher in rural areas?" gets a response from the LLM's training knowledge, not from the Census data. This is explicitly flagged in the response ("based on general knowledge"), and the system suggests a follow-up quantitative question. The ideal implementation routes qualitative questions to a RAG pipeline over Census Bureau methodology documentation. This wasn't built because it requires ingesting and indexing that documentation, and because misclassifying a quantitative question to that path produces a worse answer than CANNOT_ANSWER.

**MENA/Middle Eastern is not a Census category.** The 2020 ACS doesn't track Middle Eastern or MENA as a separate race category. People of Middle Eastern origin typically self-identified as "White alone" or "Some other race." This is handled by the qualitative path with an explicit explanation, but users often don't know this limitation going in. A UI-level note about what race categories are available would help.

**Multi-column aggregate questions.** "What is the full racial and income breakdown of Texas?" would require queries across multiple tables and some coordination in synthesis. The current system would generate SQL for one table or fail. The intent classifier might route this as AMBIGUOUS. A better handling would be to decompose multi-table questions into sequential queries and merge the results.

**Cold Snowflake warehouse.** The trial warehouse auto-suspends after inactivity. The first query after suspension takes 10-20 seconds just to wake. The 60-second response requirement is met for all subsequent queries; first-query latency on a cold warehouse can exceed it.

---

### Prominent Edge Case: Qualitative vs. Quantitative Intent Ambiguity

A subtle failure mode arises when a question is *linguistically qualitative* but *semantically quantitative*.

**Example:**

> “Does anyone in Arizona have a bachelor’s degree?”

This query is phrased as a yes/no question and was classified as `QUALITATIVE`. As a result, the system returns a correct but low-information response without querying the database.

However, the underlying user intent is quantitative and could extend to:

> “How many people in Arizona have a bachelor’s degree?”


The intent classifier prioritizes:

* surface form of the question (binary vs. numeric)
* fast routing without unnecessary SQL generation

Because the question can be answered truthfully without data access, it is routed away from the SQL pipeline.


This reflects a deliberate tradeoff:

* **Pros:** avoids unnecessary SQL generation, reduces latency, and simplifies handling of clearly qualitative queries
* **Cons:** returns *correct but uninformative answers* when a richer, data-backed response is possible + also risks outdated / wrong answers

This class of errors is particularly important because it produces **plausible outputs that are not maximally useful**, making them harder to detect than outright failures.

#### Proposed Improvement

A more robust approach would detect *latent quantitative intent* and rewrite such queries before routing.

For example:

* “Does anyone in Arizona have a bachelor’s degree?”
  → rewritten to:
* “How many people in Arizona have a bachelor’s degree?”

This would allow the system to:

* preserve natural language flexibility
* default to data-grounded answers when available

Or we could also fix this with a better understanding of our dataset and schema fed to the model and decoupling the process into one more multiagentic that allows routing between qualitative (RAG) data fetching and quantitative data fetching as required by the question.

While testing, I observed cases where the agent failed to provide answers, particularly for queries requiring more granular or relational data like comparing poverty rates between counties. For example, asking about the poverty rate in Dallas County versus Culberson County consistently returned a failure message. This highlights a current limitation: the agent can retrieve state-level or high-level summary data but struggles with fine-grained or multi-step queries. Given more time, I would enhance the system with better query decomposition, more comprehensive data coverage, and fallback strategies for partially answerable queries. Some queries returned no data at all, even though the question seemed straightforward. For instance, asking about the poverty rate in Mississippi resulted in a response indicating no matching data. This reveals a limitation in my current approach. The agent relies on preprocessed and embedded datasets much more than it should.

---

## Testing Strategy and Tradeoffs

### What the tests cover

**Guardrails** (12 tests): injection detection, greeting routing, census keyword recognition, edge cases (empty input, whitespace).

**SQL validation** (9 tests): valid SELECT, each forbidden statement type (DROP, DELETE, INSERT, CREATE, UPDATE), AST correctness (string literal with 'DROP' is allowed), LIMIT injection.

**Error classification** (5 tests): retryable vs fatal errors, user-facing error messages.

**Session management** (6 tests): append and retrieve, session isolation, empty session, history formatting, window capping, health check.

**Few-shot retriever** (3 tests): build and retrieve, formatting, graceful empty index.

**Field retriever** (5 tests): build and retrieve, B99 table filtering, disk cache write and reload, formatting, empty graceful handling.

**Intent classification** (6 tests): all five intent types, error fallback.

**SQL generation** (3 tests): successful generation, CANNOT_ANSWER, retry context injected correctly.

**Agent pipeline** (9 tests): injection blocked, greeting routed to qualitative, CANNOT_ANSWER handled, empty results handled, implausible result blocked, qualitative intent routed, timeout handled, SQL validation retry, session turn saved.

**Routes** (3 tests): health endpoint, UI served, chat endpoint.

**Eval harness** (12 tests): structural correctness of canonical SQL for 12 golden question types (correct table, correct column prefix, correct aggregation, correct FIPS code). Two additional live integration tests skipped by default, runnable with `--live`.

**Total: 73 passing unit tests, 2 skipped live tests.**

### Key tradeoff: structure over semantics

All unit tests validate SQL structure, not numeric accuracy. "Does the SQL reference `2020_CBG_B19` and use `AVG`?" Yes, but "does it return the correct median income for Texas?", only the live test knows. This was a deliberate choice to have fast, deterministic, and credential-free unit tests. Semantic validation requires live data and is covered by the eval harness's `--live` flag. It would also require gold standardand more API credits than I had access to.

### Tests I would add given more time

1. A regression suite of 50+ questions with verified numeric ground truth, runnable against a read-only Snowflake connection. The 12 structural eval cases in `test_eval.py` are the skeleton of this. With live Snowflake access they would compare actual query results against stored expected values. This is the test layer that catches things like "prompt change made SQL generation regress for poverty queries", something the static unit tests cannot detect.

2. Mutation testing on the guardrails and SQL validator to confirm that removing each check actually breaks a test.

3. Property-based testing for SQL generation (via Hypothesis). Generate=ing random question variants and assert the output is always a valid SELECT or CANNOT_ANSWER, never a mutation statement.
