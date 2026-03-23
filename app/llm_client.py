"""
LLM client — all inference calls through DigitalOcean serverless endpoint.
One client, one API key, all models (Llama 3.3 70B for quality tasks,
Llama 3.1 8B for fast/cheap tasks).

Intent types:
  QUANTITATIVE  — answerable via SQL (counts, rates, rankings)
  QUALITATIVE   — needs explanation from general knowledge
  AMBIGUOUS     — needs clarification from the user
  OFF_TOPIC     — not related to US Census data
  GREETING      — social exchange
"""
from enum import Enum
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Client setup — single OpenAI-compatible client for all models
# ---------------------------------------------------------------------------

_client = AsyncOpenAI(
    api_key=settings.do_model_access_key,
    base_url=settings.do_inference_base_url,
)


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    QUANTITATIVE = "QUANTITATIVE"
    QUALITATIVE = "QUALITATIVE"
    AMBIGUOUS = "AMBIGUOUS"
    OFF_TOPIC = "OFF_TOPIC"
    GREETING = "GREETING"


_INTENT_SYSTEM = """\
You classify questions directed at a US Census data chatbot.

Respond with EXACTLY one word. No explanation, no punctuation:

QUANTITATIVE — question asks for a specific number, rate, ranking, or count that exists in the US Census dataset
  Examples: population counts, income levels, poverty rates, racial breakdown, education levels, employment rates, housing stats, age demographics, veteran counts

QUALITATIVE — question asks for explanation, context, methodology, or analysis (not a database lookup)
  Examples: "why is poverty high in X?", "how does the Census count X?", "what does X measure?", "what factors explain X?"

AMBIGUOUS — question is unclear, underspecified, or refers to multiple places (e.g., "Springfield")
  Examples: "what about it?", "tell me about Springfield", "compare them"

GREETING — social exchange, small talk, or expression of thanks
  Examples: "hi", "how are you", "thanks", "goodbye"

OFF_TOPIC — clearly unrelated to US population/census data
  Examples: stock prices, recipes, sports scores, weather, coding help
"""


async def classify_intent(message: str, conversation_history: str = "") -> IntentType:
    context = f"\n\n{conversation_history}" if conversation_history else ""
    prompt = f"{context}\n\nQuestion: {message}\n\nClassify:"
    
    try:
        resp = await _client.chat.completions.create(
            model=settings.fast_model,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        for intent in IntentType:
            if intent.value in raw:
                return intent
        return IntentType.QUANTITATIVE  # safe default
    except Exception as e:
        logger.warning("intent_classification_error: %s", str(e))
        return IntentType.QUANTITATIVE


# ---------------------------------------------------------------------------
# SQL generation
# ---------------------------------------------------------------------------

_SQL_SYSTEM = """\
You generate Snowflake SQL for the US Open Census dataset (SafeGraph/US_OPEN_CENSUS).

DATABASE CONVENTIONS:
- All tables are in the PUBLIC schema
- Tables use Census Block Group (CBG) level data
- Quote all column names with double quotes: "B01001e1"
- Tables join on CENSUS_BLOCK_GROUP (a 12-digit FIPS code)
- ALWAYS filter states using: SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '{{state_fips}}'
  where {{state_fips}} is the 2-digit FIPS code from the table below.
  NOT by joining to geometry tables (slower, unnecessary)

STATE FIPS CODES (first 2 digits of CENSUS_BLOCK_GROUP):
{fips_context}

KEY TABLE GROUPS:
- 2020_CBG_B01: Population and age (B01001e1=total pop, B01002e1=median age)
- 2020_CBG_B02: Race categories (B02001e1=total, e2=White, e3=Black, e4=Native American, e5=Asian, e6=Pacific Islander, e7=Other, e8=Two or more)
- 2020_CBG_B03: Hispanic/Latino (B03002e1=total, B03002e12=Hispanic or Latino)
- 2020_CBG_B05: Place of birth / nativity (B05002e13=foreign born)
- 2020_CBG_B15: Educational attainment 25+ (B15003e1=total 25+, B15003e22=Bachelor's degree, B15003e23=Master's, B15003e25=Doctorate)
- 2020_CBG_B17: Poverty (B17001e1=total, B17001e2=below poverty level)
- 2020_CBG_B19: Household income (B19013e1=median household income estimate)
- 2020_CBG_B21: Veteran status (B21001e2=civilian veterans, B21001e1=total civilian 18+)
- 2020_CBG_B23: Employment/labor force (B23025e2=labor force, B23025e4=employed, B23025e5=unemployed)
- 2020_CBG_B25: Housing tenure (B25003e1=total occupied, B25003e2=owner-occupied, B25003e3=renter-occupied)
- 2020_METADATA_CBG_FIELD_DESCRIPTIONS: Column code definitions
- 2020_METADATA_CBG_FIPS_CODES: State FIPS reference

AGGREGATION RULES:
- SUM for counts (population, number of people, units)
- AVG for medians and rates (median income, median age, median rent) — block-group medians average to state-level estimates
- Exclude nulls and zeros for median/rate columns: WHERE "col" IS NOT NULL AND "col" > 0
- For percentages: SUM(numerator) * 100.0 / NULLIF(SUM(denominator), 0)
- County = first 5 chars of CENSUS_BLOCK_GROUP; State = first 2 chars

OUTPUT RULES:
- Output ONLY the SQL statement — no explanation, no markdown, no backticks
- If the question cannot be answered with available data, output exactly: CANNOT_ANSWER
- Use SELECT statements only — never INSERT, UPDATE, DELETE, DROP, CREATE
"""

_FIPS_CONTEXT = """\
AL=01, AK=02, AZ=04, AR=05, CA=06, CO=08, CT=09, DE=10, DC=11, FL=12,
GA=13, HI=15, ID=16, IL=17, IN=18, IA=19, KS=20, KY=21, LA=22, ME=23,
MD=24, MA=25, MI=26, MN=27, MS=28, MO=29, MT=30, NE=31, NV=32, NH=33,
NJ=34, NM=35, NY=36, NC=37, ND=38, OH=39, OK=40, OR=41, PA=42, RI=44,
SC=45, SD=46, TN=47, TX=48, UT=49, VT=50, VA=51, WA=53, WV=54, WI=55,
WY=56, PR=72"""


async def generate_sql(
    question: str,
    schema_context: str,
    field_context: str,
    few_shot_examples: str,
    conversation_history: str,
    previous_sql: Optional[str] = None,
    previous_error: Optional[str] = None,
) -> str:
    system = _SQL_SYSTEM.format(fips_context=_FIPS_CONTEXT)

    parts = []
    if schema_context:
        parts.append(f"RELEVANT TABLES:\n{schema_context}")
    if field_context:
        parts.append(field_context)
    if few_shot_examples:
        parts.append(few_shot_examples)
    if conversation_history:
        parts.append(conversation_history)

    if previous_sql and previous_error:
        parts.append(
            f"PREVIOUS ATTEMPT FAILED:\nSQL: {previous_sql}\nError: {previous_error}\n"
            "Fix the SQL to resolve this error. Output only the corrected SQL."
        )
        parts.append(f"Question: {question}\nCorrected SQL:")
    else:
        parts.append(f"Question: {question}\nSQL:")

    user_content = "\n\n".join(parts)

    resp = await _client.chat.completions.create(
        model=settings.smart_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_tokens=800,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# SQL error fix — uses fast model (simpler task)
# ---------------------------------------------------------------------------

async def fix_sql(
    original_question: str,
    failed_sql: str,
    error_message: str,
    schema_context: str,
) -> str:
    prompt = f"""\
Fix this Snowflake SQL that returned an error.
Output ONLY the corrected SQL, nothing else.

DATABASE CONVENTIONS:
- All tables are in the PUBLIC schema
- Tables use Census Block Group (CBG) level data
- Quote all column names with double quotes: "B01001e1"

KEY TABLE GROUPS:
- 2020_CBG_B01: Population and age (B01001e1=total pop, B01002e1=median age)
- 2020_CBG_B02: Race categories (B02001e1=total, e2=White, e3=Black, e4=Native American, e5=Asian, e6=Pacific Islander, e7=Other, e8=Two or more)
- 2020_CBG_B03: Hispanic/Latino (B03002e1=total, B03002e12=Hispanic or Latino)
- 2020_CBG_B05: Place of birth / nativity (B05002e13=foreign born)
- 2020_CBG_B15: Educational attainment 25+ (B15003e1=total 25+, B15003e22=Bachelor's degree, B15003e23=Master's, B15003e25=Doctorate)
- 2020_CBG_B17: Poverty (B17001e1=total, B17001e2=below poverty level)
- 2020_CBG_B19: Household income (B19013e1=median household income estimate)
- 2020_CBG_B21: Veteran status (B21001e2=civilian veterans, B21001e1=total civilian 18+)
- 2020_CBG_B23: Employment/labor force (B23025e2=labor force, B23025e4=employed, B23025e5=unemployed)
- 2020_CBG_B25: Housing tenure (B25003e1=total occupied, B25003e2=owner-occupied, B25003e3=renter-occupied)
- 2020_METADATA_CBG_FIELD_DESCRIPTIONS: Column code definitions
- 2020_METADATA_CBG_FIPS_CODES: State FIPS reference

Question: {original_question}
Failed SQL: {failed_sql}
Error: {error_message}
Schema context: {schema_context[:500]}
"""
    resp = await _client.chat.completions.create(
        model=settings.fast_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Answer synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """\
You are a helpful US Census data assistant. Synthesize SQL query results into a clear,
natural English answer.

Rules:
- Cite the year (2020 Census data)
- Format numbers with commas (39,346,023 not 39346023)
- Format percentages to 2 decimal places
- For state rankings, list each entry clearly
- If results show state FIPS codes, convert them to state names
- Do NOT mention SQL, databases, tables, or technical details
- Do NOT make up numbers not in the results
- Keep responses concise but complete
- If the result is a single number, say so clearly with the appropriate unit

FIPS to state name mapping:
01=Alabama, 02=Alaska, 04=Arizona, 05=Arkansas, 06=California, 08=Colorado,
09=Connecticut, 10=Delaware, 11=District of Columbia, 12=Florida, 13=Georgia,
15=Hawaii, 16=Idaho, 17=Illinois, 18=Indiana, 19=Iowa, 20=Kansas, 21=Kentucky,
22=Louisiana, 23=Maine, 24=Maryland, 25=Massachusetts, 26=Michigan, 27=Minnesota,
28=Mississippi, 29=Missouri, 30=Montana, 31=Nebraska, 32=Nevada, 33=New Hampshire,
34=New Jersey, 35=New Mexico, 36=New York, 37=North Carolina, 38=North Dakota,
39=Ohio, 40=Oklahoma, 41=Oregon, 42=Pennsylvania, 44=Rhode Island,
45=South Carolina, 46=South Dakota, 47=Tennessee, 48=Texas, 49=Utah,
50=Vermont, 51=Virginia, 53=Washington, 54=West Virginia, 55=Wisconsin,
56=Wyoming, 72=Puerto Rico
"""


async def synthesize_answer(
    question: str,
    sql: str,
    rows: list[dict],
    conversation_history: str,
) -> AsyncIterator[str]:
    """Streams the synthesized answer."""
    results_text = str(rows[:20])  # cap for prompt size
    context = f"\n\nConversation context:\n{conversation_history}" if conversation_history else ""

    prompt = f"""\
Question: {question}
SQL executed: {sql}
Results: {results_text}{context}

Provide a clear, natural answer:"""

    stream = await _client.chat.completions.create(
        model=settings.smart_model,
        messages=[
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
        temperature=0.2,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Qualitative / general knowledge answers
# ---------------------------------------------------------------------------

_QUALITATIVE_SYSTEM = """\
You are a friendly, knowledgeable US Census data assistant.

Your job:
1. If the user is greeting you or making small talk, respond warmly and invite a Census question.
2. If they ask about Census methodology, definitions, or general demographic context, answer from your knowledge.
   Always mention this is general knowledge, not live database data, and suggest a related quantitative question.
3. If the question is about something NOT in the 2020 ACS Census dataset, explain clearly what IS available.
   Topics NOT in this dataset: homelessness counts, undocumented immigrants, Middle Eastern as a separate category,
   real-time data, data after 2020, individual records, COVID data, crime statistics.
4. If the question is about what topics you can answer, explain concisely with examples.
5. If completely off-topic, politely redirect.

Census topics you CAN answer with data:
- Population counts (state, county, block group)
- Race and ethnicity (White, Black, Asian, Native American, Pacific Islander, Hispanic/Latino)
- Age distribution and median age
- Household income and poverty rates
- Educational attainment
- Housing (homeownership, renting, vacancy)
- Employment and unemployment
- Veteran status
- Place of birth / nativity

Keep responses concise and helpful. End with a specific suggested question where appropriate.
"""


async def qualitative_answer(
    message: str,
    conversation_history: str,
) -> AsyncIterator[str]:
    """Streams a qualitative/conversational answer."""
    context = f"\n\nConversation context:\n{conversation_history}" if conversation_history else ""
    prompt = f"{context}\n\nUser: {message}"

    stream = await _client.chat.completions.create(
        model=settings.smart_model,
        messages=[
            {"role": "system", "content": _QUALITATIVE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.3,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Plausibility check
# ---------------------------------------------------------------------------

_PLAUSIBILITY_SYSTEM = """\
You check whether query results are plausible for a given US Census question.
Respond with ONLY "PLAUSIBLE" or "IMPLAUSIBLE: <one-line reason>".

Context:
- US total population: ~330 million
- State populations: 500K (smallest) to 39M (California)
- County populations: <1K to 10M+
- Median household income: $30K–$120K per state
- Poverty rates: 5%–25%
- Percentages must be 0–100
- Veteran count cannot exceed working-age civilian population
- Homeownership rates: 40%–80%
- Education (bachelor+): 10%–65%
"""


async def check_plausibility(question: str, result_summary: str) -> str:
    """Returns 'PLAUSIBLE' or 'IMPLAUSIBLE: reason'."""
    try:
        resp = await _client.chat.completions.create(
            model=settings.fast_model,
            messages=[
                {"role": "system", "content": _PLAUSIBILITY_SYSTEM},
                {"role": "user", "content": f"Question: {question}\nResult: {result_summary}\nIs this plausible?"},
            ],
            max_tokens=60,
            temperature=0,
        )
        return (resp.choices[0].message.content or "PLAUSIBLE").strip()
    except Exception:
        return "PLAUSIBLE"  # fail open — don't block valid answers
