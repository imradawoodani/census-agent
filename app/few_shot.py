"""
Few-shot examples — verified question and SQL pairs for the dataset.
These examples teach the LLM:
  - FIPS-based state filtering: SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '{fips}'
  - Table join pattern: tables join on CENSUS_BLOCK_GROUP
  - Aggregation conventions: SUM for counts, AVG for medians/rates
  - Column quoting with double quotes

Column codes here were verified against the Census metadata table.
"""
from app.config import settings
from app.embeddings import embed_documents, embed_query, top_k
from app.logging_config import get_logger

logger = get_logger(__name__)

# Verified examples, column codes confirmed from 2020_METADATA_CBG_FIELD_DESCRIPTIONS
EXAMPLES: list[dict[str, str]] = [
    {
        "question": "What is the total population of California?",
        "sql": """SELECT SUM("B01001e1") AS total_population
FROM "2020_CBG_B01"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '06'""",
        "note": "B01001e1=total population, CA FIPS=06",
    },
    {
        "question": "Which 5 states have the highest total population?",
        "sql": """SELECT SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) AS state_fips,
       SUM("B01001e1") AS total_population
FROM "2020_CBG_B01"
GROUP BY state_fips
ORDER BY total_population DESC
LIMIT 5""",
        "note": "Group by FIPS to rank all states",
    },
    {
        "question": "What is the median household income in Texas?",
        "sql": """SELECT AVG("B19013e1") AS avg_median_household_income
FROM "2020_CBG_B19"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '48'
  AND "B19013e1" IS NOT NULL AND "B19013e1" > 0""",
        "note": "B19013e1=median household income estimate. Use AVG across block groups. TX FIPS=48",
    },
    {
        "question": "Which 5 states have the highest median household income?",
        "sql": """SELECT SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) AS state_fips,
       AVG("B19013e1") AS avg_median_household_income
FROM "2020_CBG_B19"
WHERE "B19013e1" IS NOT NULL AND "B19013e1" > 0
GROUP BY state_fips
ORDER BY avg_median_household_income DESC
LIMIT 5""",
        "note": "Use AVG of B19013e1 (median HH income estimate) across block groups per state",
    },
    {
        "question": "What is the poverty rate in Mississippi?",
        "sql": """SELECT SUM("B17001e2") * 100.0 / NULLIF(SUM("B17001e1"), 0) AS poverty_rate_pct
FROM "2020_CBG_B17"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '28'""",
        "note": "B17001e1=total population for poverty, B17001e2=below poverty level. MS FIPS=28",
    },
    {
        "question": "Which 5 counties have the highest poverty rates?",
        "sql": """SELECT SUBSTR(CENSUS_BLOCK_GROUP, 1, 5) AS county_fips,
       SUM("B17001e2") * 100.0 / NULLIF(SUM("B17001e1"), 0) AS poverty_rate_pct
FROM "2020_CBG_B17"
GROUP BY county_fips
HAVING SUM("B17001e1") > 1000
ORDER BY poverty_rate_pct DESC
LIMIT 5""",
        "note": "County = first 5 chars of CENSUS_BLOCK_GROUP. Filter small counties with HAVING",
    },
    {
        "question": "What percentage of Florida residents are Hispanic?",
        "sql": """SELECT SUM("B03002e12") * 100.0 / NULLIF(SUM("B03002e1"), 0) AS hispanic_pct
FROM "2020_CBG_B03"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '12'""",
        "note": "B03002e1=total, B03002e12=Hispanic or Latino. FL FIPS=12",
    },
    {
        "question": "What is the racial breakdown of New York?",
        "sql": """SELECT
  SUM("B02001e2") AS white_alone,
  SUM("B02001e3") AS black_alone,
  SUM("B02001e4") AS native_american_alone,
  SUM("B02001e5") AS asian_alone,
  SUM("B02001e6") AS pacific_islander_alone,
  SUM("B02001e7") AS other_race_alone,
  SUM("B02001e8") AS two_or_more_races,
  SUM("B02001e1") AS total_population
FROM "2020_CBG_B02"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '36'""",
        "note": "B02001 covers race categories. NY FIPS=36",
    },
    {
        "question": "What percentage of California adults have a bachelor's degree?",
        "sql": """SELECT SUM("B15003e22") * 100.0 / NULLIF(SUM("B15003e1"), 0) AS bachelors_pct
FROM "2020_CBG_B15"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '06'""",
        "note": "B15003e1=total 25+, B15003e22=bachelor's degree. CA FIPS=06",
    },
    {
        "question": "What is the homeownership rate in Ohio?",
        "sql": """SELECT SUM("B25003e2") * 100.0 / NULLIF(SUM("B25003e1"), 0) AS homeownership_rate_pct
FROM "2020_CBG_B25"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '39'""",
        "note": "B25003e1=total occupied units, B25003e2=owner-occupied. OH FIPS=39",
    },
    {
        "question": "How many veterans live in Texas?",
        "sql": """SELECT SUM("B21001e2") AS civilian_veterans
FROM "2020_CBG_B21"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '48'""",
        "note": "B21001e2=civilian veterans 18+. TX FIPS=48",
    },
    {
        "question": "What is the median age in Florida?",
        "sql": """SELECT AVG("B01002e1") AS median_age
FROM "2020_CBG_B01"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '12'
  AND "B01002e1" IS NOT NULL AND "B01002e1" > 0""",
        "note": "B01002e1=median age estimate. Use AVG across block groups. FL FIPS=12",
    },
    {
        "question": "What percentage of households in California are renter-occupied?",
        "sql": """SELECT SUM("B25003e3") * 100.0 / NULLIF(SUM("B25003e1"), 0) AS renter_pct
FROM "2020_CBG_B25"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '06'""",
        "note": "B25003e3=renter-occupied, B25003e1=total. CA FIPS=06",
    },
    {
        "question": "What is the unemployment rate in Michigan?",
        "sql": """SELECT SUM("B23025e5") * 100.0 / NULLIF(SUM("B23025e2"), 0) AS unemployment_rate_pct
FROM "2020_CBG_B23"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '26'""",
        "note": "B23025e2=labor force, B23025e5=unemployed. MI FIPS=26",
    },
    {
        "question": "How many people in Illinois were born outside the US?",
        "sql": """SELECT SUM("B05002e13") AS foreign_born
FROM "2020_CBG_B05"
WHERE SUBSTR(CENSUS_BLOCK_GROUP, 1, 2) = '17'""",
        "note": "B05002e13=foreign born. IL FIPS=17",
    },
]


class FewShotRetriever:
    def __init__(self) -> None:
        self._index: list[dict] = []

    async def build(self) -> None:
        texts = [e["question"] for e in EXAMPLES]
        embeddings = await embed_documents(texts)
        self._index = [
            {**ex, "embedding": emb} for ex, emb in zip(EXAMPLES, embeddings)
        ]
        logger.info("few_shot_ready count=%d", len(self._index))

    async def retrieve(self, question: str) -> list[dict]:
        if not self._index:
            return []
        q_emb = await embed_query(question)
        return top_k(q_emb, self._index, k=settings.few_shot_top_k)

    def format_examples(self, examples: list[dict]) -> str:
        parts = ["VERIFIED EXAMPLE QUERIES — follow these patterns exactly:\n"]
        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f"Question: {ex['question']}")
            parts.append(f"SQL:\n{ex['sql']}")
            if ex.get("note"):
                parts.append(f"Notes: {ex['note']}")
            parts.append("")
        return "\n".join(parts)
