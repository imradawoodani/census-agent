"""
Snowflake client — async wrapper around the synchronous connector.
Exposes query execution and schema/metadata helpers used at startup.
"""
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import snowflake.connector

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


def _get_connection() -> snowflake.connector.SnowflakeConnection:
    return snowflake.connector.connect(
        account=settings.snowflake_account,
        user=settings.snowflake_user,
        password=settings.snowflake_password,
        database=settings.snowflake_database,
        schema=settings.snowflake_schema,
        warehouse=settings.snowflake_warehouse,
        session_parameters={"QUERY_TAG": "census-agent"},
    )


def _execute_sync(sql: str, limit: int) -> tuple[list[dict[str, Any]], float]:
    """Run a query synchronously. Called in a thread pool."""
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor(snowflake.connector.DictCursor)
        # Inject LIMIT safely if not already present
        normalised = sql.strip().rstrip(";").upper()
        if "LIMIT" not in normalised:
            sql = f"{sql.strip().rstrip(';')} LIMIT {limit}"
        cur.execute(sql, timeout=settings.query_timeout_seconds)
        rows = cur.fetchall()
        return rows, cur.sfqid  # type: ignore[return-value]
    finally:
        if conn:
            conn.close()


async def run_query(
    sql: str,
    trace_id: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """
    Execute SQL asynchronously.
    Returns (rows, elapsed_ms).
    Raises snowflake.connector.errors.ProgrammingError on bad SQL.
    """
    tid = trace_id or str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    t0 = time.monotonic()
    try:
        rows, _ = await asyncio.wait_for(
            loop.run_in_executor(
                _executor, _execute_sync, sql, settings.max_result_rows
            ),
            timeout=settings.query_timeout_seconds + 5,
        )
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("snowflake_query_ok trace_id=%s rows=%d elapsed_ms=%d", tid, len(rows), round(elapsed))
        return rows, elapsed
    except asyncio.TimeoutError:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("snowflake_query_timeout trace_id=%s elapsed_ms=%d", tid, round(elapsed))
        raise TimeoutError("Query exceeded 55 second timeout")
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("snowflake_query_error trace_id=%s: %s", tid, str(e))
        raise


# ---------------------------------------------------------------------------
# Schema helpers — called once at startup
# ---------------------------------------------------------------------------

def _fetch_tables_sync() -> list[dict]:
    """Return all tables in the configured schema."""
    conn = _get_connection()
    try:
        cur = conn.cursor(snowflake.connector.DictCursor)
        cur.execute(
            f"""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{settings.snowflake_schema}'
            AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
            """
        )
        return cur.fetchall()
    finally:
        conn.close()


def _fetch_columns_sync(table_name: str) -> list[dict]:
    """Return column names and data types for a table."""
    conn = _get_connection()
    try:
        cur = conn.cursor(snowflake.connector.DictCursor)
        cur.execute(
            f"""
            SELECT COLUMN_NAME, DATA_TYPE, COMMENT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{settings.snowflake_schema}'
            AND TABLE_NAME = '{table_name}'
            ORDER BY ORDINAL_POSITION
            """
        )
        return cur.fetchall()
    finally:
        conn.close()


def _fetch_field_descriptions_sync() -> list[dict]:
    """
    Fetch the full field description metadata from Census metadata tables.
    These describe what each column code means in plain English.
    Tries both 2020 and 2019 metadata tables and merges.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor(snowflake.connector.DictCursor)
        results = []
        for year in ("2020", "2019"):
            meta_table = f"{year}_METADATA_CBG_FIELD_DESCRIPTIONS"
            try:
                cur.execute(
                    f"""
                    SELECT
                        TABLE_NUMBER,
                        TABLE_TITLE,
                        FIELD_LEVEL_1,
                        FIELD_LEVEL_2,
                        FIELD_LEVEL_3,
                        FIELD_LEVEL_4
                    FROM "{meta_table}"
                    """
                )
                rows = cur.fetchall()
                for row in rows:
                    row["_year"] = year
                results.extend(rows)
            except Exception:
                pass  # table may not exist for all years
        return results
    finally:
        conn.close()


def _fetch_fips_codes_sync() -> dict[str, str]:
    """
    Return a mapping of state name / abbreviation → FIPS code.
    FIPS code = first 2 chars of CENSUS_BLOCK_GROUP.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor(snowflake.connector.DictCursor)
        # Try 2020 then 2019
        for year in ("2020", "2019"):
            fips_table = f"{year}_METADATA_CBG_FIPS_CODES"
            try:
                cur.execute(f'SELECT DISTINCT STATE, STATE_FIPS FROM "{fips_table}"')
                rows = cur.fetchall()
                mapping: dict[str, str] = {}
                for row in rows:
                    state = (row.get("STATE") or "").strip()
                    fips = str(row.get("STATE_FIPS") or "").zfill(2)
                    if state and fips != "00":
                        mapping[state.upper()] = fips
                        mapping[fips] = fips  # identity lookup
                if mapping:
                    return mapping
            except Exception:
                pass
        return {}
    finally:
        conn.close()


async def fetch_tables() -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _fetch_tables_sync)


async def fetch_columns(table_name: str) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _fetch_columns_sync, table_name)


async def fetch_field_descriptions() -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _fetch_field_descriptions_sync)


async def fetch_fips_codes() -> dict[str, str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _fetch_fips_codes_sync)


async def health_check() -> bool:
    """Returns True if Snowflake is reachable."""
    try:
        rows, _ = await run_query("SELECT 1 AS ping")
        return bool(rows)
    except Exception:
        return False
