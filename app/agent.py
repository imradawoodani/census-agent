"""
CensusAgent — the main pipeline orchestrator.
Lifecycle: classify intent → route → retrieve context → generate SQL →
           validate → execute → plausibility check → synthesize answer
"""
import time
import uuid
from typing import AsyncIterator

from app import guardrails, llm_client, snowflake_client, sql_validator
from app.cache import classify_error_message, is_retryable
from app.config import settings
from app.embeddings import embed_query
from app.few_shot import FewShotRetriever
from app.field_retriever import FieldRetriever
from app.guardrails import GuardrailResult
from app.llm_client import IntentType
from app.logging_config import get_logger
from app.schema_retriever import SchemaRetriever
from app.session import SessionManager

logger = get_logger(__name__)


class CensusAgent:
    def __init__(self) -> None:
        self.schema_retriever = SchemaRetriever()
        self.field_retriever = FieldRetriever()
        self.few_shot = FewShotRetriever()
        self.session_manager = SessionManager()
        self._ready = False

    async def init(self) -> None:
        """
        Startup sequence:
        1. Fetch Snowflake schema (tables + columns)
        2. Fetch Census field descriptions (metadata)
        3. Build schema embeddings (enriched with field meanings)
        4. Build field-level embeddings (disk-cached)
        5. Build few-shot embeddings
        6. Init session manager
        """
        t0 = time.monotonic()
        logger.info("agent_initializing")

        try:
            # --- Snowflake schema ---
            tables = await snowflake_client.fetch_tables()
            columns_by_table: dict[str, list[dict]] = {}
            # Fetch columns for key tables (not all 70+; save startup time)
            key_prefixes = ("2020_CBG_", "2019_CBG_")
            for t in tables:
                tname = t["TABLE_NAME"]
                if any(tname.startswith(p) for p in key_prefixes):
                    cols = await snowflake_client.fetch_columns(tname)
                    columns_by_table[tname] = cols

            # --- Field descriptions (plain-English column meanings) ---
            field_desc_rows = await snowflake_client.fetch_field_descriptions()
            # Build a flat mapping: TABLE_NUMBER+LEVEL -> description text
            # Used to enrich schema embeddings
            field_descriptions: dict[str, str] = {}
            for row in field_desc_rows:
                tnum = (row.get("TABLE_NUMBER") or "").strip()
                parts = [
                    row.get("FIELD_LEVEL_1") or "",
                    row.get("FIELD_LEVEL_2") or "",
                    row.get("FIELD_LEVEL_3") or "",
                ]
                desc = " > ".join(p.strip() for p in parts if p.strip())
                if tnum and desc:
                    # The column code is approximated from table number + eN pattern
                    # We use the table number as key for embedding enrichment
                    existing = field_descriptions.get(tnum, "")
                    if not existing:
                        field_descriptions[tnum] = desc

            # Also build code → description from metadata for column enrichment
            code_to_desc: dict[str, str] = {}
            for row in field_desc_rows:
                tnum = (row.get("TABLE_NUMBER") or "").strip()
                title = (row.get("TABLE_TITLE") or "").strip()
                if tnum and title:
                    code_to_desc[tnum] = title

            # --- Schema embeddings ---
            await self.schema_retriever.build(tables, columns_by_table, code_to_desc)
            logger.info("agent_schema_ready table_count=%d", len(tables))
        except Exception as e:
            logger.error("agent_schema_init_failed: %s", str(e))
            # Non-fatal — agent can still answer with degraded schema context

        try:
            # --- Field-level embeddings (disk-cached) ---
            field_desc_rows_full = await snowflake_client.fetch_field_descriptions()
            await self.field_retriever.build(field_desc_rows_full)
        except Exception as e:
            logger.error("agent_field_init_failed: %s", str(e))

        try:
            # --- Few-shot embeddings ---
            await self.few_shot.build()
        except Exception as e:
            logger.error("agent_few_shot_init_failed: %s", str(e))

        # --- Session manager ---
        await self.session_manager.init()

        elapsed = round((time.monotonic() - t0) * 1000)
        self._ready = True
        logger.info("agent_ready total_ms=%d", elapsed)

    async def chat(
        self, message: str, session_id: str
    ) -> AsyncIterator[str]:
        """
        Process one chat turn. Yields status tokens then the response text.
        Status tokens are prefixed with __STATUS__: for the frontend to handle.
        """
        trace_id = str(uuid.uuid4())[:8]
        history = await self.session_manager.get_history(session_id)
        history_text = self.session_manager.format_history(history)

        # ---------------------------------------------------------------------------
        # 1. Guardrail check
        # ---------------------------------------------------------------------------
        guard = guardrails.check(message)

        if guard == GuardrailResult.INJECTION:
            response = "I can't process that request."
            await self.session_manager.append_turn(session_id, message, response)
            yield response
            return

        if guard == GuardrailResult.GREETING:
            # Route directly to qualitative without burning an LLM classification call
            response_parts = []
            async for token in llm_client.qualitative_answer(message, history_text):
                response_parts.append(token)
                yield token
            await self.session_manager.append_turn(session_id, message, "".join(response_parts))
            return

        # ---------------------------------------------------------------------------
        # 2. Intent classification
        # ---------------------------------------------------------------------------
        yield "__STATUS__:Classifying question..."
        intent = await llm_client.classify_intent(message, history_text)
        logger.info("intent_classified trace_id=%s intent=%s", trace_id, intent.value)

        if intent in (IntentType.OFF_TOPIC, IntentType.QUALITATIVE, IntentType.AMBIGUOUS, IntentType.GREETING):
            response_parts = []
            async for token in llm_client.qualitative_answer(message, history_text):
                response_parts.append(token)
                yield token
            await self.session_manager.append_turn(session_id, message, "".join(response_parts))
            return

        # ---------------------------------------------------------------------------
        # 3. QUANTITATIVE path — retrieve context
        # ---------------------------------------------------------------------------
        yield "__STATUS__:Retrieving schema..."

        q_emb = await embed_query(message)

        # Schema retrieval
        relevant_tables = self.schema_retriever.retrieve(q_emb)
        schema_context = self.schema_retriever.format_schema(relevant_tables)

        # Field retrieval (dynamic — no hardcoding)
        relevant_fields = await self.field_retriever.retrieve(message)
        field_context = self.field_retriever.format_field_context(relevant_fields)

        # Few-shot examples
        examples = await self.few_shot.retrieve(message)
        few_shot_text = self.few_shot.format_examples(examples)

        # ---------------------------------------------------------------------------
        # 4. SQL generation + retry loop
        # ---------------------------------------------------------------------------
        yield "__STATUS__:Generating query..."

        sql = ""
        last_error = ""
        rows = []

        for attempt in range(settings.max_sql_retries + 1):
            try:
                if attempt == 0:
                    sql = await llm_client.generate_sql(
                        question=message,
                        schema_context=schema_context,
                        field_context=field_context,
                        few_shot_examples=few_shot_text,
                        conversation_history=history_text,
                    )
                else:
                    sql = await llm_client.generate_sql(
                        question=message,
                        schema_context=schema_context,
                        field_context=field_context,
                        few_shot_examples=few_shot_text,
                        conversation_history=history_text,
                        previous_sql=sql,
                        previous_error=last_error,
                    )

                if not sql or sql.upper().startswith("CANNOT_ANSWER"):
                    response = (
                        "The Census dataset doesn't contain data to answer that specific question. "
                        "I can help with population counts, income, poverty rates, race and ethnicity, education, housing, employment, veteran status, and age distribution."
                    )
                    await self.session_manager.append_turn(session_id, message, response)
                    yield response
                    return

                logger.info("sql_generated trace_id=%s attempt=%d", trace_id, attempt)

                # Validate SQL structurally
                try:
                    validated_sql = sql_validator.validate_and_limit(sql, settings.max_result_rows)
                except sql_validator.SQLValidationError as e:
                    logger.warning("sql_validation_failed trace_id=%s: %s", trace_id, str(e))
                    last_error = str(e)
                    continue

                # Execute
                yield "__STATUS__:Querying database..."
                rows, _ = await snowflake_client.run_query(validated_sql, trace_id=trace_id)
                sql = validated_sql
                break  # success

            except TimeoutError:
                response = "This query took too long (over 55 seconds). Try asking about a specific state instead of nationwide data."
                await self.session_manager.append_turn(session_id, message, response)
                yield response
                return
            except Exception as e:
                last_error = str(e)
                logger.warning("sql_execution_error trace_id=%s attempt=%d: %s", trace_id, attempt, last_error[:200])
                if not is_retryable(e):
                    response = classify_error_message(e)
                    await self.session_manager.append_turn(session_id, message, response)
                    yield response
                    return
                if attempt >= settings.max_sql_retries:
                    response = f"I couldn't retrieve that data after multiple attempts. Try rephrasing your question or asking about a different metric."
                    await self.session_manager.append_turn(session_id, message, response)
                    yield response
                    return

        # ---------------------------------------------------------------------------
        # 5. Empty results
        # ---------------------------------------------------------------------------
        if not rows:
            response = (
                "I found no data matching that question. This might mean:\n"
                "* The geographic area you mentioned doesn't match Census data\n"
                "* That metric isn't available at the requested level\n\n"
                "Try asking about a state or county, or rephrase the question."
            )
            await self.session_manager.append_turn(session_id, message, response)
            yield response
            return

        # ---------------------------------------------------------------------------
        # 6. Plausibility check
        # ---------------------------------------------------------------------------
        result_summary = str(rows[:3])
        plausibility = await llm_client.check_plausibility(message, result_summary)
        if plausibility.startswith("IMPLAUSIBLE"):
            reason = plausibility.replace("IMPLAUSIBLE:", "").strip()
            logger.warning("plausibility_check_failed trace_id=%s reason=%s", trace_id, reason)
            response = (
                f"I found data but the results don't look right ({reason}). "
                "This may be a data interpretation issue. Try rephrasing, for example "
                "by being more specific about which metric you want."
            )
            await self.session_manager.append_turn(session_id, message, response)
            yield response
            return

        # ---------------------------------------------------------------------------
        # 7. Synthesize answer
        # ---------------------------------------------------------------------------
        yield "__STATUS__:Generating answer..."
        response_parts = []
        async for token in llm_client.synthesize_answer(message, sql, rows, history_text):
            response_parts.append(token)
            yield token

        full_response = "".join(response_parts)
        await self.session_manager.append_turn(session_id, message, full_response)
        logger.info("chat_complete trace_id=%s len=%d", trace_id, len(full_response))
