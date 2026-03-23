"""
Field retriever — dynamically embeds all Census field descriptions from metadata
and retrieves the specific columns relevant to any question.

This is the core mechanism that eliminates hardcoded column mappings.
The Census metadata (fetched from Snowflake at startup) is the ground truth.

Cache: embeddings are saved to field_embeddings_cache.json to avoid
burning through Cohere API limits on every restart.
"""
import json
import os
from pathlib import Path

from app.config import settings
from app.embeddings import embed_documents, embed_query, top_k
from app.logging_config import get_logger

logger = get_logger(__name__)

def _get_cache_path() -> Path:
    return Path(settings.field_cache_path)


class FieldRetriever:
    def __init__(self) -> None:
        self._field_index: list[dict] = []

    async def build(self, field_descriptions: list[dict], cache_path: str | None = None) -> None:
        """
        Build field-level embedding index from Census metadata rows.
        Each row has TABLE_NUMBER, FIELD_LEVEL_1..4 describing what a column means.
        We embed the plain-English description and associate it with the column code.
        Skips imputation flag columns (_m suffix = margin of error, B99 tables).
        Caches to disk to avoid recomputing on restarts.
        """
        cache = Path(cache_path) if cache_path else _get_cache_path()
        if cache.exists():
            try:
                with open(cache) as f:
                    cached = json.load(f)
                # Validate format: each entry must have embedding as a list of floats.
                # Old cache versions may store embeddings differently — discard and rebuild.
                if (
                    cached
                    and isinstance(cached[0].get("embedding"), list)
                    and cached[0]["embedding"]
                    and isinstance(cached[0]["embedding"][0], (int, float))
                ):
                    self._field_index = cached
                    logger.info("field_cache_loaded count=%d source=disk", len(self._field_index))
                    return
                else:
                    logger.warning("field_cache_format_invalid — rebuilding")
                    cache.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("field_cache_load_failed: %s", str(e))

        # Build from metadata rows
        entries: list[dict] = []
        seen_codes: set[str] = set()

        for row in field_descriptions:
            tnum = (row.get("TABLE_NUMBER") or "").strip()
            if not tnum:
                continue
            # Skip imputation-flag tables (B99 prefix)
            if tnum.upper().startswith("B99"):
                continue

            # Build a plain-English description from the field hierarchy
            parts = [
                str(row.get("TABLE_TITLE") or ""),
                str(row.get("FIELD_LEVEL_1") or ""),
                str(row.get("FIELD_LEVEL_2") or ""),
                str(row.get("FIELD_LEVEL_3") or ""),
                str(row.get("FIELD_LEVEL_4") or ""),
            ]
            description = " > ".join(p.strip() for p in parts if p.strip())
            if not description:
                continue

            entry_key = f"{tnum}::{description}"
            if entry_key in seen_codes:
                continue
            seen_codes.add(entry_key)
            entries.append({"table_number": tnum, "description": description, "raw": row})

        if not entries:
            logger.warning("field_retriever_no_entries")
            return

        api_calls = (len(entries) + 95) // 96
        logger.info("field_retriever_embedding count=%d api_calls=%d", len(entries), api_calls)

        texts = [e["description"] for e in entries]
        embeddings = await embed_documents(texts)

        for i, emb in enumerate(embeddings):
            entries[i]["embedding"] = emb

        self._field_index = entries
        logger.info("field_retriever_ready count=%d", len(self._field_index))

        # Persist to disk
        try:
            to_save = [
                {k: v for k, v in e.items() if k != "raw"}
                for e in self._field_index
            ]
            with open(cache, "w") as f:
                json.dump(to_save, f)
            size_kb = cache.stat().st_size // 1024
            logger.info("field_cache_saved count=%d size_kb=%d", len(self._field_index), size_kb)
        except Exception as e:
            logger.warning("field_cache_save_failed: %s", str(e))

    async def retrieve(self, question: str) -> list[dict]:
        """
        Return the top-K most relevant field descriptions for a question.
        Returns empty list gracefully if index is empty.
        """
        if not self._field_index:
            return []
        q_emb = await embed_query(question)
        results = top_k(q_emb, self._field_index, k=settings.field_top_k)
        return results

    def format_field_context(self, fields: list[dict]) -> str:
        """
        Format retrieved fields for injection into the SQL prompt.
        Groups by table_number for clarity.
        """
        if not fields:
            return ""
        by_table: dict[str, list[str]] = {}
        for f in fields:
            tnum = f["table_number"]
            by_table.setdefault(tnum, []).append(f["description"])
        lines = ["RELEVANT CENSUS FIELDS (from official metadata — use these columns):"]
        for tnum, descs in by_table.items():
            lines.append(f"  Table {tnum}:")
            for d in descs:
                lines.append(f"    - {d}")
        return "\n".join(lines)

    @property
    def is_ready(self) -> bool:
        return len(self._field_index) > 0
