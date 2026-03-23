"""
Schema retriever — embeds table descriptions enriched with field metadata
so that semantic search on the question finds the right tables.
"""
from app.config import settings
from app.embeddings import embed_documents, embed_query, top_k
from app.logging_config import get_logger

logger = get_logger(__name__)


class SchemaRetriever:
    def __init__(self) -> None:
        self._table_index: list[dict] = []

    async def build(
        self,
        tables: list[dict],
        columns_by_table: dict[str, list[dict]],
        field_descriptions: dict[str, str],
    ) -> None:
        """
        Build the embedding index for table-level retrieval.
        Each entry describes a table using column codes + their plain-English meanings.
        """
        descriptions: list[str] = []
        metadata: list[dict] = []

        for table in tables:
            tname = table["TABLE_NAME"]
            cols = columns_by_table.get(tname, [])
            # Skip metadata-only and geometry tables from the retrieval index
            # (they'll still be injected into the SQL prompt structurally)
            col_parts = []
            for col in cols[:30]:  # cap at 30 cols per table for embedding
                cname = col["COLUMN_NAME"]
                desc = field_descriptions.get(cname, "")
                if desc:
                    col_parts.append(f"{cname} ({desc})")
                else:
                    col_parts.append(cname)
            text = f"Table {tname}: {', '.join(col_parts)}"
            descriptions.append(text)
            metadata.append({"table_name": tname, "description": text})

        logger.info("schema_retriever_embedding count=%d", len(descriptions))
        embeddings = await embed_documents(descriptions)
        for i, emb in enumerate(embeddings):
            metadata[i]["embedding"] = emb

        self._table_index = metadata
        logger.info("schema_retriever_ready count=%d", len(self._table_index))

    def retrieve(self, query_embedding: list[float], k: int | None = None) -> list[dict]:
        """Return the top-k most relevant tables for a query embedding."""
        k = k or settings.schema_top_k
        return top_k(query_embedding, self._table_index, k=k)

    def format_schema(self, tables: list[dict]) -> str:
        """Format retrieved tables as a schema context string for the prompt."""
        parts = []
        for entry in tables:
            parts.append(f"  {entry['description']}")
        return "\n".join(parts)
