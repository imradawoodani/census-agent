"""
SQL validator. Uses sqlglot AST parsing for structural correctness.
No regex: checks the parse tree, not the text.
"""
import sqlglot
import sqlglot.expressions as exp

from app.logging_config import get_logger

logger = get_logger(__name__)

_FORBIDDEN_STATEMENT_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
)


class SQLValidationError(Exception):
    pass


def validate_and_limit(sql: str, limit: int = 500) -> str:
    """
    Parse, validate, and return SQL with a safe LIMIT injected.
    Raises SQLValidationError if the SQL is unsafe or unparseable.
    """
    sql = sql.strip().rstrip(";")
    if not sql:
        raise SQLValidationError("Empty SQL")

    try:
        tree = sqlglot.parse_one(sql, dialect="snowflake")
    except Exception as e:
        raise SQLValidationError(f"SQL parse error: {e}") from e

    if tree is None:
        raise SQLValidationError("Could not parse SQL")

    # Must be a SELECT at the root
    if not isinstance(tree, exp.Select):
        raise SQLValidationError(
            f"Only SELECT statements are allowed, got {type(tree).__name__}"
        )

    # Walk the tree and reject forbidden node types
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_STATEMENT_TYPES):
            raise SQLValidationError(
                f"Forbidden operation in SQL: {type(node).__name__}"
            )

    # Inject LIMIT via AST manipulation (safe — avoids string injection)
    existing_limit = tree.find(exp.Limit)
    if existing_limit is None:
        tree = tree.limit(limit)
    
    return tree.sql(dialect="snowflake")
