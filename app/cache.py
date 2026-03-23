"""
Error classifier. Determines whether a Snowflake error is worth retrying.
Fast fail on unrecoverable errors to avoid burning the timeout budget.
"""
from app.logging_config import get_logger

logger = get_logger(__name__)

# Errors worth retrying — the LLM can often fix these
_RETRYABLE_PATTERNS = [
    "invalid identifier",
    "column",
    "object does not exist",
    "syntax error",
    "unexpected",
    "expected",
    "ambiguous",
    "does not exist",
]

# Errors NOT worth retrying
_FATAL_PATTERNS = [
    "query execution time exceeded",
    "insufficient privileges",
    "access denied",
    "authentication",
    "account does not exist",
    "network error",
    "connection",
]


def is_retryable(error: Exception) -> bool:
    msg = str(error).lower()
    for pat in _FATAL_PATTERNS:
        if pat in msg:
            return False
    for pat in _RETRYABLE_PATTERNS:
        if pat in msg:
            return True
    # Default: retry once for unknown errors
    return True


def classify_error_message(error: Exception) -> str:
    """Return a user-facing explanation for a Snowflake error."""
    msg = str(error).lower()
    if "timeout" in msg or "query execution time exceeded" in msg:
        return "This query took too long to execute. Try asking about a smaller geographic area or a simpler metric."
    if "insufficient privileges" in msg or "access denied" in msg:
        return "I don't have permission to access that data."
    if "object does not exist" in msg or "invalid identifier" in msg:
        return "I couldn't find the data for that question. Try rephrasing it."
    return "I ran into a database error. Please try rephrasing your question."
