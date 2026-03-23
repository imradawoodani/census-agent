"""
Guardrails — input safety checks applied before any LLM call.
Layer 1: pattern-based injection detection (instant, free)
Layer 2: greeting / off-topic routing (handled by intent classifier in agent)
"""
import re
from enum import Enum

from app.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Injection patterns
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"forget\s+(everything|all|your\s+instructions)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+|an\s+)?(?!census|data)", re.I),
    re.compile(r"pretend\s+you\s+are", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\s+mode", re.I),
    re.compile(r"<\s*script\s*>", re.I),
    re.compile(r"system\s+prompt", re.I),
]

# ---------------------------------------------------------------------------
# Greeting keywords — these pass straight to qualitative path
# ---------------------------------------------------------------------------
_GREETING_TERMS = {
    "hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon",
    "good evening", "sup", "what's up", "whats up", "thanks", "thank you",
    "bye", "goodbye", "see you", "take care",
}

# ---------------------------------------------------------------------------
# Census-related keywords — used as a fast positive signal
# ---------------------------------------------------------------------------
_CENSUS_KEYWORDS = {
    "population", "census", "demographic", "resident", "state", "county",
    "income", "poverty", "race", "hispanic", "latino", "white", "black",
    "asian", "education", "degree", "household", "housing", "rent", "own",
    "employment", "unemploy", "veteran", "age", "median", "average",
    "percent", "rate", "total", "how many", "how much", "which state",
    "which county", "largest", "smallest", "highest", "lowest", "top",
    "born", "foreign", "immigration", "citizen", "language",
}


class GuardrailResult(str, Enum):
    ALLOW = "allow"
    GREETING = "greeting"
    INJECTION = "injection"
    LIKELY_OFF_TOPIC = "likely_off_topic"


def check(message: str) -> GuardrailResult:
    """
    Fast pre-LLM check.
    Returns:
      INJECTION — block immediately
      GREETING — route to qualitative/friendly path
      ALLOW — proceed to intent classification
      LIKELY_OFF_TOPIC — proceed to intent classification (LLM decides)
    """
    if not message or not message.strip():
        return GuardrailResult.LIKELY_OFF_TOPIC

    msg_lower = message.lower().strip()

    # Injection check
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(message):
            logger.warning("guardrail_injection_detected: %s", message[:80])
            return GuardrailResult.INJECTION

    # Greeting check (short messages that are purely social)
    if len(msg_lower.split()) <= 6:
        for term in _GREETING_TERMS:
            if term in msg_lower:
                return GuardrailResult.GREETING

    # Has census signal — allow through
    for kw in _CENSUS_KEYWORDS:
        if kw in msg_lower:
            return GuardrailResult.ALLOW

    # No census signal — let intent classifier decide
    return GuardrailResult.LIKELY_OFF_TOPIC
