"""
Test configuration. Mocks external dependencies so tests run without
Snowflake, Cohere, or DigitalOcean credentials.
"""
import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock snowflake before any app module imports it
# ---------------------------------------------------------------------------
_sf_mock = MagicMock()
_sf_mock.connector = MagicMock()
_sf_mock.connector.DictCursor = MagicMock()
_sf_mock.connector.errors = MagicMock()
_sf_mock.connector.errors.ProgrammingError = Exception
sys.modules["snowflake"] = _sf_mock
sys.modules["snowflake.connector"] = _sf_mock.connector

# ---------------------------------------------------------------------------
# Set all required env vars before any app module is imported
# ---------------------------------------------------------------------------
os.environ.setdefault("SNOWFLAKE_ACCOUNT", "test-account")
os.environ.setdefault("SNOWFLAKE_USER", "test-user")
os.environ.setdefault("SNOWFLAKE_PASSWORD", "test-password")
os.environ.setdefault("SNOWFLAKE_DATABASE", "US_OPEN_CENSUS")
os.environ.setdefault("SNOWFLAKE_SCHEMA", "PUBLIC")
os.environ.setdefault("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
os.environ.setdefault("DO_MODEL_ACCESS_KEY", "test-do-key")
os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")

# Invalidate any cached settings
try:
    from app.config import get_settings
    get_settings.cache_clear()
except Exception:
    pass
