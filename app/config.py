"""
Application configuration — all env vars live here.
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Snowflake
    snowflake_account: str = Field(..., description="Snowflake account identifier")
    snowflake_user: str = Field(..., description="Snowflake username")
    snowflake_password: str = Field(..., description="Snowflake password")
    snowflake_database: str = Field("US_OPEN_CENSUS")
    snowflake_schema: str = Field("PUBLIC")
    snowflake_warehouse: str = Field("COMPUTE_WH")

    # DigitalOcean Inference (all LLM calls)
    do_model_access_key: str = Field(..., description="DO serverless inference key")
    do_inference_base_url: str = Field("https://inference.do-ai.run/v1")

    # Model names — DO inference endpoint
    fast_model: str = Field("llama3-8b-instruct")
    smart_model: str = Field("llama3.3-70b-instruct")

    # Cohere (embeddings only)
    cohere_api_key: str = Field(..., description="Cohere API key for embeddings")

    # Redis (optional — falls back to in-memory)
    redis_url: Optional[str] = Field(None)

    # Field embedding cache path
    field_cache_path: str = Field("field_embeddings_cache.json")

    # Pipeline tuning
    max_sql_retries: int = Field(3)
    query_timeout_seconds: int = Field(55)
    max_result_rows: int = Field(500)
    schema_top_k: int = Field(8)
    field_top_k: int = Field(15)
    few_shot_top_k: int = Field(3)

    # Logging
    log_level: str = Field("INFO")
    log_format: str = Field("json")

    # App version
    version: str = Field("3.0.0")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
