"""
Session management — stores conversation history per session.
Uses Redis if REDIS_URL is configured, falls back to in-memory.
"""
import json
from collections import defaultdict
from typing import Any, Optional

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_MAX_TURNS = 6  # number of (user, assistant) pairs to keep in context
_SUMMARIZE_AFTER = 4  # summarize oldest turns after this many


class SessionManager:
    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = defaultdict(list)
        self._redis: Optional[Any] = None

    async def init(self) -> None:
        if settings.redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = await aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.info("session_manager_redis_connected")
            except Exception as e:
                logger.warning("session_manager_redis_failed: %s — falling back to in-memory", str(e))
                self._redis = None
        else:
            logger.info("session_manager_memory_mode")

    async def get_history(self, session_id: str) -> list[dict]:
        if self._redis:
            raw = await self._redis.get(f"session:{session_id}")
            if raw:
                return json.loads(raw)
            return []
        return list(self._store[session_id])

    async def append_turn(
        self, session_id: str, user_msg: str, assistant_msg: str
    ) -> None:
        history = await self.get_history(session_id)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

        # Keep only the last N turns
        if len(history) > _MAX_TURNS * 2:
            history = history[-((_MAX_TURNS) * 2):]

        if self._redis:
            await self._redis.setex(
                f"session:{session_id}",
                3600,  # 1 hour TTL
                json.dumps(history),
            )
        else:
            self._store[session_id] = history

    def format_history(self, history: list[dict]) -> str:
        """Format conversation history for injection into prompts."""
        if not history:
            return ""
        parts = ["RECENT CONVERSATION:"]
        for msg in history[-6:]:  # last 3 turns
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content'][:300]}")
        return "\n".join(parts)

    async def health_check(self) -> bool:
        if self._redis:
            try:
                await self._redis.ping()
                return True
            except Exception:
                return False
        return True  # in-memory is always healthy
