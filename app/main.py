"""
FastAPI application — chat endpoint with SSE streaming.
"""
import time
import uuid
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import CensusAgent
from app.config import settings
from app.logging_config import get_logger, setup_logging
from app import snowflake_client

setup_logging()
logger = get_logger(__name__)

_agent = CensusAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # await _agent.init()
    asyncio.create_task(_agent.init())
    yield


app = FastAPI(
    title="Census Data Agent",
    version=settings.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open("frontend/index.html") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Census Agent</h1><p>Frontend not found.</p>"


@app.get("/api/health")
async def health():
    sf_ok = await snowflake_client.health_check()
    redis_ok = await _agent.session_manager.health_check()
    return {
        "status": "ok" if _agent._ready else "starting",
        "snowflake": sf_ok,
        "redis": redis_ok,
        "agent_ready": _agent._ready,
        "version": settings.version,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    """Non-streaming endpoint — waits for full response."""
    session_id = req.session_id or str(uuid.uuid4())
    t0 = time.monotonic()
    parts = []
    async for token in _agent.chat(req.message, session_id):
        if not token.startswith("__STATUS__:"):
            parts.append(token)
    response = "".join(parts)
    elapsed = round((time.monotonic() - t0) * 1000)
    logger.info("http_request POST /api/chat status=200 latency_ms=%d", elapsed)
    return ChatResponse(response=response, session_id=session_id)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming endpoint — sends tokens via SSE as they arrive."""
    session_id = req.session_id or str(uuid.uuid4())

    async def generate():
        try:
            async for token in _agent.chat(req.message, session_id):
                # Status tokens — send as SSE event type "status"
                if token.startswith("__STATUS__:"):
                    status = token.replace("__STATUS__:", "")
                    yield f"event: status\ndata: {status}\n\n"
                else:
                    yield f"data: {token}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            logger.error("stream_error: %s", str(e))
            yield f"event: error\ndata: An unexpected error occurred.\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )
