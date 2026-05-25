"""
AetherServe HTTP API — thin FastAPI layer over the inference engine.

Exposes /generate (request-response), /metrics, and /health.
The engine runs as a shared background loop; requests are enqueued and
polled asynchronously to avoid blocking the event loop between steps.
"""
import asyncio
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.engine import AetherEngine
from src.models import SequenceRequest, SequenceStatus

app = FastAPI(
    title="AetherServe",
    description=(
        "High-throughput LLM inference engine featuring continuous batching, "
        "PagedAttention-style KV-cache management, and CPU swap-tier preemption."
    ),
    version="0.2.0",
)

_engine: Optional[AetherEngine] = None
_engine_lock = asyncio.Lock()


@app.on_event("startup")
async def _startup() -> None:
    global _engine
    _engine = AetherEngine(num_gpu_blocks=256, num_cpu_blocks=512, max_batch_size=16)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text")
    max_tokens: int = Field(default=64, ge=1, le=512)
    request_id: Optional[str] = Field(default=None)


class GenerateResponse(BaseModel):
    request_id: str
    tokens_generated: int
    ttft_ms: Optional[float]
    total_latency_ms: Optional[float]


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    engine = _engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    rid = req.request_id or str(uuid.uuid4())
    prompt_tokens = [ord(c) % 50256 for c in req.prompt[:512]]

    sequence = SequenceRequest(
        request_id=rid,
        prompt_tokens=prompt_tokens,
        max_output_tokens=req.max_tokens,
        arrival_time=time.monotonic(),
    )
    engine.add_request(sequence)

    async with _engine_lock:
        while not sequence.is_finished:
            await engine.step()

    return GenerateResponse(
        request_id=rid,
        tokens_generated=len(sequence.generated_tokens),
        ttft_ms=round(sequence.ttft() * 1000, 2) if sequence.ttft() else None,
        total_latency_ms=round(sequence.total_latency() * 1000, 2) if sequence.total_latency() else None,
    )


@app.get("/metrics")
async def metrics() -> JSONResponse:
    return JSONResponse(_engine.metrics.summary())


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "scheduler": _engine.scheduler_stats(),
        "memory": _engine.memory_stats(),
    })
