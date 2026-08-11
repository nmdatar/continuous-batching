from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from common.model import TinyModel
from common.types import GenerationJob, SchedulerProtocol


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = Field(default=32, ge=1, le=256)


def create_app(
    name: str, scheduler_factory: Callable[[TinyModel], SchedulerProtocol]
) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        model = TinyModel()
        scheduler = scheduler_factory(model)
        state["scheduler"] = scheduler
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.stop()

    app = FastAPI(title=name, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "scheduler": name}

    @app.get("/stats")
    async def stats() -> dict[str, int | str]:
        scheduler = state["scheduler"]
        return {
            "scheduler": name,
            "queued": scheduler.queue.qsize(),
            "active": scheduler.active_count,
        }

    @app.post("/generate")
    async def generate(body: GenerateBody, request: Request) -> StreamingResponse:
        scheduler = state["scheduler"]
        job = GenerationJob(body.prompt, body.max_new_tokens)
        scheduler.model.prepare(job)
        await scheduler.submit(job)

        async def stream():
            try:
                while True:
                    event = await job.events.get()
                    yield json.dumps(event, separators=(",", ":")) + "\n"
                    if event["type"] == "done":
                        return
                    if await request.is_disconnected():
                        job.cancelled = True
                        return
            finally:
                # Cancellation is observed at the scheduler's next decode boundary.
                if await request.is_disconnected():
                    job.cancelled = True

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app
