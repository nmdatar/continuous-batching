from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GenerationJob:
    prompt: str
    max_new_tokens: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    prompt_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    events: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=time.perf_counter)
    cancelled: bool = False

    @property
    def generated_tokens(self) -> int:
        return len(self.output_ids)

    def emit_token(self, token_id: int, text: str) -> None:
        self.output_ids.append(token_id)
        self.events.put_nowait(
            {
                "type": "token",
                "request_id": self.id,
                "index": self.generated_tokens - 1,
                "token_id": token_id,
                "text": text,
            }
        )

    def finish(self, reason: str) -> None:
        self.events.put_nowait(
            {
                "type": "done",
                "request_id": self.id,
                "finish_reason": reason,
                "generated_tokens": self.generated_tokens,
                "server_latency_ms": (time.perf_counter() - self.created_at) * 1000,
            }
        )


class SchedulerProtocol:
    model: Any
    queue: asyncio.Queue[GenerationJob]
    active_count: int

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def submit(self, job: GenerationJob) -> None: ...
