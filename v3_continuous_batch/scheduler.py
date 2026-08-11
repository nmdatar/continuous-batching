from __future__ import annotations

import asyncio
import os
from contextlib import suppress

from common.model import TinyModel
from common.types import GenerationJob


class ContinuousBatchScheduler:
    """Refill free batch slots between token-generation iterations."""

    def __init__(self, model: TinyModel) -> None:
        self.model = model
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "8"))
        self.queue: asyncio.Queue[GenerationJob] = asyncio.Queue()
        self.active_count = 0
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="continuous-batch-scheduler")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def submit(self, job: GenerationJob) -> None:
        await self.queue.put(job)

    async def _run(self) -> None:
        active: list[GenerationJob] = []
        while True:
            if not active:
                active.append(await self.queue.get())
            self._refill(active)
            active = [job for job in active if not self._drop_cancelled(job)]
            if not active:
                continue

            self.active_count = len(active)
            token_ids = await asyncio.to_thread(self.model.next_token_ids, active)
            survivors: list[GenerationJob] = []
            for job, token_id in zip(active, token_ids):
                job.emit_token(token_id, self.model.decode_token(token_id))
                if token_id == self.model.eos_token_id:
                    job.finish("eos")
                    self.queue.task_done()
                elif job.generated_tokens >= job.max_new_tokens:
                    job.finish("length")
                    self.queue.task_done()
                else:
                    survivors.append(job)
            active = survivors
            self.active_count = len(active)
            # The next loop refills every newly freed slot before decoding again.

    def _refill(self, active: list[GenerationJob]) -> None:
        while len(active) < self.max_batch_size:
            try:
                active.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return

    def _drop_cancelled(self, job: GenerationJob) -> bool:
        if job.cancelled:
            self.queue.task_done()
            return True
        return False
