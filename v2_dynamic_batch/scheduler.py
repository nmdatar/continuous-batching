from __future__ import annotations

import asyncio
import os
from contextlib import suppress

from common.model import TinyModel
from common.types import GenerationJob


class DynamicBatchScheduler:
    """Collect briefly, then decode that fixed cohort until all rows finish."""

    def __init__(self, model: TinyModel) -> None:
        self.model = model
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "8"))
        self.batch_window = int(os.getenv("BATCH_WINDOW_MS", "10")) / 1000
        self.queue: asyncio.Queue[GenerationJob] = asyncio.Queue()
        self.active_count = 0
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="dynamic-batch-scheduler")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def submit(self, job: GenerationJob) -> None:
        await self.queue.put(job)

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_window
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except TimeoutError:
                    break

            self.active_count = len(batch)
            try:
                await self._decode_fixed_batch(batch)
            finally:
                self.active_count = 0
                for _ in batch:
                    self.queue.task_done()

    async def _decode_fixed_batch(self, batch: list[GenerationJob]) -> None:
        active = batch.copy()
        while active:
            active = [job for job in active if not job.cancelled]
            if not active:
                return
            token_ids = await asyncio.to_thread(self.model.next_token_ids, active)
            survivors: list[GenerationJob] = []
            for job, token_id in zip(active, token_ids):
                job.emit_token(token_id, self.model.decode_token(token_id))
                if token_id == self.model.eos_token_id:
                    job.finish("eos")
                elif job.generated_tokens >= job.max_new_tokens:
                    job.finish("length")
                else:
                    survivors.append(job)
            active = survivors
            self.active_count = len(active)
