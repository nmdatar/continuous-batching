from __future__ import annotations

import asyncio
from contextlib import suppress

from common.model import TinyModel
from common.types import GenerationJob


class SerialScheduler:
    """Finish one request completely before admitting the next one."""

    def __init__(self, model: TinyModel) -> None:
        self.model = model
        self.queue: asyncio.Queue[GenerationJob] = asyncio.Queue()
        self.active_count = 0
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="serial-scheduler")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def submit(self, job: GenerationJob) -> None:
        await self.queue.put(job)

    async def _run(self) -> None:
        while True:
            job = await self.queue.get()
            self.active_count = 1
            try:
                await self._decode(job)
            finally:
                self.active_count = 0
                self.queue.task_done()

    async def _decode(self, job: GenerationJob) -> None:
        while not job.cancelled:
            token_id = (await asyncio.to_thread(self.model.next_token_ids, [job]))[0]
            job.emit_token(token_id, self.model.decode_token(token_id))
            if token_id == self.model.eos_token_id:
                job.finish("eos")
                return
            if job.generated_tokens >= job.max_new_tokens:
                job.finish("length")
                return
