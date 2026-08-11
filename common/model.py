from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.types import GenerationJob


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TinyModel:
    """A deliberately simple greedy, full-context decoder."""

    def __init__(self) -> None:
        model_id = os.getenv("MODEL_ID", "sshleifer/tiny-gpt2")
        self.device = torch.device(os.getenv("DEVICE", _default_device()))
        requested_context = int(os.getenv("MAX_CONTEXT_TOKENS", "512"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding makes the final position the next-token position for every row.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        model_context = getattr(self.model.config, "max_position_embeddings", requested_context)
        self.max_context_tokens = min(requested_context, model_context)
        self.eos_token_id = self.tokenizer.eos_token_id

    def prepare(self, job: GenerationJob) -> None:
        room_for_prompt = max(1, self.max_context_tokens - job.max_new_tokens)
        ids = self.tokenizer.encode(job.prompt, add_special_tokens=False)
        job.prompt_ids = ids[-room_for_prompt:] or [self.eos_token_id]

    @torch.inference_mode()
    def next_token_ids(self, jobs: Sequence[GenerationJob]) -> list[int]:
        sequences = [job.prompt_ids + job.output_ids for job in jobs]
        width = max(map(len, sequences))
        pad_id = self.tokenizer.pad_token_id
        padded = [[pad_id] * (width - len(ids)) + ids for ids in sequences]
        masks = [[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits[:, -1, :].argmax(dim=-1).tolist()

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=True)
