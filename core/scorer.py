"""
VietBinoculars Scorer.
Loads the PhoGPT-4B model pair and computes Binoculars scores for text chunks.
"""

from typing import Union

import torch

from config.settings import (
    DEVICE_1,
    DEVICE_2,
    HF_TOKEN,
    MAX_MODEL_TOKENS,
    SCORER_BATCH_SIZE,
    USE_BFLOAT16,
)
from schemas.models import ChunkDetail
from core.metrics import perplexity, entropy

# Disable gradient computation globally for inference
torch.set_grad_enabled(False)


class VietBinocularsScorer:
    """Binoculars AI-text scorer adapted for Vietnamese using PhoGPT-4B pair."""

    def __init__(
        self,
        observer_name: str,
        performer_name: str,
        device1: str = DEVICE_1,
        device2: str = DEVICE_2,
        use_bf16: bool = USE_BFLOAT16,
        max_tokens: int = MAX_MODEL_TOKENS,
    ):
        """Initialize scorer by loading both models onto their respective GPUs."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_kwargs = dict(trust_remote_code=True, token=HF_TOKEN)
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Load shared tokenizer from observer model
        self.tokenizer = AutoTokenizer.from_pretrained(observer_name, **hf_kwargs)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load observer model (base) onto device1
        model_kwargs = dict(**hf_kwargs, torch_dtype=dtype)
        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_name, device_map={"": device1}, **model_kwargs
        ).eval()

        # Load performer model (chat-tuned) onto device2
        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_name, device_map={"": device2}, **model_kwargs
        ).eval()

        self.device1 = device1
        self.device2 = device2
        self.max_tokens = max_tokens

    def _tokenize(self, batch: list[str]):
        """Tokenize a batch of text strings for model input."""
        return self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True,
            max_length=self.max_tokens,
            return_token_type_ids=False,
        ).to(self.device1)

    @torch.inference_mode()
    def compute_score(self, text: Union[str, list[str]]) -> Union[float, list[float]]:
        """Compute VietBinoculars score(s) for the given text(s)."""
        batch = [text] if isinstance(text, str) else text
        enc = self._tokenize(batch)

        # Forward pass through both models
        obs_logits = self.observer(**enc.to(self.device1)).logits
        perf_logits = self.performer(**enc.to(self.device2)).logits

        if self.device1 != "cpu":
            torch.cuda.synchronize()

        # Compute PPL and X-PPL
        ppl = perplexity(enc, perf_logits)
        x_ppl = entropy(
            obs_logits.to(self.device1),
            perf_logits.to(self.device1),
            enc.to(self.device1),
            self.tokenizer.pad_token_id,
        )

        scores = (ppl / x_ppl).tolist()
        return scores[0] if isinstance(text, str) else scores

    def score_chunks(self, chunks: list[ChunkDetail]) -> list[ChunkDetail]:
        """Score all chunks in batches for memory-efficient processing."""
        texts = [c.text for c in chunks]
        all_scores: list[float] = []

        for i in range(0, len(texts), SCORER_BATCH_SIZE):
            batch = texts[i: i + SCORER_BATCH_SIZE]
            batch_scores = self.compute_score(batch)
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            all_scores.extend(batch_scores)

        for chunk, score in zip(chunks, all_scores):
            chunk.score = round(float(score), 6)

        return chunks
