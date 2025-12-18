"""Wikitext language modeling benchmark."""

from typing import Iterable

from .base import Sample, Task, WorkloadOpts
from .evaluators import EMF1Evaluator
from .loaders import _hf_load
from .utils import _to_text, require


class WikitextSPSR(Task):
    """Wikitext language modeling perplexity (simple eval: token overlap)."""

    name = "wikitext"

    def __init__(self, opts: WorkloadOpts):
        super().__init__(opts)
        self.evaluator = EMF1Evaluator(name="overlap")

    def load(self) -> Iterable[Sample]:
        require(["datasets"])
        ds = _hf_load("wikitext", "wikitext-2-raw-v1", "test", self.opts.cache_dir)
        # WikiText is usually used for perplexity, but for a generate-eval we can sample prompts
        # Take first N words as prompt, rest as reference
        for i, row in enumerate(ds):
            text = _to_text(row.get("text"))
            if not text or len(text.split()) < 20:
                continue
            words = text.split()
            mid = len(words) // 2
            prompt_text = " ".join(words[:mid])
            ref_text = " ".join(words[mid:])
            prompt = f"Continue the following text:\n\n{prompt_text}"
            yield Sample(
                id=f"wikitext-{i}",
                prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                references=[ref_text],
                extra={},
            )
