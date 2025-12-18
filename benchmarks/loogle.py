"""LooGLE legal text summarization benchmark."""

from typing import Iterable

from .base import Sample, Task, WorkloadOpts
from .evaluators import RougeLEvaluator
from .loaders import _hf_load
from .utils import _to_text


class LooGLE(Task):
    """LooGLE summarization -> ROUGE-L"""

    name = "loogle"

    def __init__(self, opts: WorkloadOpts):
        super().__init__(opts)
        self.evaluator = RougeLEvaluator()

    def load(self) -> Iterable[Sample]:
        # Load LooGLE dataset - fail if not available
        ds = _hf_load("bigai-nlco/LooGLE", "summarization", "test", self.opts.cache_dir)
        for i, row in enumerate(ds):
            doc = _to_text(
                row.get("document")
                or row.get("context")
                or row.get("text")
                or row.get("passage")
            )
            summ = _to_text(
                row.get("summary") or row.get("target") or row.get("answer")
            )
            if not doc or not summ:
                continue
            prompt = f"Summarize the following legal text concisely:\n\n{doc}"
            yield Sample(
                id=f"loogle-summarization-{i}",
                prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                references=[summ],
                extra={"subset": "summarization"},
            )
