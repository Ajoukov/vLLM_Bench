"""HumanEval code completion benchmark."""

from typing import Iterable

from .base import Sample, Task, WorkloadOpts
from .evaluators import HumanEvalEvaluator
from .loaders import _hf_load
from .utils import _to_text, require


class HumanEval(Task):
    """HumanEval code completion -> pass@k via tests"""

    name = "humaneval"

    def __init__(self, opts: WorkloadOpts):
        super().__init__(opts)
        ks = (
            (1, 5, 10)
            if self.opts.best_of >= 10
            else (1, 5)
            if self.opts.best_of >= 5
            else (1,)
        )
        self.evaluator = HumanEvalEvaluator(ks=ks)

    def load(self) -> Iterable[Sample]:
        require(["datasets"])
        ds = _hf_load("openai_humaneval", None, "test", self.opts.cache_dir)
        for i, row in enumerate(ds):
            task_id = _to_text(row.get("task_id"))
            prompt = _to_text(row.get("prompt"))
            tests = _to_text(row.get("test") or row.get("tests"))
            if not prompt or not tests:
                continue
            # Use /completions with prompt as prefix; model returns code continuation
            yield Sample(
                id=f"humaneval-{i}",
                prompt=prompt,
                messages=None,
                references=[],
                extra={"tests": tests, "task_id": task_id, "prompt_leading": prompt},
            )
