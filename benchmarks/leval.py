"""LEval long-document comprehension benchmark."""

from typing import Iterable

from .base import Sample, Task, WorkloadOpts
from .evaluators import RougeLEvaluator
from .loaders import _load_leval_manual
from .utils import _to_text, require


class LevalPrefix(Task):
    """LongEval (LEval) dataset with shared instruction prefix."""

    name = "leval"

    def __init__(self, opts: WorkloadOpts):
        super().__init__(opts)
        self.evaluator = RougeLEvaluator()

    def load(self) -> Iterable[Sample]:
        require(["datasets"])
        # Use manual loader - LEval has dataset script issues
        ds = _load_leval_manual("narrative_qa", self.opts.cache_dir)
        shared_sys = "You are an expert at analyzing long documents. Answer comprehensively based on the provided context."
        for i, row in enumerate(ds):
            # LEval format: 'input' (context), 'instructions' (questions list), 'outputs' (answers list)
            doc = _to_text(row.get("input") or "")
            questions = row.get("instructions", [])
            outputs = row.get("outputs", [])

            if not doc or not questions or not outputs:
                continue
            # Use first question/answer pair
            q = _to_text(
                questions[0] if isinstance(questions, list) and questions else questions
            )
            ans = _to_text(
                outputs[0] if isinstance(outputs, list) and outputs else outputs
            )

            prompt = f"{q}\n\nContext:\n{doc}"
            msgs = [
                {"role": "system", "content": shared_sys},
                {"role": "user", "content": prompt},
            ]
            yield Sample(
                id=f"leval-{i}",
                prompt=f"{shared_sys}\n{prompt}",
                messages=msgs,
                references=[ans],
                extra={"shared_sys": shared_sys},
            )
