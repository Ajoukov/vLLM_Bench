"""LongBench meeting summarization benchmark tasks."""

from typing import Iterable

from .base import Sample, Task, WorkloadOpts
from .evaluators import RougeLEvaluator
from .loaders import _load_longbench_manual
from .utils import _to_text


class LongBenchQMSum(Task):
    """LongBench QMSum meeting summarization -> ROUGE-L"""

    name = "longbench_qmsum"

    def __init__(self, opts: WorkloadOpts):
        super().__init__(opts)
        self.evaluator = RougeLEvaluator()

    def load(self) -> Iterable[Sample]:
        # LongBench dataset scripts no longer supported, use manual loader
        ds = _load_longbench_manual("qmsum", self.opts.cache_dir)
        for i, row in enumerate(ds):
            doc = _to_text(row.get("context") or row.get("input"))
            # LongBench stores answer/summary in "answers" field (list)
            answers = row.get("answers", [])
            summ = _to_text(answers[0] if answers else "")
            if not doc or not summ:
                continue
            prompt = f"Generate a concise meeting summary:\n\n{doc}"
            yield Sample(
                id=f"lbq-{i}",
                prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                references=[summ],
                extra={},
            )


class LongBenchQMSumBeam(Task):
    """LongBench qmsum with beam search."""

    name = "longbench_qmsum"

    def __init__(self, opts: WorkloadOpts):
        super().__init__(opts)
        self.evaluator = RougeLEvaluator()

    def load(self) -> Iterable[Sample]:
        ds = _load_longbench_manual("qmsum", self.opts.cache_dir)
        for i, row in enumerate(ds):
            doc = _to_text(row.get("context") or row.get("input"))
            answers = row.get("answers", [])
            summ = _to_text(answers[0] if answers else "")
            if not doc or not summ:
                continue
            prompt = f"Generate a concise meeting summary:\n\n{doc}"
            yield Sample(
                id=f"beam-lbq-{i}",
                prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                references=[summ],
                extra={},
            )
