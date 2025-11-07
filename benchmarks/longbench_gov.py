"""LongBench government report summarization benchmark tasks."""

from typing import Iterable
from .base import Task, Sample, WorkloadOpts
from .evaluators import RougeLEvaluator
from .loaders import _load_longbench_manual
from .utils import _to_text, require


class LongBenchGov(Task):
    """LongBench summarization subsets -> ROUGE-L"""
    name="longbench_gov"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = RougeLEvaluator()
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        # LongBench dataset scripts no longer supported, use manual loader
        ds = _load_longbench_manual("gov_report", self.opts.cache_dir)
        for i,row in enumerate(ds):
            doc=_to_text(row.get("context") or row.get("input"))
            # LongBench stores answer/summary in "answers" field (list)
            answers = row.get("answers", [])
            summ = _to_text(answers[0] if answers else "")
            if not doc or not summ: continue
            prompt=f"Summarize the following government report succinctly:\n\n{doc}"
            yield Sample(id=f"lbg-{i}", prompt=prompt, messages=[{"role":"user","content":prompt}], references=[summ], extra={})


class LongBenchGovBeam(Task):
    """LongBench gov_report with beam search for summarization."""
    name="longbench_gov"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = RougeLEvaluator()
    def load(self)->Iterable[Sample]:
        ds = _load_longbench_manual("gov_report", self.opts.cache_dir)
        for i,row in enumerate(ds):
            doc=_to_text(row.get("context") or row.get("input"))
            answers = row.get("answers", [])
            summ = _to_text(answers[0] if answers else "")
            if not doc or not summ: continue
            prompt=f"Summarize the following government report succinctly:\n\n{doc}"
            yield Sample(id=f"beam-lbg-{i}", prompt=prompt, messages=[{"role":"user","content":prompt}], references=[summ], extra={})

