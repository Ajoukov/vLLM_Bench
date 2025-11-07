"""NarrativeQA benchmark tasks."""

from typing import Iterable
from .base import Task, Sample, WorkloadOpts
from .evaluators import EMF1Evaluator
from .loaders import _hf_load
from .utils import _to_text, _short_answer


class NarrativeQATask(Task):
    """NarrativeQA (summary) validation -> F1 on abstractive answers"""
    name="narrativeqa"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="f1_only")
    def load(self)->Iterable[Sample]:
        ds=_hf_load("narrativeqa", None, "validation", self.opts.cache_dir)
        for i,row in enumerate(ds):
            q=_to_text(row.get("question")); ans=_to_text(row.get("answers"))
            if not q or not ans: continue
            prompt = (
                    "Provide a concise answer (a few words) to the question. No explanation.\n\n"
                    f"Question: {q}\nFinal answer:"
                    )
            yield Sample(id=f"nqa-{i}", prompt=prompt,
                         messages=[{"role":"user","content":prompt}],
                         references=[ans], extra={})
    def postprocess_pred(self, pred: str, sample: "Sample") -> str:
        return _short_answer(pred)


class NarrativeQASPSR(Task):
    """NarrativeQA in single-prompt format."""
    name="narrativeqa"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="f1_only")
    def load(self)->Iterable[Sample]:
        ds=_hf_load("narrativeqa", None, "validation", self.opts.cache_dir)
        for i,row in enumerate(ds):
            q=_to_text(row.get("question")); ans=_to_text(row.get("answers"))
            if not q or not ans: continue
            prompt = f"Provide a concise answer (a few words) to the question. No explanation.\n\nQuestion: {q}\nFinal answer:"
            yield Sample(id=f"spsr-nqa-{i}", prompt=prompt,
                         messages=[{"role":"user","content":prompt}],
                         references=[ans], extra={})
    def postprocess_pred(self, pred: str, sample: "Sample") -> str:
        return _short_answer(pred)


class NarrativeQABeam(Task):
    """NarrativeQA with beam search."""
    name="narrativeqa"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="f1_only")
    def load(self)->Iterable[Sample]:
        ds=_hf_load("narrativeqa", None, "validation", self.opts.cache_dir)
        for i,row in enumerate(ds):
            q=_to_text(row.get("question")); ans=_to_text(row.get("answers"))
            if not q or not ans: continue
            prompt = f"Provide a concise answer (a few words) to the question. No explanation.\n\nQuestion: {q}\nFinal answer:"
            yield Sample(id=f"beam-nqa-{i}", prompt=prompt,
                         messages=[{"role":"user","content":prompt}],
                         references=[ans], extra={})
    def postprocess_pred(self, pred: str, sample: "Sample") -> str:
        return _short_answer(pred)

