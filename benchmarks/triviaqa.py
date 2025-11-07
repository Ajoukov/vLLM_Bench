"""TriviaQA benchmark tasks."""

from typing import Iterable
from .base import Task, Sample, WorkloadOpts
from .evaluators import EMF1Evaluator
from .loaders import _hf_load
from .utils import _to_text, _short_answer


class TriviaQATask(Task):
    """TriviaQA (rc) validation -> EM/F1"""
    name="triviaqa"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator()
    def load(self)->Iterable[Sample]:
        ds=_hf_load("trivia_qa","rc","validation",self.opts.cache_dir)
        for i,row in enumerate(ds):
            q=_to_text(row.get("question"))
            aobj=row.get("answer") or {}
            golds=[]
            if isinstance(aobj, dict):
                golds.append(_to_text(aobj.get("value")))
                golds += [x for x in (aobj.get("aliases") or []) if x]
            golds=[g for g in golds if g]
            if not q or not golds: continue
            prompt = (
                    "Answer the following question with a short phrase or name ONLY.\n"
                    "Do not include explanations.\n\n"
                    f"Question: {q}\nFinal answer:"
                    )
            yield Sample(id=f"tqa-{i}", prompt=prompt,
                         messages=[{"role":"user","content":prompt}],
                         references=golds, extra={})
    def postprocess_pred(self, pred: str, sample: "Sample") -> str:
        return _short_answer(pred)


class TriviaQASPSR(Task):
    """TriviaQA in single-prompt format."""
    name="triviaqa"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator()
    def load(self)->Iterable[Sample]:
        ds=_hf_load("trivia_qa","rc","validation",self.opts.cache_dir)
        for i,row in enumerate(ds):
            q=_to_text(row.get("question"))
            aobj=row.get("answer") or {}
            golds=[]
            if isinstance(aobj, dict):
                golds.append(_to_text(aobj.get("value")))
                golds += [x for x in (aobj.get("aliases") or []) if x]
            golds=[g for g in golds if g]
            if not q or not golds: continue
            prompt = f"Answer the following question with a short phrase or name ONLY.\n\nQuestion: {q}\nFinal answer:"
            yield Sample(id=f"spsr-tqa-{i}", prompt=prompt,
                         messages=[{"role":"user","content":prompt}],
                         references=golds, extra={})
    def postprocess_pred(self, pred: str, sample: "Sample") -> str:
        return _short_answer(pred)


class TriviaQABeam(Task):
    """TriviaQA with beam search."""
    name="triviaqa"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator()
    def load(self)->Iterable[Sample]:
        ds=_hf_load("trivia_qa","rc","validation",self.opts.cache_dir)
        for i,row in enumerate(ds):
            q=_to_text(row.get("question"))
            aobj=row.get("answer") or {}
            golds=[]
            if isinstance(aobj, dict):
                golds.append(_to_text(aobj.get("value")))
                golds += [x for x in (aobj.get("aliases") or []) if x]
            golds=[g for g in golds if g]
            if not q or not golds: continue
            prompt = f"Answer the following question with a short phrase or name ONLY.\n\nQuestion: {q}\nFinal answer:"
            yield Sample(id=f"beam-tqa-{i}", prompt=prompt,
                         messages=[{"role":"user","content":prompt}],
                         references=golds, extra={})
    def postprocess_pred(self, pred: str, sample: "Sample") -> str:
        return _short_answer(pred)

