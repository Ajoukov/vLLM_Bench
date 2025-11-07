"""Alpaca instruction-following benchmark."""

from typing import Iterable
from .base import Task, Sample, WorkloadOpts
from .evaluators import EMF1Evaluator
from .loaders import _hf_load
from .utils import _to_text, require


class AlpacaSPSR(Task):
    """Alpaca instruction-following in single-turn format."""
    name="alpaca"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        ds=_hf_load("yahma/alpaca-cleaned", None, "train", self.opts.cache_dir)
        for i,row in enumerate(ds):
            inst=_to_text(row.get("instruction"))
            inp=_to_text(row.get("input"))
            out=_to_text(row.get("output"))
            if not inst or not out: continue
            prompt = inst if not inp else f"{inst}\n\nInput: {inp}"
            yield Sample(id=f"alpaca-{i}", prompt=prompt,
                         messages=[{"role":"user","content":prompt}],
                         references=[out], extra={})

