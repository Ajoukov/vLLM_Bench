"""ShareGPT conversation benchmarks."""

from typing import Iterable
from .base import Task, Sample, WorkloadOpts
from .evaluators import EMF1Evaluator
from .loaders import _load_sharegpt_manual
from .utils import _to_text, require


class ShareGPTPrefix(Task):
    """ShareGPT conversations with shared system prompt for prefix caching."""
    name="sharegpt"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        # Use manual loader - ShareGPT no longer has standard dataset config
        ds = _load_sharegpt_manual(self.opts.cache_dir)
        shared_sys = "You are a helpful, knowledgeable, and friendly assistant. Provide clear and concise responses."
        for i,row in enumerate(ds):
            rec = self._norm(row)
            q = rec.get("prompt") or ""
            a = rec.get("answer") or ""
            if not q or not a: continue
            msgs=[{"role":"system","content":shared_sys},
                  {"role":"user","content":q}]
            yield Sample(id=f"sharegpt-prefix-{i}", prompt=f"{shared_sys}\n{q}", messages=msgs, references=[a], extra={"shared_sys":shared_sys})
    def _norm(self,row):
        conv=row.get("conversations") or row.get("messages")
        if conv and isinstance(conv, list):
            q=_to_text(next((m.get("value") or m.get("content") for m in conv if (m.get('from') in ('human','user') or m.get('role')=='user')), ""))
            a=_to_text(next((m.get("value") or m.get("content") for m in conv if (m.get('from') in ('gpt','assistant') or m.get('role')=='assistant')), ""))
            return {"prompt":q,"answer":a}
        return {"prompt":_to_text(row.get("instruction")),"answer":_to_text(row.get("output"))}


class ShareGPTChat(Task):
    """ShareGPT conversational format for chatbot evaluation."""
    name="sharegpt"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        # Use manual loader - ShareGPT no longer has standard dataset config
        ds = _load_sharegpt_manual(self.opts.cache_dir)
        for i,row in enumerate(ds):
            rec = self._norm(row)
            q = rec.get("prompt") or ""
            a = rec.get("answer") or ""
            if not q or not a: continue
            msgs=[{"role":"user","content":q}]
            yield Sample(id=f"sharegpt-chat-{i}", prompt=q, messages=msgs, references=[a], extra={})
    def _norm(self,row):
        conv=row.get("conversations") or row.get("messages")
        if conv and isinstance(conv, list):
            q=_to_text(next((m.get("value") or m.get("content") for m in conv if (m.get('from') in ('human','user') or m.get('role')=='user')), ""))
            a=_to_text(next((m.get("value") or m.get("content") for m in conv if (m.get('from') in ('gpt','assistant') or m.get('role')=='assistant')), ""))
            return {"prompt":q,"answer":a}
        return {"prompt":_to_text(row.get("instruction")),"answer":_to_text(row.get("output"))}

