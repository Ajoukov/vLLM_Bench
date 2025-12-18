"""KV probe benchmark for prefix caching."""

from typing import Iterable

from .base import Sample, Task, WorkloadOpts
from .evaluators import EMF1Evaluator
from .loaders import _hf_load
from .utils import _to_text, require


class PrefixKV(Task):
    """Prefix-KV probe (shared system+user prefix) -> no automatic metric; logs prefill stats"""

    name = "prefix_kv"

    def __init__(self, opts: WorkloadOpts):
        super().__init__(opts)
        self.evaluator = EMF1Evaluator(name="null")  # placeholder

    def load(self) -> Iterable[Sample]:
        require(["datasets"])
        # Use a small ShareGPT-like set; fallback to alpaca
        try:
            ds = _hf_load(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                "ShareGPT_V4.3_unfiltered_cleaned_split",
                "train",
                self.opts.cache_dir,
            )
            it = (self._norm(row) for row in ds)
        except Exception:
            alp = _hf_load("yahma/alpaca-cleaned", None, "train", self.opts.cache_dir)
            it = (
                {
                    "prompt": _to_text(r["instruction"])
                    + ("\n\nInput: " + _to_text(r["input"]) if r.get("input") else ""),
                    "answer": _to_text(r["output"]),
                }
                for r in alp
            )
        shared_sys = "You are a concise assistant. Always answer directly."
        shared_user = "Common preface: Use bullet points if appropriate. Now answer:"
        for i, rec in enumerate(it):
            q = rec.get("prompt") or rec.get("question") or ""
            a = rec.get("answer") or ""
            if not q or not a:
                continue
            msgs = [
                {"role": "system", "content": shared_sys},
                {"role": "user", "content": shared_user},
                {"role": "user", "content": q},
            ]
            extra = {"shared_sys": shared_sys, "shared_user": shared_user}
            yield Sample(
                id=f"kv-{i}",
                prompt=f"{shared_sys}\n{shared_user}\n{q}",
                messages=msgs,
                references=[a],
                extra=extra,
            )

    def _norm(self, row):
        conv = row.get("conversations") or row.get("messages")
        if conv and isinstance(conv, list):
            q = _to_text(
                next(
                    (
                        m.get("value") or m.get("content")
                        for m in conv
                        if (
                            m.get("from") in ("human", "user")
                            or m.get("role") == "user"
                        )
                    ),
                    "",
                )
            )
            a = _to_text(
                next(
                    (
                        m.get("value") or m.get("content")
                        for m in conv
                        if (
                            m.get("from") in ("gpt", "assistant")
                            or m.get("role") == "assistant"
                        )
                    ),
                    "",
                )
            )
            return {"prompt": q, "answer": a}
        return {
            "prompt": _to_text(row.get("instruction")),
            "answer": _to_text(row.get("output")),
        }
