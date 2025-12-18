"""Base classes and data types for benchmark tasks."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class WorkloadOpts:
    endpoint: str
    model: str
    api_kind: str
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    qps: float
    limit: int
    out_dir: str
    cache_dir: str
    system_prompt: Optional[str]
    use_beam_search: bool
    num_beams: int
    best_of: int
    seed: int
    stream: bool
    metrics_tokenizer: str
    metrics_tokenizer_model: Optional[str]
    verbose: bool
    timeout_s: float
    output_format: Optional[str]  # json, jsonl, csv, parquet, etc.
    output_file: Optional[str]  # path to output file
    # LMCache configuration
    lmcache_enabled: bool = False
    lmcache_chunk_size: int = 256
    lmcache_local_cpu: bool = True
    lmcache_max_local_cpu_size: int = 20
    lmcache_backend: str = "local"


@dataclass
class Sample:
    id: str
    prompt: str  # for /completions
    messages: Optional[List[dict]]  # for /chat/completions
    references: List[str]  # gold answers or target summaries
    extra: Dict[str, Any]  # task-specific


class Evaluator:
    name: str

    def update(self, pred: str, sample: Sample): ...
    def finalize(self) -> Dict[str, Any]: ...


class Task:
    name: str
    evaluator: Evaluator

    def __init__(self, opts: WorkloadOpts):
        self.opts = opts

    def load(self) -> Iterable[Sample]: ...
    def build_payload(self, sample: Sample) -> Dict[str, Any]:
        if self.opts.api_kind == "chat":
            msgs = sample.messages or [{"role": "user", "content": sample.prompt}]
            if self.opts.system_prompt:
                msgs = [{"role": "system", "content": self.opts.system_prompt}] + msgs
            return {
                "model": self.opts.model,
                "messages": msgs,
                "max_tokens": self.opts.max_tokens,
            }
        else:
            prompt = sample.prompt
            if self.opts.system_prompt:
                prompt = (
                    f"System: {self.opts.system_prompt}\nUser: {prompt}\nAssistant:"
                )
            return {
                "model": self.opts.model,
                "prompt": prompt,
                "max_tokens": self.opts.max_tokens,
            }

    def decoding_overrides(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.opts.use_beam_search:
            width = max(2, int(self.opts.num_beams))
            payload.update(
                {
                    "use_beam_search": True,
                    "n": width,
                    "best_of": max(self.opts.best_of, width),
                    "temperature": 0,
                    "top_p": 1.0,
                }
            )
        else:
            payload.update(
                {
                    "temperature": float(self.opts.temperature),
                    "top_p": float(self.opts.top_p),
                }
            )
            if self.opts.top_k > 0:
                payload["top_k"] = int(self.opts.top_k)
        if self.opts.seed is not None:
            payload["seed"] = int(self.opts.seed)
        return payload
