#!/usr/bin/env python3
# pylint: disable=too-many-lines
"""
Industry-grade LLM benchmarking harness over vLLM OpenAI routes.

Key features
- Comprehensive workload coverage across multiple categories
- Canonical dataset adapters with HuggingFace Datasets integration
- Advanced decoding control: sampling/beam search, best_of, seed, temperature, top_p, max_tokens
- Detailed measurements: TTFT (streaming), latency, tokens, TPOT
- Tokenizer support: served-model -> tiktoken -> whitespace fallback
- Full reproducibility: run manifests with model info, environment, seed, dataset revision
- Professional outputs: JSONL per-task + aggregate metrics JSON

Categories and Workloads:
  spsr (Single-Prompt-Single-Response):
    - alpaca: Instruction-following with Alpaca dataset
    - triviaqa: TriviaQA question-answering (EM/F1)
    - narrativeqa: NarrativeQA reading comprehension (F1)
    - wikitext: Language modeling continuation task

  beam (Beam Search Evaluation):
    - longbench_gov: Government report summarization (ROUGE-L)
    - longbench_qmsum: Meeting summarization (ROUGE-L)
    - narrativeqa: NarrativeQA with beam search (F1)
    - triviaqa: TriviaQA with beam search (EM/F1)

  prefix (Shared Prefix / Prefix Caching):
    - kvprobe: Prefix caching probe with shared system/user prompts
    - sharegpt: ShareGPT conversations with shared system prompt
    - leval: LEval long-document comprehension (ROUGE-L)
    - longchat: LongChat multi-turn conversations

  chat (Chatbot Evaluation):
    - sharegpt: ShareGPT conversational format
    - longchat: LongChat dialogue evaluation

  qa (Question Answering):
    - triviaqa: TriviaQA reading comprehension (EM/F1)
    - narrativeqa: NarrativeQA abstractive QA (F1)

  summ (Summarization):
    - longbench_gov: Government report summarization (ROUGE-L)
    - longbench_qmsum: Meeting summarization (ROUGE-L)
    - loogle: LooGLE legal text summarization (ROUGE-L)

  code (Code Completion):
    - humaneval: HumanEval Python code completion (pass@k)

CLI
  ./bench.py list
  ./bench.py run config.json [category] [workload]

Examples
  # List all available workloads
  ./bench.py list
  
  # Run specific workload
  ./bench.py run config.json spsr alpaca
  
  # Run all enabled workloads in a category
  ./bench.py run config.json beam
  
  # Run all enabled workloads
  ./bench.py run config.json
"""

import argparse, json, os, re, sys, time, uuid, math, random, shutil, tempfile, subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
from contextlib import contextmanager

import requests

try:
    import datasets as hfds
except Exception:
    hfds = None
try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None
try:
    import tiktoken
except Exception:
    tiktoken = None
try:
    import orjson
    def jdump(obj): return orjson.dumps(obj).decode("utf-8")
except Exception:
    def jdump(obj): return json.dumps(obj, ensure_ascii=False)

# ---------------- Registry ----------------

class Workload:
    def __init__(self, name: str, category: str, task_cls: "Task"):
        self.name = name; self.category = category; self.task_cls = task_cls

_REGISTRY: Dict[str, List[Workload]] = {}

def register(category: str, name: str):
    def deco(task_cls):
        _REGISTRY.setdefault(category, []).append(Workload(name, category, task_cls))
        return task_cls
    return deco

def list_all() -> Dict[str, List[str]]:
    return {cat: [w.name for w in wls] for cat, wls in _REGISTRY.items()}

# ---------------- Utils ----------------

def require(pkgs: List[str]):
    missing = [p for p in pkgs if p == "datasets" and hfds is None or p == "rouge-score" and rouge_scorer is None]
    if missing:
        print(f"Missing deps: {missing}. pip install " + " ".join(missing), file=sys.stderr); sys.exit(2)

def _to_text(x):
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, dict):
        for k in ("text","value","answer","content","string","output","summary","target"):
            if k in x: 
                t=_to_text(x[k]); return t
        for v in x.values():
            t=_to_text(v)
            if t: return t
        return ""
    if isinstance(x, (list,tuple)):
        for y in x:
            t=_to_text(y)
            if t: return t
        return ""
    return str(x)

_WS = re.compile(r"\s+")
_PUN = re.compile(r"[\W_]+", re.UNICODE)
_ART = {"a","an","the"}

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = _PUN.sub(" ", s)
    s = " ".join(w for w in _WS.sub(" ", s).strip().split() if w not in _ART)
    return s

def f1_score(pred: str, golds: List[str]) -> float:
    pred_tokens = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        gold_tokens = normalize_answer(g).split()
        common = {}
        for tok in pred_tokens:
            common[tok] = min(pred_tokens.count(tok), gold_tokens.count(tok))
        num_same = sum(common.values())
        if num_same == 0: best = max(best, 0.0); continue
        precision = num_same / max(len(pred_tokens), 1)
        recall = num_same / max(len(gold_tokens), 1)
        best = max(best, (2*precision*recall)/(precision+recall))
    return best

def exact_match(pred: str, golds: List[str]) -> bool:
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)

def now() -> float:
    try: return time.perf_counter()
    except Exception: return time.time()

def list_models(endpoint: str) -> set:
    r = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=10); r.raise_for_status()
    return {m.get("id") for m in r.json().get("data", []) if m.get("id")}

def assert_server_up(endpoint: str, timeout_s: float = 5.0):
    r = requests.get(f"{endpoint.rstrip('/')}/health", timeout=timeout_s); r.raise_for_status()

def model_tokenizer(model_id: str):
    # heuristic mapping to a tiktoken encoding
    if not tiktoken: return None
    try:
        return tiktoken.encoding_for_model(model_id)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

def count_tokens(text: str, enc) -> int:
    if enc is None: return len(re.findall(r"\S+", text))
    try: return len(enc.encode(text))
    except Exception: return len(re.findall(r"\S+", text))

# ---------------- HTTP (non-stream + stream) ----------------

def post_completion(endpoint: str, api_kind: str, payload: Dict[str,Any], verbose=False):
    url = f"{endpoint.rstrip('/')}/v1/{'chat/completions' if api_kind=='chat' else 'completions'}"
    headers = {"Content-Type":"application/json","X-Bench-Run":str(uuid.uuid4())}
    t0 = now()
    r = requests.post(url, headers=headers, data=jdump(payload), timeout=600)
    ctype = (r.headers.get("content-type") or "").lower()
    if verbose: print(f"[HTTP] {r.status_code} {ctype}")
    if r.status_code >= 400 and verbose: print("[HTTP][err-body]", r.text[:2000])
    r.raise_for_status()
    t1 = now()
    obj = r.json()
    if api_kind == "chat":
        text = (obj.get("choices",[{}])[0].get("message",{}).get("content") or "")
    else:
        text = (obj.get("choices",[{}])[0].get("text") or "")
    usage = obj.get("usage")
    return text, max(t1-t0,0.0), usage, None  # None=ttft

def stream_completion(endpoint: str, api_kind: str, payload: Dict[str,Any], verbose=False):
    """Return text, latency, usage(None for streaming), ttft."""
    url = f"{endpoint.rstrip('/')}/v1/{'chat/completions' if api_kind=='chat' else 'completions'}"
    headers = {"Content-Type":"application/json","X-Bench-Run":str(uuid.uuid4())}
    payload = dict(payload); payload["stream"]=True
    t0 = now(); ttft = None; chunks=[]
    with requests.post(url, headers=headers, data=jdump(payload), stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line: continue
            if line.startswith("data: "): line = line[6:]
            if line.strip() == "[DONE]": break
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ttft is None: ttft = now() - t0
            if api_kind == "chat":
                delta = ev.get("choices",[{}])[0].get("delta",{}).get("content","")
            else:
                delta = ev.get("choices",[{}])[0].get("text","")
            chunks.append(delta)
    text = "".join(chunks); latency = now()-t0
    return text, max(latency,0.0), None, ttft

# ---------------- Core data types ----------------

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

@dataclass
class Sample:
    id: str
    prompt: str                    # for /completions
    messages: Optional[List[dict]] # for /chat/completions
    references: List[str]          # gold answers or target summaries
    extra: Dict[str, Any]          # task-specific

class Evaluator:
    name: str
    def update(self, pred: str, sample: Sample): ...
    def finalize(self) -> Dict[str, Any]: ...

class Task:
    name: str
    evaluator: Evaluator
    def __init__(self, opts: WorkloadOpts): self.opts = opts
    def load(self) -> Iterable[Sample]: ...
    def build_payload(self, sample: Sample) -> Dict[str,Any]:
        if self.opts.api_kind == "chat":
            msgs = sample.messages or [{"role":"user","content":sample.prompt}]
            if self.opts.system_prompt:
                msgs = [{"role":"system","content":self.opts.system_prompt}]+msgs
            return {"model": self.opts.model, "messages": msgs, "max_tokens": self.opts.max_tokens}
        else:
            prompt = sample.prompt
            if self.opts.system_prompt:
                prompt = f"System: {self.opts.system_prompt}\nUser: {prompt}\nAssistant:"
            return {"model": self.opts.model, "prompt": prompt, "max_tokens": self.opts.max_tokens}
    def decoding_overrides(self, payload: Dict[str,Any]) -> Dict[str,Any]:
        if self.opts.use_beam_search:
            width = max(2, int(self.opts.num_beams))
            payload.update({"use_beam_search": True, "n": width, "best_of": max(self.opts.best_of,width),
                            "temperature": 0, "top_p": 1.0})
        else:
            payload.update({"temperature": float(self.opts.temperature),
                            "top_p": float(self.opts.top_p)})
            if self.opts.top_k > 0: payload["top_k"] = int(self.opts.top_k)
        if self.opts.seed is not None: payload["seed"] = int(self.opts.seed)
        return payload

# ---------------- Evaluators ----------------

class EMF1Evaluator(Evaluator):
    def __init__(self, name="em_f1"): self.name=name; self.n=0; self.em=0; self.f1=0.0
    def update(self, pred: str, sample: Sample):
        self.n += 1
        self.em += 1 if exact_match(pred, sample.references) else 0
        self.f1 += f1_score(pred, sample.references)
    def finalize(self): 
        return {"metric": self.name, "samples": self.n,
                "EM": (self.em/self.n if self.n else 0.0),
                "F1": (self.f1/self.n if self.n else 0.0)}

class RougeLEvaluator(Evaluator):
    def __init__(self, use_r1=False, use_r2=False, name="rouge"):
        self.name=name; self.use_r1=use_r1; self.use_r2=use_r2
        self.scores=[]; 
        if rouge_scorer is not None:
            metrics=["rougeL"]
            if use_r1: metrics.append("rouge1")
            if use_r2: metrics.append("rouge2")
            self._scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        else:
            self._scorer = None
    def update(self, pred: str, sample: Sample):
        ref = sample.references[0] if sample.references else ""
        if self._scorer:
            rs = self._scorer.score(ref, pred)
            out={"rougeL_f": rs["rougeL"].fmeasure}
            if self.use_r1: out["rouge1_f"]=rs["rouge1"].fmeasure
            if self.use_r2: out["rouge2_f"]=rs["rouge2"].fmeasure
        else:
            # Fallback: LCS-based F1 rough
            out={"rougeL_f": _lcs_f1(ref, pred)}
        self.scores.append(out)
    def finalize(self):
        agg={}
        if not self.scores: return {"metric": self.name}
        keys=self.scores[0].keys()
        for k in keys:
            agg[k]=sum(s[k] for s in self.scores)/len(self.scores)
        agg["metric"]=self.name; agg["samples"]=len(self.scores); return agg

def _lcs_f1(a: str, b: str)->float:
    A=a.split(); B=b.split()
    dp=[[0]*(len(B)+1) for _ in range(len(A)+1)]
    for i in range(len(A)):
        for j in range(len(B)):
            dp[i+1][j+1]=dp[i][j]+1 if A[i]==B[j] else max(dp[i][j+1], dp[i+1][j])
    lcs=dp[-1][-1]; 
    if lcs==0: return 0.0
    p=lcs/max(1,len(B)); r=lcs/max(1,len(A)); return (2*p*r)/(p+r)


_ANS_PATTERNS = [
        re.compile(r"(?i)^\s*final\s*answer\s*[:\-]\s*(.+)$"),
        re.compile(r"(?i)^\s*answer\s*[:\-]\s*(.+)$"),
        ]
def _short_answer(pred: str) -> str:
    # take first non-empty line
    line = next((ln.strip() for ln in pred.splitlines() if ln.strip()), pred.strip())
    for rx in _ANS_PATTERNS:
        m = rx.search(line)
        if m: line = m.group(1).strip(); break
    # remove trailing punctuation and explanations after a period/semicolon
    line = re.split(r"[.;\n]", line, maxsplit=1)[0].strip()
    return line

class HumanEvalEvaluator(Evaluator):
    """
    Executes HumanEval tests to compute pass@k. Requires dataset 'openai_humaneval'
    which contains 'test' code per task. We support k in {1,5,10} configurable.
    """
    def __init__(self, ks=(1,)): self.name="pass@k"; self.ks=tuple(sorted(set(ks))); self.records=[]
    def update(self, pred: str, sample: Sample):
        self.records.append((sample, pred))
    def finalize(self):
        # group by task id, run tests for each pred (one per sample id)
        ok_counts={k:0 for k in self.ks}; totals={}
        for (sample, pred) in self.records:
            task = sample.extra.get("task_id", sample.id)
            totals[task]=totals.get(task,0)+1
            passed = _run_humaneval_tests(pred, sample.extra.get("tests",""), sample.extra.get("prompt_leading",""))
            # For pass@k computation we count 1 success per task if any of its k generations passed.
            # Here we assume exactly one generation per sample; aggregating at task granularity:
            sample.extra.setdefault("_passes", []).append(bool(passed))
        # Aggregate per task
        task_passes = {}
        for (sample, _pred) in self.records:
            task = sample.extra.get("task_id", sample.id)
            if task in task_passes: continue
            task_passes[task] = sample.extra.get("_passes", [])
        n_tasks=len(task_passes)
        for k in self.ks:
            s=0
            for passes in task_passes.values():
                m=len(passes)
                if m==0: continue
                # pass@k estimator (OpenAI): 1 - C(n - c, k)/C(n, k) where c=#successes, n=#samples
                c=sum(1 for x in passes if x)
                if k>m: est = 1.0 if c>0 else 0.0
                else:
                    from math import comb
                    est = 1.0 - (comb(m-c, k)/comb(m, k))
                s += est
            ok_counts[k]= s / max(1,n_tasks)
        return {"metric":"pass@k","tasks":n_tasks, **{f"pass@{k}":ok_counts[k] for k in self.ks}}

def _run_humaneval_tests(pred_code: str, tests_code: str, prompt_leading: str) -> bool:
    """
    Writes a temp file combining model completion with the provided test code.
    Executes under a subprocess with timeout. Returns True if exit code 0.
    SECURITY: Executes untrusted code; use in sandbox/container.
    """
    with tempfile.TemporaryDirectory() as td:
        prog = os.path.join(td, "prog.py")
        with open(prog, "w", encoding="utf-8") as f:
            # prompt_leading contains the function signature/docstring; completion should continue from here
            f.write(prompt_leading)
            f.write(pred_code)
            f.write("\n\nif __name__ == '__main__':\n")
            f.write("    pass\n")
        test = os.path.join(td, "test.py")
        with open(test, "w", encoding="utf-8") as f:
            f.write("from prog import *\n")
            f.write(tests_code)
        try:
            res = subprocess.run([sys.executable, "-u", test], cwd=td,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=30)
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

# ---------------- Tasks ----------------

def _hf_load(name, config, split, cache_dir):
    require(["datasets"])
    return hfds.load_dataset(name, config, split=split, cache_dir=cache_dir) if config else hfds.load_dataset(name, split=split, cache_dir=cache_dir)

def _load_longbench_manual(task_name: str, cache_dir: str):
    """
    Manual loader for LongBench dataset (dataset scripts no longer supported).
    Downloads data.zip from HuggingFace Hub and loads JSONL files.
    """
    import os, json, zipfile
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub required for LongBench. Install: pip install huggingface_hub")
    
    # Download data.zip if not already cached
    extract_dir = os.path.join(cache_dir, "longbench_extracted")
    data_file = os.path.join(extract_dir, "data", f"{task_name}.jsonl")
    
    if not os.path.exists(data_file):
        os.makedirs(extract_dir, exist_ok=True)
        zip_path = hf_hub_download(
            repo_id="THUDM/LongBench",
            filename="data.zip",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    # Load JSONL file
    data = []
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        raise FileNotFoundError(f"LongBench task '{task_name}' not found. Available in data.zip: {os.listdir(os.path.join(extract_dir, 'data')) if os.path.exists(os.path.join(extract_dir, 'data')) else 'none'}")
    
    return data

def _load_sharegpt_manual(cache_dir: str):
    """
    Manual loader for ShareGPT dataset (no standard config available).
    Downloads JSON file directly from HuggingFace Hub.
    """
    import os, json
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub required for ShareGPT. Install: pip install huggingface_hub")
    
    json_path = hf_hub_download(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        filename="ShareGPT_V3_unfiltered_cleaned_split.json",
        repo_type="dataset",
        cache_dir=cache_dir
    )
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def _load_leval_manual(task_name: str, cache_dir: str):
    """
    Manual loader for LEval dataset (dataset scripts no longer supported).
    Downloads JSONL files directly from HuggingFace Hub.
    """
    import os, json
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub required for LEval. Install: pip install huggingface_hub")
    
    # Map task names to LEval file paths
    task_map = {
        "narrative_qa": "LEval/Generation/narrative_qa.jsonl",
        "natural_question": "LEval/Generation/natural_question.jsonl",
        "meeting_summ": "LEval/Generation/meeting_summ.jsonl",
        "gov_report_summ": "LEval/Generation/gov_report_summ.jsonl",
        "multidoc_qa": "LEval/Generation/multidoc_qa.jsonl",
        "qasper": "LEval/Exam/quality.jsonl",  # Fallback
    }
    
    # Use narrative_qa as default
    filepath = task_map.get(task_name, "LEval/Generation/narrative_qa.jsonl")
    
    jsonl_path = hf_hub_download(
        repo_id="L4NLP/LEval",
        filename=filepath,
        repo_type="dataset",
        cache_dir=cache_dir
    )
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return data

# TriviaQA (rc) validation -> EM/F1
@register("qa", "triviaqa")
class TriviaQATask(Task):
    name="triviaqa"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator()
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

# NarrativeQA (summary) validation -> F1 on abstractive answers
@register("qa", "narrativeqa")
class NarrativeQATask(Task):
    name="narrativeqa"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="f1_only")
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

# LongBench summarization subsets -> ROUGE-L
@register("summ", "longbench_gov")
class LongBenchGov(Task):
    name="longbench_gov"
    def __init__(self, opts): super().__init__(opts); self.evaluator = RougeLEvaluator()
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

@register("summ", "longbench_qmsum")
class LongBenchQMSum(Task):
    name="longbench_qmsum"
    def __init__(self, opts): super().__init__(opts); self.evaluator = RougeLEvaluator()
    def load(self)->Iterable[Sample]:
        # LongBench dataset scripts no longer supported, use manual loader
        ds = _load_longbench_manual("qmsum", self.opts.cache_dir)
        for i,row in enumerate(ds):
            doc=_to_text(row.get("context") or row.get("input"))
            # LongBench stores answer/summary in "answers" field (list)
            answers = row.get("answers", [])
            summ = _to_text(answers[0] if answers else "")
            if not doc or not summ: continue
            prompt=f"Generate a concise meeting summary:\n\n{doc}"
            yield Sample(id=f"lbq-{i}", prompt=prompt, messages=[{"role":"user","content":prompt}], references=[summ], extra={})

# LooGLE summarization -> ROUGE-L
@register("summ", "loogle")
class LooGLE(Task):
    name="loogle"
    def __init__(self, opts): super().__init__(opts); self.evaluator = RougeLEvaluator()
    def load(self)->Iterable[Sample]:
        # Load LooGLE dataset - fail if not available
        ds = _hf_load("bigai-nlco/LooGLE", "summarization", "test", self.opts.cache_dir)
        for i, row in enumerate(ds):
            doc  = _to_text(row.get("document") or row.get("context") or row.get("text") or row.get("passage"))
            summ = _to_text(row.get("summary") or row.get("target") or row.get("answer"))
            if not doc or not summ:
                continue
            prompt = f"Summarize the following legal text concisely:\n\n{doc}"
            yield Sample(
                id=f"loogle-summarization-{i}",
                prompt=prompt,
                messages=[{"role":"user","content":prompt}],
                references=[summ],
                extra={"subset": "summarization"}
            )


# HumanEval code completion -> pass@k via tests
@register("code", "humaneval")
class HumanEval(Task):
    name="humaneval"
    def __init__(self, opts):
        super().__init__(opts)
        ks = (1,5,10) if self.opts.best_of>=10 else (1,5) if self.opts.best_of>=5 else (1,)
        self.evaluator = HumanEvalEvaluator(ks=ks)
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        ds=_hf_load("openai_humaneval", None, "test", self.opts.cache_dir)
        for i,row in enumerate(ds):
            task_id=_to_text(row.get("task_id"))
            prompt=_to_text(row.get("prompt"))
            tests=_to_text(row.get("test") or row.get("tests"))
            if not prompt or not tests: continue
            # Use /completions with prompt as prefix; model returns code continuation
            yield Sample(id=f"humaneval-{i}", prompt=prompt, messages=None,
                         references=[], extra={"tests":tests, "task_id":task_id, "prompt_leading":prompt})

# Prefix-KV probe (shared system+user prefix) -> no automatic metric; logs prefill stats
@register("prefix", "kvprobe")
class PrefixKV(Task):
    name="prefix_kv"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")  # placeholder
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        # Use a small ShareGPT-like set; fallback to alpaca
        try:
            ds=_hf_load("anon8231489123/ShareGPT_Vicuna_unfiltered","ShareGPT_V4.3_unfiltered_cleaned_split","train", self.opts.cache_dir)
            it = (self._norm(row) for row in ds)
        except Exception:
            alp=_hf_load("yahma/alpaca-cleaned", None, "train", self.opts.cache_dir)
            it = ({"prompt": _to_text(r["instruction"]) + ("\n\nInput: "+_to_text(r["input"]) if r.get("input") else ""),
                   "answer": _to_text(r["output"])} for r in alp)
        shared_sys = "You are a concise assistant. Always answer directly."
        shared_user = "Common preface: Use bullet points if appropriate. Now answer:"
        for i, rec in enumerate(it):
            q = rec.get("prompt") or rec.get("question") or ""
            a = rec.get("answer") or ""
            if not q or not a: continue
            msgs=[{"role":"system","content":shared_sys},
                  {"role":"user","content":shared_user},
                  {"role":"user","content":q}]
            extra={"shared_sys":shared_sys,"shared_user":shared_user}
            yield Sample(id=f"kv-{i}", prompt=f"{shared_sys}\n{shared_user}\n{q}", messages=msgs, references=[a], extra=extra)
    def _norm(self,row):
        conv=row.get("conversations") or row.get("messages")
        if conv and isinstance(conv, list):
            q=_to_text(next((m.get("value") or m.get("content") for m in conv if (m.get('from') in ('human','user') or m.get('role')=='user')), ""))
            a=_to_text(next((m.get("value") or m.get("content") for m in conv if (m.get('from') in ('gpt','assistant') or m.get('role')=='assistant')), ""))
            return {"prompt":q,"answer":a}
        return {"prompt":_to_text(row.get("instruction")),"answer":_to_text(row.get("output"))}

# ========== SPSR (Single-prompt-single-response) category ==========

@register("spsr", "alpaca")
class AlpacaSPSR(Task):
    """Alpaca instruction-following in single-turn format."""
    name="alpaca"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
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

@register("spsr", "triviaqa")
class TriviaQASPSR(Task):
    """TriviaQA in single-prompt format."""
    name="triviaqa"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator()
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

@register("spsr", "narrativeqa")
class NarrativeQASPSR(Task):
    """NarrativeQA in single-prompt format."""
    name="narrativeqa"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="f1_only")
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

@register("spsr", "wikitext")
class WikitextSPSR(Task):
    """Wikitext language modeling perplexity (simple eval: token overlap)."""
    name="wikitext"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="overlap")
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        ds=_hf_load("wikitext", "wikitext-2-raw-v1", "test", self.opts.cache_dir)
        # WikiText is usually used for perplexity, but for a generate-eval we can sample prompts
        # Take first N words as prompt, rest as reference
        for i,row in enumerate(ds):
            text=_to_text(row.get("text"))
            if not text or len(text.split())<20: continue
            words=text.split()
            mid=len(words)//2
            prompt_text=" ".join(words[:mid])
            ref_text=" ".join(words[mid:])
            prompt = f"Continue the following text:\n\n{prompt_text}"
            yield Sample(id=f"wikitext-{i}", prompt=prompt,
                         messages=[{"role":"user","content":prompt}],
                         references=[ref_text], extra={})

# ========== BEAM search category ==========

@register("beam", "longbench_gov")
class LongBenchGovBeam(Task):
    """LongBench gov_report with beam search for summarization."""
    name="longbench_gov"
    def __init__(self, opts): super().__init__(opts); self.evaluator = RougeLEvaluator()
    def load(self)->Iterable[Sample]:
        ds = _load_longbench_manual("gov_report", self.opts.cache_dir)
        for i,row in enumerate(ds):
            doc=_to_text(row.get("context") or row.get("input"))
            answers = row.get("answers", [])
            summ = _to_text(answers[0] if answers else "")
            if not doc or not summ: continue
            prompt=f"Summarize the following government report succinctly:\n\n{doc}"
            yield Sample(id=f"beam-lbg-{i}", prompt=prompt, messages=[{"role":"user","content":prompt}], references=[summ], extra={})

@register("beam", "longbench_qmsum")
class LongBenchQMSumBeam(Task):
    """LongBench qmsum with beam search."""
    name="longbench_qmsum"
    def __init__(self, opts): super().__init__(opts); self.evaluator = RougeLEvaluator()
    def load(self)->Iterable[Sample]:
        ds = _load_longbench_manual("qmsum", self.opts.cache_dir)
        for i,row in enumerate(ds):
            doc=_to_text(row.get("context") or row.get("input"))
            answers = row.get("answers", [])
            summ = _to_text(answers[0] if answers else "")
            if not doc or not summ: continue
            prompt=f"Generate a concise meeting summary:\n\n{doc}"
            yield Sample(id=f"beam-lbq-{i}", prompt=prompt, messages=[{"role":"user","content":prompt}], references=[summ], extra={})

@register("beam", "narrativeqa")
class NarrativeQABeam(Task):
    """NarrativeQA with beam search."""
    name="narrativeqa"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="f1_only")
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

@register("beam", "triviaqa")
class TriviaQABeam(Task):
    """TriviaQA with beam search."""
    name="triviaqa"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator()
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

# ========== PREFIX (shared prefixes) category additions ==========

@register("prefix", "sharegpt")
class ShareGPTPrefix(Task):
    """ShareGPT conversations with shared system prompt for prefix caching."""
    name="sharegpt"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
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

@register("prefix", "leval")
class LevalPrefix(Task):
    """LongEval (LEval) dataset with shared instruction prefix."""
    name="leval"
    def __init__(self, opts): super().__init__(opts); self.evaluator = RougeLEvaluator()
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        # Use manual loader - LEval has dataset script issues
        ds = _load_leval_manual("narrative_qa", self.opts.cache_dir)
        shared_sys = "You are an expert at analyzing long documents. Answer comprehensively based on the provided context."
        for i,row in enumerate(ds):
            # LEval format: 'input' (context), 'instructions' (questions list), 'outputs' (answers list)
            doc = _to_text(row.get("input") or "")
            questions = row.get("instructions", [])
            outputs = row.get("outputs", [])
            
            if not doc or not questions or not outputs: continue
            # Use first question/answer pair
            q = _to_text(questions[0] if isinstance(questions, list) and questions else questions)
            ans = _to_text(outputs[0] if isinstance(outputs, list) and outputs else outputs)
            
            prompt = f"{q}\n\nContext:\n{doc}"
            msgs=[{"role":"system","content":shared_sys},
                  {"role":"user","content":prompt}]
            yield Sample(id=f"leval-{i}", prompt=f"{shared_sys}\n{prompt}", messages=msgs, references=[ans], extra={"shared_sys":shared_sys})

@register("prefix", "longchat")
class LongChatPrefix(Task):
    """LongChat-7k-v1.5 multi-turn conversations with shared system prompt."""
    name="longchat"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        # LongChat dataset (lmsys/longchat-lines) is not publicly available
        raise RuntimeError(
            "LongChat dataset (lmsys/longchat-lines) is not available. "
            "This dataset is private or does not exist. "
            "Please use an alternative dataset or disable this workload."
        )

# ========== CHAT (chatbots) category ==========

@register("chat", "sharegpt")
class ShareGPTChat(Task):
    """ShareGPT conversational format for chatbot evaluation."""
    name="sharegpt"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
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

@register("chat", "longchat")
class LongChatChat(Task):
    """LongChat conversations for chatbot evaluation."""
    name="longchat"
    def __init__(self, opts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        # LongChat dataset (lmsys/longchat-lines) is not publicly available
        raise RuntimeError(
            "LongChat dataset (lmsys/longchat-lines) is not available. "
            "This dataset is private or does not exist. "
            "Please use an alternative dataset or disable this workload."
        )

# ---------------- Config / opts ----------------

def _merge(dflt: dict, over: dict) -> dict:
    out = dict(dflt or {})
    for k, v in (over or {}).items(): out[k] = v
    return out

def _enforce(cat: str, opts: WorkloadOpts) -> WorkloadOpts:
    # Reasonable defaults by category
    if cat in ("summ",):
        opts.temperature = 0.7; opts.top_p = 0.95; opts.use_beam_search = False
        opts.api_kind = opts.api_kind or "chat"
    if cat in ("qa",):
        opts.temperature = 0.0; opts.top_p = 1.0; opts.use_beam_search = False
    if cat in ("code",):
        opts.temperature = 0.2; opts.top_p = 0.95; opts.api_kind = "completions"; opts.best_of = max(opts.best_of, 5)
    if cat in ("prefix",):
        opts.temperature = 0.0; opts.top_p = 1.0
        opts.api_kind = opts.api_kind or "chat"
    if cat in ("spsr",):
        # Single-prompt-single-response: deterministic or low temperature
        opts.temperature = 0.0; opts.top_p = 1.0; opts.use_beam_search = False
        opts.api_kind = opts.api_kind or "chat"
    if cat in ("beam",):
        # Beam search configurations
        opts.use_beam_search = True
        opts.num_beams = opts.num_beams if opts.num_beams > 1 else 4
        opts.best_of = max(opts.best_of, opts.num_beams)
        opts.temperature = 0.0  # Required for beam search
        opts.top_p = 1.0
        opts.api_kind = opts.api_kind or "chat"
    if cat in ("chat",):
        # Chatbot evaluations: more diverse, conversational
        opts.temperature = 0.7; opts.top_p = 0.95; opts.use_beam_search = False
        opts.api_kind = "chat"  # Force chat API for chatbots
    return opts

def _build_opts(global_endpoint: str, defaults: dict, per: dict, cat: str) -> WorkloadOpts:
    m = _merge(defaults, per)
    endpoint = global_endpoint or m.get("endpoint") or "http://127.0.0.1:8080"
    opts = WorkloadOpts(
        endpoint=endpoint,
        model=m.get("model"),
        api_kind=m.get("api_kind", "chat"),
        temperature=float(m.get("temperature", 0.0)),
        max_tokens=int(m.get("max_tokens", 256)),
        top_p=float(m.get("top_p", 1.0)),
        top_k=int(m.get("top_k", -1)),
        qps=float(m.get("qps", 4.0)),
        limit=int(m.get("limit", 100)),
        out_dir=m.get("out_dir", "./runs"),
        cache_dir=m.get("cache_dir", "./.hf-cache"),
        system_prompt=m.get("system_prompt"),
        use_beam_search=bool(m.get("use_beam_search", False)),
        num_beams=int(m.get("num_beams", 1)),
        best_of=int(m.get("best_of", 1)),
        seed=int(m.get("seed", 1234)),
        stream=bool(m.get("stream", True)),
        metrics_tokenizer=str(m.get("metrics_tokenizer", "model")),  # "model"|"tiktoken"|"whitespace"
        metrics_tokenizer_model=m.get("metrics_tokenizer_model"),
        verbose=bool(m.get("verbose", False)),
        timeout_s=float(m.get("timeout_s", 600.0)),
    )
    return _enforce(cat, opts)

# ---------------- Runner ----------------

@contextmanager
def _rate_limiter(qps: float):
    t_last = [0.0]
    interval = 1.0 / max(qps, 1e-6)
    def wait():
        nowt = now()
        to_sleep = (t_last[0] + interval) - nowt
        if to_sleep > 0: time.sleep(to_sleep)
        t_last[0] = now()
    yield wait

def run_task(opts: WorkloadOpts, w: Workload):
    os.makedirs(opts.out_dir, exist_ok=True); os.makedirs(opts.cache_dir, exist_ok=True)
    rng = random.Random(opts.seed)
    tokenizer = model_tokenizer(opts.metrics_tokenizer_model or opts.model) if opts.metrics_tokenizer!="whitespace" else None

    task: Task = w.task_cls(opts)
    evaluator = task.evaluator
    out_path = os.path.join(opts.out_dir, f"{w.category}_{w.name}.jsonl")
    manifest_path = os.path.join(opts.out_dir, f"{w.category}_{w.name}_manifest.json")

    n=0; sum_lat=0.0; sum_ctok=0; sum_ptok=0; ttft_list=[]
    with open(out_path,"w",encoding="utf-8") as f, _rate_limiter(opts.qps) as throttle:
        for sample in task.load():
            if n >= opts.limit: break
            payload = task.build_payload(sample)
            payload = task.decoding_overrides(payload)
            payload["model"] = opts.model
            # deterministic seeding per-sample
            if "seed" in payload: payload["seed"] = (opts.seed + n) & 0x7fffffff

            # request
            throttle()
            try:
                if opts.stream:
                    pred, latency, usage, ttft = stream_completion(opts.endpoint, opts.api_kind, payload, verbose=opts.verbose)
                else:
                    pred, latency, usage, ttft = post_completion(opts.endpoint, opts.api_kind, payload, verbose=opts.verbose)
            except Exception as e:
                pred=f"<<ERROR: {e}>>"; latency=0.0; usage=None; ttft=None

            # tokens
            if usage:
                ptok=int(usage.get("prompt_tokens",0)); ctok=int(usage.get("completion_tokens",0)); ttot=int(usage.get("total_tokens", ptok+ctok))
            else:
                prompt_text = payload.get("prompt") if "prompt" in payload else jdump(payload.get("messages", []))
                if opts.metrics_tokenizer=="model":
                    ptok=count_tokens(prompt_text, tokenizer); ctok=count_tokens(pred, tokenizer)
                elif opts.metrics_tokenizer=="tiktoken":
                    enc = model_tokenizer(opts.metrics_tokenizer_model or "gpt-3.5-turbo")
                    ptok=count_tokens(prompt_text, enc); ctok=count_tokens(pred, enc)
                else:
                    ptok=count_tokens(prompt_text, None); ctok=count_tokens(pred, None)
                ttot=ptok+ctok

            tpot = (ctok/latency) if latency>0 else None
            # Apply postprocessing if available
            pred_for_eval = pred
            if hasattr(task, 'postprocess_pred'):
                pred_for_eval = task.postprocess_pred(pred, sample)
            evaluator.update(pred_for_eval, sample)

            record = {
                "id": sample.id,
                "request": payload,
                "output": pred,
                "references": sample.references,
                "metrics": {
                    "ttft_s": ttft,
                    "latency_s": latency,
                    "tokens": {"prompt": ptok, "completion": ctok, "total": ttot},
                    "tpot_tokens_per_s": tpot
                }
            }
            f.write(jdump(record)+"\n")

            n += 1; sum_lat += latency; sum_ctok += ctok; sum_ptok += ptok
            if ttft is not None: ttft_list.append(ttft)

    agg = {
        "category": w.category, "workload": w.name, "samples": n,
        "avg_latency_s": (sum_lat/n if n else None),
        "avg_ttft_s": (sum(ttft_list)/len(ttft_list) if ttft_list else None),
        "avg_completion_tokens": (sum_ctok/n if n else None),
        "agg_tpot_tokens_per_s": (sum_ctok/sum_lat if sum_lat>0 else None),
        "evaluator": evaluator.finalize(),
        "out_file": out_path
    }
    with open(manifest_path,"w",encoding="utf-8") as mf:
        manifest = {
            "opts": asdict(opts),
            "served_models": sorted(list(list_models(opts.endpoint))),
            "task": {"category": w.category, "name": w.name},
            "env": {"python": sys.version, "argv": sys.argv},
        }
        mf.write(jdump(manifest))
    print(json.dumps(agg, indent=2, sort_keys=True))

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    runp = sub.add_parser("run")
    runp.add_argument("config")
    runp.add_argument("--model", help="Model to use for benchmarking")
    runp.add_argument("category", nargs="?")
    runp.add_argument("workload", nargs="?")
    args = ap.parse_args()

    if args.cmd == "list":
        print(json.dumps(list_all(), indent=2, sort_keys=True)); return

    cfg = _load_json(args.config)
    
    # Get model-specific configuration
    model_name = args.model
    if not model_name:
        print("Error: --model parameter is required", file=sys.stderr)
        sys.exit(1)
    
    models_cfg = cfg.get("models", {})
    if model_name not in models_cfg:
        print(f"Error: Model '{model_name}' not found in config", file=sys.stderr)
        available_models = list(models_cfg.keys())
        print(f"Available models: {available_models}", file=sys.stderr)
        sys.exit(1)
    
    # Use model-specific defaults
    model_config = models_cfg[model_name]
    endpoint = model_config.get("endpoint", "http://127.0.0.1:8080")
    defaults = model_config  # Use entire model config as defaults
    cfg_wls = cfg.get("workloads") or {}

    assert_server_up(endpoint); served = list_models(endpoint)

    targets: List[Tuple[str,str]]=[]
    if args.category and args.workload:
        if args.category not in cfg_wls or args.workload not in (cfg_wls.get(args.category) or {}):
            print(f"Workload '{args.category}/{args.workload}' not in config.", file=sys.stderr); sys.exit(2)
        targets=[(args.category,args.workload)]
    elif args.category:
        for wl_name, per in (cfg_wls.get(args.category) or {}).items():
            if per.get("enabled", False): targets.append((args.category, wl_name))
    else:
        for cat, d in cfg_wls.items():
            for wl_name, per in (d or {}).items():
                if per.get("enabled", False): targets.append((cat, wl_name))
    if not targets: print("No workloads selected.", file=sys.stderr); sys.exit(2)

    for cat, wl_name in targets:
        wlist = _REGISTRY.get(cat) or []
        w = next((x for x in wlist if x.name == wl_name), None)
        if w is None:
            print(f"Unknown workload '{wl_name}' in '{cat}'. Available: {list_all().get(cat, [])}", file=sys.stderr); sys.exit(2)
        per_cfg = (cfg_wls.get(cat) or {}).get(wl_name) or {}
        opts = _build_opts(endpoint, defaults, per_cfg, cat)
        if opts.model not in served:
            raise SystemExit(f"Model '{opts.model}' not served at {opts.endpoint}. Served: {sorted(served)}")
        run_task(opts, w)

def _load_json(path: str) -> dict:
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

if __name__ == "__main__":
    main()

