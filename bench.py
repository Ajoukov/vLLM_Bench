#!/usr/bin/env python3
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

import argparse
import json
import os
import random
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import requests

# Import from benchmarks module
from benchmarks import get_registry, list_all
from benchmarks.base import Task, WorkloadOpts
from benchmarks.output_writer import OutputWriter
from benchmarks.utils import count_tokens, model_tokenizer

try:
    import orjson

    def jdump(obj):
        return orjson.dumps(obj).decode("utf-8")
except Exception:

    def jdump(obj):
        return json.dumps(obj, ensure_ascii=False)

# ---------------- Utils ----------------


def now() -> float:
    try:
        return time.perf_counter()
    except Exception:
        return time.time()


def list_models(endpoint: str) -> set:
    r = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=10)
    r.raise_for_status()
    return {m.get("id") for m in r.json().get("data", []) if m.get("id")}


def assert_server_up(endpoint: str, timeout_s: float = 5.0):
    r = requests.get(f"{endpoint.rstrip('/')}/health", timeout=timeout_s)
    r.raise_for_status()


# ---------------- HTTP (non-stream + stream) ----------------


def post_completion(
    endpoint: str, api_kind: str, payload: Dict[str, Any], verbose=False
):
    url = f"{endpoint.rstrip('/')}/v1/{'chat/completions' if api_kind == 'chat' else 'completions'}"
    headers = {"Content-Type": "application/json", "X-Bench-Run": str(uuid.uuid4())}
    t0 = now()
    r = requests.post(url, headers=headers, data=jdump(payload), timeout=600)
    ctype = (r.headers.get("content-type") or "").lower()
    if verbose:
        print(f"[HTTP] {r.status_code} {ctype}")
    if r.status_code >= 400 and verbose:
        print("[HTTP][err-body]", r.text[:2000])
    r.raise_for_status()
    t1 = now()
    obj = r.json()
    if api_kind == "chat":
        text = obj.get("choices", [{}])[0].get("message", {}).get("content") or ""
    else:
        text = obj.get("choices", [{}])[0].get("text") or ""
    usage = obj.get("usage")
    return text, max(t1 - t0, 0.0), usage, None  # None=ttft


def stream_completion(
    endpoint: str, api_kind: str, payload: Dict[str, Any], verbose=False
):
    """Return text, latency, usage(None for streaming), ttft."""
    url = f"{endpoint.rstrip('/')}/v1/{'chat/completions' if api_kind == 'chat' else 'completions'}"
    headers = {"Content-Type": "application/json", "X-Bench-Run": str(uuid.uuid4())}
    payload = dict(payload)
    payload["stream"] = True
    t0 = now()
    ttft = None
    chunks = []
    with requests.post(
        url, headers=headers, data=jdump(payload), stream=True, timeout=600
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ttft is None:
                ttft = now() - t0
            if api_kind == "chat":
                delta = ev.get("choices", [{}])[0].get("delta", {}).get("content", "")
            else:
                delta = ev.get("choices", [{}])[0].get("text", "")
            chunks.append(delta)
    text = "".join(chunks)
    latency = now() - t0
    return text, max(latency, 0.0), None, ttft


# ---------------- Config / opts ----------------


def _merge(dflt: dict, over: dict) -> dict:
    out = dict(dflt or {})
    for k, v in (over or {}).items():
        out[k] = v
    return out


def _enforce(cat: str, opts: WorkloadOpts) -> WorkloadOpts:
    # Reasonable defaults by category
    if cat in ("summ",):
        opts.temperature = 0.7
        opts.top_p = 0.95
        opts.use_beam_search = False
        opts.api_kind = opts.api_kind or "chat"
    if cat in ("qa",):
        opts.temperature = 0.0
        opts.top_p = 1.0
        opts.use_beam_search = False
    if cat in ("code",):
        opts.temperature = 0.2
        opts.top_p = 0.95
        opts.api_kind = "completions"
        opts.best_of = max(opts.best_of, 5)
    if cat in ("prefix",):
        opts.temperature = 0.0
        opts.top_p = 1.0
        opts.api_kind = opts.api_kind or "chat"
    if cat in ("spsr",):
        # Single-prompt-single-response: deterministic or low temperature
        opts.temperature = 0.0
        opts.top_p = 1.0
        opts.use_beam_search = False
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
        opts.temperature = 0.7
        opts.top_p = 0.95
        opts.use_beam_search = False
        opts.api_kind = "chat"  # Force chat API for chatbots
    return opts


def _build_opts(
    global_endpoint: str, defaults: dict, per: dict, cat: str, data_dir: str
) -> WorkloadOpts:
    m = _merge(defaults, per)
    endpoint = global_endpoint or m.get("endpoint") or "http://127.0.0.1:8080"

    # Parse LMCache configuration
    lmcache_config = m.get("lmcache", {})

    # Use data_dir as the default cache directory for datasets
    # In Docker mode, this will be mounted to /data for persistence
    cache_dir = m.get("cache_dir") or data_dir

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
        cache_dir=cache_dir,
        system_prompt=m.get("system_prompt"),
        use_beam_search=bool(m.get("use_beam_search", False)),
        num_beams=int(m.get("num_beams", 1)),
        best_of=int(m.get("best_of", 1)),
        seed=int(m.get("seed", 1234)),
        stream=bool(m.get("stream", True)),
        metrics_tokenizer=str(
            m.get("metrics_tokenizer", "model")
        ),  # "model"|"tiktoken"|"whitespace"
        metrics_tokenizer_model=m.get("metrics_tokenizer_model"),
        verbose=bool(m.get("verbose", False)),
        timeout_s=float(m.get("timeout_s", 600.0)),
        output_format=m.get("output_format"),  # json, jsonl, csv, parquet
        output_file=m.get("output_file"),  # path to output file
        # LMCache configuration
        lmcache_enabled=bool(lmcache_config.get("enabled", False)),
        lmcache_chunk_size=int(lmcache_config.get("chunk_size", 256)),
        lmcache_local_cpu=bool(lmcache_config.get("local_cpu", True)),
        lmcache_max_local_cpu_size=int(lmcache_config.get("max_local_cpu_size", 20)),
        lmcache_backend=str(lmcache_config.get("backend", "local")),
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
        if to_sleep > 0:
            time.sleep(to_sleep)
        t_last[0] = now()

    yield wait


def run_task(opts: WorkloadOpts, w, shared_output_writer=None):
    os.makedirs(opts.out_dir, exist_ok=True)
    os.makedirs(opts.cache_dir, exist_ok=True)
    rng = random.Random(opts.seed)
    tokenizer = (
        model_tokenizer(opts.metrics_tokenizer_model or opts.model)
        if opts.metrics_tokenizer != "whitespace"
        else None
    )

    task: Task = w.task_cls(opts)
    evaluator = task.evaluator
    out_path = os.path.join(opts.out_dir, f"{w.category}_{w.name}.jsonl")
    manifest_path = os.path.join(opts.out_dir, f"{w.category}_{w.name}_manifest.json")

    n = 0
    sum_lat = 0.0
    sum_ctok = 0
    sum_ptok = 0
    ttft_list = []
    with (
        open(out_path, "w", encoding="utf-8") as f,
        _rate_limiter(opts.qps) as throttle,
    ):
        for sample in task.load():
            if n >= opts.limit:
                break
            payload = task.build_payload(sample)
            payload = task.decoding_overrides(payload)
            payload["model"] = opts.model
            # deterministic seeding per-sample
            if "seed" in payload:
                payload["seed"] = (opts.seed + n) & 0x7FFFFFFF

            # request
            throttle()
            try:
                if opts.stream:
                    pred, latency, usage, ttft = stream_completion(
                        opts.endpoint, opts.api_kind, payload, verbose=opts.verbose
                    )
                else:
                    pred, latency, usage, ttft = post_completion(
                        opts.endpoint, opts.api_kind, payload, verbose=opts.verbose
                    )
            except Exception as e:
                pred = f"<<ERROR: {e}>>"
                latency = 0.0
                usage = None
                ttft = None

            # tokens
            if usage:
                ptok = int(usage.get("prompt_tokens", 0))
                ctok = int(usage.get("completion_tokens", 0))
                ttot = int(usage.get("total_tokens", ptok + ctok))
            else:
                prompt_text = (
                    payload.get("prompt")
                    if "prompt" in payload
                    else jdump(payload.get("messages", []))
                )
                if opts.metrics_tokenizer == "model":
                    ptok = count_tokens(prompt_text, tokenizer)
                    ctok = count_tokens(pred, tokenizer)
                elif opts.metrics_tokenizer == "tiktoken":
                    enc = model_tokenizer(
                        opts.metrics_tokenizer_model or "gpt-3.5-turbo"
                    )
                    ptok = count_tokens(prompt_text, enc)
                    ctok = count_tokens(pred, enc)
                else:
                    ptok = count_tokens(prompt_text, None)
                    ctok = count_tokens(pred, None)
                ttot = ptok + ctok

            tpot = (ctok / latency) if latency > 0 else None
            # Apply postprocessing if available
            pred_for_eval = pred
            if hasattr(task, "postprocess_pred"):
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
                    "tpot_tokens_per_s": tpot,
                },
            }
            f.write(jdump(record) + "\n")

            n += 1
            sum_lat += latency
            sum_ctok += ctok
            sum_ptok += ptok
            if ttft is not None:
                ttft_list.append(ttft)

    # Get evaluator results (includes metrics like EM, F1, ROUGE, pass@k, etc.)
    eval_results = evaluator.finalize()

    agg = {
        "model": opts.model,
        "category": w.category,
        "workload": w.name,
        "samples": n,
        "avg_latency_s": (sum_lat / n if n else None),
        "avg_ttft_s": (sum(ttft_list) / len(ttft_list) if ttft_list else None),
        "avg_completion_tokens": (sum_ctok / n if n else None),
        "avg_prompt_tokens": (sum_ptok / n if n else None),
        "agg_tpot_tokens_per_s": (sum_ctok / sum_lat if sum_lat > 0 else None),
        "out_file": out_path,
    }
    # Merge evaluator results into aggregate (adds EM, F1, ROUGE-L, etc.)
    agg.update(eval_results)

    with open(manifest_path, "w", encoding="utf-8") as mf:
        manifest = {
            "opts": asdict(opts),
            "served_models": sorted(list(list_models(opts.endpoint))),
            "task": {"category": w.category, "name": w.name},
            "env": {"python": sys.version, "argv": sys.argv},
        }
        mf.write(jdump(manifest))
    print(json.dumps(agg, indent=2, sort_keys=True))

    # Add aggregate result to shared output writer if provided
    if shared_output_writer:
        shared_output_writer.add_aggregate_result(agg)


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
        print(json.dumps(list_all(), indent=2, sort_keys=True))
        return

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

    # Get data directory for persistent dataset storage
    # In Docker mode, HF_HOME environment variable points to mounted /data
    # Otherwise, use the config value or default to ./data
    data_dir = os.environ.get("HF_HOME") or cfg.get("data_dir", "./data")

    assert_server_up(endpoint)
    served = list_models(endpoint)

    targets: List[Tuple[str, str]] = []
    if args.category and args.workload:
        if args.category not in cfg_wls or args.workload not in (
            cfg_wls.get(args.category) or {}
        ):
            print(
                f"Workload '{args.category}/{args.workload}' not in config.",
                file=sys.stderr,
            )
            sys.exit(2)
        targets = [(args.category, args.workload)]
    elif args.category:
        for wl_name, per in (cfg_wls.get(args.category) or {}).items():
            if per.get("enabled", False):
                targets.append((args.category, wl_name))
    else:
        for cat, d in cfg_wls.items():
            for wl_name, per in (d or {}).items():
                if per.get("enabled", False):
                    targets.append((cat, wl_name))
    if not targets:
        print("No workloads selected.", file=sys.stderr)
        sys.exit(2)

    # Get registry from benchmarks module
    _REGISTRY = get_registry()

    # Create a single shared output writer for all benchmarks
    # Use global output settings from config if available
    global_output_format = cfg.get("output_format") or defaults.get("output_format")
    global_output_file = cfg.get("output_file") or defaults.get("output_file")
    shared_writer = None
    if global_output_format and global_output_file:
        shared_writer = OutputWriter(global_output_format, global_output_file)

    for cat, wl_name in targets:
        wlist = _REGISTRY.get(cat) or []
        w = next((x for x in wlist if x.name == wl_name), None)
        if w is None:
            print(
                f"Unknown workload '{wl_name}' in '{cat}'. Available: {list_all().get(cat, [])}",
                file=sys.stderr,
            )
            sys.exit(2)
        per_cfg = (cfg_wls.get(cat) or {}).get(wl_name) or {}
        opts = _build_opts(endpoint, defaults, per_cfg, cat, data_dir)
        if opts.model not in served:
            raise SystemExit(
                f"Model '{opts.model}' not served at {opts.endpoint}. Served: {sorted(served)}"
            )

        # Use per-workload writer if configured, otherwise use shared writer
        workload_writer = None
        if opts.output_format and opts.output_file:
            workload_writer = OutputWriter(opts.output_format, opts.output_file)
        elif shared_writer:
            workload_writer = shared_writer

        run_task(opts, w, workload_writer)

        # Write workload-specific output immediately if configured
        if opts.output_format and opts.output_file and workload_writer != shared_writer:
            workload_writer.write(mode="append")
            print(
                f"[Output] Written aggregate result to {opts.output_file} (format: {opts.output_format})"
            )

    # Write shared output once at the end
    if shared_writer and shared_writer.aggregate_results:
        shared_writer.write(mode="append")
        print(
            f"[Output] Written {len(shared_writer.aggregate_results)} aggregate results to {global_output_file} (format: {global_output_format})"
        )


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
