# Configuration Reference

## Usage

```bash
# Run benchmarks (creates venv, deploys K8s resources, runs tests)
./init.sh config.json

# Run in isolated Docker container (recommended for CI/CD)
./init.sh --docker config.json

# Cleanup all resources (K8s, cache, venv, port-forwards)
./init.sh --cleanup config.json
```

---

## Global Configuration

### `namespace` (string, required)
Kubernetes namespace where vLLM pods will be deployed.

### `hf_token` (string, required)
Hugging Face API token. Required for downloading models and datasets. For gated models (e.g., Llama), ensure token account has accepted the license.

### `output_format` (string, optional)
Output format for benchmark results. Options: `"json"` or `"csv"`. Default: `"csv"`.

### `output_file` (string, optional)
Path where aggregated results will be saved. Default: `"./runs/benchmark_results.csv"`.

### `data_dir` (string, optional)
Directory for caching downloaded datasets. This directory persists between runs to avoid re-downloading datasets. In Docker mode, this directory is mounted to the container for persistent storage. Default: `"./data"`.

---

## Model Configuration

Each model is defined under the `models` object with the model name as key.

### `model` (string, required)
Hugging Face model identifier (e.g., `"facebook/opt-125m"`).

### `max_model_len` (integer, required)
Maximum context length the model will support during inference.

### `endpoint` (string, required)
Local endpoint URL where the model API will be accessible (e.g., `"http://127.0.0.1:8080"`).

### `port_local` (integer, required)
Local port for port-forwarding to the Kubernetes pod.

### `port_remote` (integer, required)
Remote port inside the Kubernetes pod where vLLM server listens. Default vLLM port: `8000`.

### `api_kind` (string, required)
API type to use. Options: `"completions"` or `"chat"`.

### `max_tokens` (integer, optional)
Maximum number of tokens to generate per request. Default: `256`.

### `temperature` (float, optional)
Sampling temperature (0.0 = greedy, higher = more random). Range: `0.0` - `2.0`. Default: `0.0`.

### `top_p` (float, optional)
Nucleus sampling threshold. Range: `0.0` - `1.0`. Default: `1.0`.

### `qps` (float, optional)
Queries per second rate limit for benchmarking. Default: `2.0`.

### `limit` (integer, optional)
Maximum number of samples to process per benchmark. Useful for quick testing. Default: process all.

### `stream` (boolean, optional)
Enable streaming responses from the API. Default: `true`.

### `seed` (integer, optional)
Random seed for reproducibility. Default: `1234`.

### `out_dir` (string, optional)
Directory where individual benchmark outputs are saved. Default: `"./runs"`.

### `cache_dir` (string, optional)
Directory for caching Hugging Face models and datasets. If not specified, defaults to the global `data_dir` setting.

---

## LMCache Configuration

LMCache is vLLM's KV cache optimization system. Configure under `models.<model_name>.lmcache`.

### `enabled` (boolean, required)
Enable or disable LMCache for this model. Default: `false`.

### `chunk_size` (integer, optional)
Token chunk size for cache granularity. Smaller = more granular but higher overhead. Default: `256`.

### `local_cpu` (boolean, optional)
Enable CPU memory offloading for KV cache. Default: `true`.

### `max_local_cpu_size` (integer, optional)
Maximum CPU memory allocation for cache in GB. Default: `20`.

### `local_disk` (boolean, optional)
Enable local disk-based cache storage. Default: `false`.

### `local_disk_path` (string, conditional)
Path for disk cache storage. Required if `local_disk: true`.

### `max_local_disk_size` (integer, conditional)
Maximum disk space for cache in GB. Required if `local_disk: true`.

### `remote_url` (string, optional)
URL for remote cache backend (e.g., Redis server). Leave empty for local-only caching.

### `remote_serde` (string, conditional)
Serialization format for remote cache. Options: `"safetensors"` or `"pickle"`. Required if `remote_url` is set.

---

## Workload Configuration

Workloads are organized by category. Each workload can be individually configured.

### Workload Categories

- **`spsr`** - Shared Prefix Suffix Removal (tests prefix caching)
- **`beam`** - Beam search generation
- **`prefix`** - Prefix-based benchmarks
- **`chat`** - Chat/conversation benchmarks
- **`qa`** - Question answering
- **`summ`** - Summarization
- **`code`** - Code generation

### Common Workload Parameters

#### `enabled` (boolean, required)
Enable or disable this specific workload. Default: `false`.

#### `temperature` (float, optional)
Override global temperature for this workload.

#### `top_p` (float, optional)
Override global top_p for this workload.

#### `limit` (integer, optional)
Override global sample limit for this workload.

#### `max_tokens` (integer, optional)
Override global max_tokens for this workload.

### Beam Search Specific

#### `use_beam_search` (boolean, required for beam)
Enable beam search decoding. Required for `beam` category workloads.

#### `num_beams` (integer, required for beam)
Number of beams for beam search. Higher = better quality but slower.

#### `best_of` (integer, required for beam)
Number of candidates to generate and rank. Must be ≥ `num_beams`.

### Prefix Specific

#### `system_prompt` (string, optional)
System message prepended to all prompts in this workload (e.g., `"You are a helpful assistant."`).

### Code Specific

#### `best_of` (integer, optional)
Number of candidate solutions to generate. Used for pass@k metrics in code generation.

---

## Available Benchmarks

### SPSR (Shared Prefix Suffix Removal)
- **alpaca** - Instruction following
- **triviaqa** - Trivia questions
- **narrativeqa** - Reading comprehension
- **wikitext** - Language modeling

### Beam Search
- **longbench_gov** - Government report summarization
- **longbench_qmsum** - Meeting summarization
- **narrativeqa** - Reading comprehension with beam search
- **triviaqa** - Trivia with beam search

### Prefix Caching
- **kvprobe** - KV cache efficiency test
- **sharegpt** - Multi-turn conversations
- **leval** - Long context evaluation (requires 130k+ context)
- **longchat** - Extended conversations

### Chat
- **sharegpt** - Conversational AI
- **longchat** - Long-form conversations

### Question Answering
- **triviaqa** - Open-domain QA
- **narrativeqa** - Story-based QA

### Summarization
- **loogle** - Long document summarization
- **longbench_gov** - Government documents
- **longbench_qmsum** - Meeting transcripts

### Code
- **humaneval** - Python code generation (pass@k evaluation)

---

## Example Configuration

```json
{
  "namespace": "usr-myname-namespace",
  "hf_token": "hf_xxxxxxxxxxxxxxxxxxxxx",
  "data_dir": "./data",
  "output_format": "csv",
  "output_file": "./runs/results.csv",
  
  "models": {
    "facebook/opt-125m": {
      "model": "facebook/opt-125m",
      "max_model_len": 2048,
      "endpoint": "http://127.0.0.1:8080",
      "port_local": 8080,
      "port_remote": 8000,
      "api_kind": "completions",
      "max_tokens": 256,
      "temperature": 0.0,
      "top_p": 1.0,
      "qps": 5.0,
      "limit": 10,
      "stream": true,
      "seed": 42,
      "out_dir": "./runs",
      
      "lmcache": {
        "enabled": true,
        "chunk_size": 256,
        "local_cpu": true,
        "max_local_cpu_size": 20
      }
    }
  },
  
  "workloads": {
    "spsr": {
      "alpaca": { "enabled": true, "limit": 50 }
    },
    "qa": {
      "triviaqa": { "enabled": true, "temperature": 0.0 }
    },
    "code": {
      "humaneval": { "enabled": true, "best_of": 10 }
    }
  }
}
```

---

## Output Files

All outputs are saved to the configured `out_dir` (default: `./runs/`):

- **`<category>_<benchmark>.jsonl`** - Individual request/response pairs
- **`<category>_<benchmark>_manifest.json`** - Benchmark metadata and metrics
- **`benchmark_results.{csv|json}`** - Aggregated results across all benchmarks

---

## Docker Mode

Running with `--docker` provides:
- Isolated environment (no host system pollution)
- Automatic dependency installation
- Containerized execution
- Persistent results via volume mounts
- Persistent dataset caching (datasets downloaded once are reused across runs)
- Automatic cleanup after completion

Requirements:
- Docker installed
- Kubernetes config at `~/.kube/config` or `$KUBECONFIG`
- Config file in working directory

The container mounts:
- Kubeconfig (read-only)
- Config file (read-only)
- `./runs/` directory (for output persistence)
- `data_dir` (configured in config, default: `./data`) → `/data` in container (for dataset persistence)

---

## Cleanup Mode

Running with `--cleanup` removes:
- All Kubernetes resources (jobs, pods, services, PVCs, configmaps, secrets)
- Port-forward processes
- Generated `k8s/` directory
- Python cache (`__pycache__`, `*.pyc`, `*.pyo`)
- Virtual environment (`.venv`)
- Dataset cache directory (`data_dir`, default: `./data`) - prompts before deletion

This ensures cached datasets are preserved by default, avoiding re-downloads on subsequent runs.

