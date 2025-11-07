"""Dataset loading functions for various benchmarks."""

import os
import json
import zipfile

from .utils import require

try:
    import datasets as hfds
except Exception:
    hfds = None


def _hf_load(name, config, split, cache_dir):
    """Load dataset from HuggingFace Hub."""
    require(["datasets"])
    return hfds.load_dataset(name, config, split=split, cache_dir=cache_dir) if config else hfds.load_dataset(name, split=split, cache_dir=cache_dir)


def _load_longbench_manual(task_name: str, cache_dir: str):
    """
    Manual loader for LongBench dataset (dataset scripts no longer supported).
    Downloads data.zip from HuggingFace Hub and loads JSONL files.
    """
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

