"""Utility functions for benchmarking."""

import re
import sys
from typing import List

try:
    import tiktoken
except Exception:
    tiktoken = None


def require(pkgs: List[str]):
    """Check for required dependencies."""
    try:
        import datasets as hfds
    except Exception:
        hfds = None
    try:
        from rouge_score import rouge_scorer
    except Exception:
        rouge_scorer = None
    
    missing = [p for p in pkgs if p == "datasets" and hfds is None or p == "rouge-score" and rouge_scorer is None]
    if missing:
        print(f"Missing deps: {missing}. pip install " + " ".join(missing), file=sys.stderr)
        sys.exit(2)


def _to_text(x):
    """Extract text from various data structures."""
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
    """Normalize answer text for comparison."""
    s = s.lower()
    s = _PUN.sub(" ", s)
    s = " ".join(w for w in _WS.sub(" ", s).strip().split() if w not in _ART)
    return s


def f1_score(pred: str, golds: List[str]) -> float:
    """Calculate token-level F1 score."""
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
    """Check if prediction exactly matches any gold answer."""
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)


def model_tokenizer(model_id: str):
    """Get tokenizer for a model (heuristic mapping to tiktoken encoding)."""
    if not tiktoken: return None
    try:
        return tiktoken.encoding_for_model(model_id)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, enc) -> int:
    """Count tokens using the provided encoder, fallback to whitespace."""
    if enc is None: return len(re.findall(r"\S+", text))
    try: return len(enc.encode(text))
    except Exception: return len(re.findall(r"\S+", text))


_ANS_PATTERNS = [
        re.compile(r"(?i)^\s*final\s*answer\s*[:\-]\s*(.+)$"),
        re.compile(r"(?i)^\s*answer\s*[:\-]\s*(.+)$"),
        ]


def _short_answer(pred: str) -> str:
    """Extract short answer from prediction text."""
    # take first non-empty line
    line = next((ln.strip() for ln in pred.splitlines() if ln.strip()), pred.strip())
    for rx in _ANS_PATTERNS:
        m = rx.search(line)
        if m: line = m.group(1).strip(); break
    # remove trailing punctuation and explanations after a period/semicolon
    line = re.split(r"[.;\n]", line, maxsplit=1)[0].strip()
    return line


def _lcs_f1(a: str, b: str)->float:
    """Calculate F1 based on longest common subsequence."""
    A=a.split(); B=b.split()
    dp=[[0]*(len(B)+1) for _ in range(len(A)+1)]
    for i in range(len(A)):
        for j in range(len(B)):
            dp[i+1][j+1]=dp[i][j]+1 if A[i]==B[j] else max(dp[i][j+1], dp[i+1][j])
    lcs=dp[-1][-1]; 
    if lcs==0: return 0.0
    p=lcs/max(1,len(B)); r=lcs/max(1,len(A)); return (2*p*r)/(p+r)

