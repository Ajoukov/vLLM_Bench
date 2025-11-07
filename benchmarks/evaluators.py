"""Evaluator classes for different benchmark metrics."""

import os
import sys
import subprocess
import tempfile
from math import comb
from typing import Any, Dict, Tuple

from .base import Evaluator, Sample
from .utils import exact_match, f1_score, _lcs_f1

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None


class EMF1Evaluator(Evaluator):
    """Evaluator for Exact Match and F1 score."""
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
    """Evaluator for ROUGE-L (and optionally ROUGE-1, ROUGE-2) scores."""
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

