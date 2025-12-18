"""Benchmark tasks registry and imports."""

from typing import Dict, List

from .alpaca import AlpacaSPSR
from .humaneval import HumanEval
from .kvprobe import PrefixKV
from .leval import LevalPrefix
from .longbench_gov import LongBenchGov, LongBenchGovBeam
from .longbench_qmsum import LongBenchQMSum, LongBenchQMSumBeam
from .longchat import LongChatChat, LongChatPrefix
from .loogle import LooGLE
from .narrativeqa import NarrativeQABeam, NarrativeQASPSR, NarrativeQATask

# Import OutputWriter for export
from .output_writer import OutputWriter
from .sharegpt import ShareGPTChat, ShareGPTPrefix

# Import all task classes
from .triviaqa import TriviaQABeam, TriviaQASPSR, TriviaQATask
from .wikitext import WikitextSPSR


class Workload:
    def __init__(self, name: str, category: str, task_cls):
        self.name = name
        self.category = category
        self.task_cls = task_cls


_REGISTRY: Dict[str, List[Workload]] = {}


def register(category: str, name: str):
    """Decorator to register a task class."""

    def deco(task_cls):
        _REGISTRY.setdefault(category, []).append(Workload(name, category, task_cls))
        return task_cls

    return deco


def list_all() -> Dict[str, List[str]]:
    """List all registered workloads by category."""
    return {cat: [w.name for w in wls] for cat, wls in _REGISTRY.items()}


def get_registry() -> Dict[str, List[Workload]]:
    """Get the full registry."""
    return _REGISTRY


# Register all tasks
register("qa", "triviaqa")(TriviaQATask)
register("qa", "narrativeqa")(NarrativeQATask)

register("summ", "longbench_gov")(LongBenchGov)
register("summ", "longbench_qmsum")(LongBenchQMSum)
register("summ", "loogle")(LooGLE)

register("code", "humaneval")(HumanEval)

register("prefix", "kvprobe")(PrefixKV)
register("prefix", "sharegpt")(ShareGPTPrefix)
register("prefix", "leval")(LevalPrefix)
register("prefix", "longchat")(LongChatPrefix)

register("spsr", "alpaca")(AlpacaSPSR)
register("spsr", "triviaqa")(TriviaQASPSR)
register("spsr", "narrativeqa")(NarrativeQASPSR)
register("spsr", "wikitext")(WikitextSPSR)

register("beam", "longbench_gov")(LongBenchGovBeam)
register("beam", "longbench_qmsum")(LongBenchQMSumBeam)
register("beam", "narrativeqa")(NarrativeQABeam)
register("beam", "triviaqa")(TriviaQABeam)

register("chat", "sharegpt")(ShareGPTChat)
register("chat", "longchat")(LongChatChat)


# Export key classes and functions
__all__ = [
    "Workload",
    "register",
    "list_all",
    "get_registry",
    "OutputWriter",
    # Task classes
    "TriviaQATask",
    "TriviaQASPSR",
    "TriviaQABeam",
    "NarrativeQATask",
    "NarrativeQASPSR",
    "NarrativeQABeam",
    "LongBenchGov",
    "LongBenchGovBeam",
    "LongBenchQMSum",
    "LongBenchQMSumBeam",
    "LooGLE",
    "HumanEval",
    "PrefixKV",
    "ShareGPTPrefix",
    "ShareGPTChat",
    "LevalPrefix",
    "LongChatPrefix",
    "LongChatChat",
    "AlpacaSPSR",
    "WikitextSPSR",
]
