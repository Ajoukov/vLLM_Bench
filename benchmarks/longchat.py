"""LongChat multi-turn conversation benchmarks."""

from typing import Iterable
from .base import Task, Sample, WorkloadOpts
from .evaluators import EMF1Evaluator
from .utils import require


class LongChatPrefix(Task):
    """LongChat-7k-v1.5 multi-turn conversations with shared system prompt."""
    name="longchat"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        # LongChat dataset (lmsys/longchat-lines) is not publicly available
        raise RuntimeError(
            "LongChat dataset (lmsys/longchat-lines) is not available. "
            "This dataset is private or does not exist. "
            "Please use an alternative dataset or disable this workload."
        )


class LongChatChat(Task):
    """LongChat conversations for chatbot evaluation."""
    name="longchat"
    def __init__(self, opts: WorkloadOpts): super().__init__(opts); self.evaluator = EMF1Evaluator(name="null")
    def load(self)->Iterable[Sample]:
        require(["datasets"])
        # LongChat dataset (lmsys/longchat-lines) is not publicly available
        raise RuntimeError(
            "LongChat dataset (lmsys/longchat-lines) is not available. "
            "This dataset is private or does not exist. "
            "Please use an alternative dataset or disable this workload."
        )

