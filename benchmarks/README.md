# Benchmarks Module

This directory contains all the benchmark tasks separated into individual modules for better organization and maintainability.

## Structure

### Core Modules

- **`base.py`** - Base classes and data types (`Task`, `Sample`, `Evaluator`, `WorkloadOpts`)
- **`evaluators.py`** - Evaluation metrics (`EMF1Evaluator`, `RougeLEvaluator`, `HumanEvalEvaluator`)
- **`utils.py`** - Utility functions (text processing, tokenization, scoring)
- **`loaders.py`** - Dataset loading functions (HuggingFace, manual loaders)

### Benchmark Task Modules

Each benchmark is in its own file for easy maintenance:

- **`triviaqa.py`** - TriviaQA question-answering benchmark (QA, SPSR, Beam variants)
- **`narrativeqa.py`** - NarrativeQA reading comprehension (QA, SPSR, Beam variants)
- **`longbench_gov.py`** - LongBench government report summarization (Summ, Beam variants)
- **`longbench_qmsum.py`** - LongBench meeting summarization (Summ, Beam variants)
- **`loogle.py`** - LooGLE legal text summarization
- **`humaneval.py`** - HumanEval code completion benchmark
- **`kvprobe.py`** - Prefix caching probe benchmark
- **`sharegpt.py`** - ShareGPT conversation benchmarks (Prefix, Chat variants)
- **`leval.py`** - LEval long-document comprehension
- **`longchat.py`** - LongChat multi-turn conversations (Prefix, Chat variants)
- **`alpaca.py`** - Alpaca instruction-following
- **`wikitext.py`** - Wikitext language modeling

### Registry

- **`__init__.py`** - Module initialization, task registration, and exports

## Categories

The benchmarks are organized into the following categories:

- **qa** - Question Answering tasks
- **summ** - Summarization tasks
- **code** - Code completion tasks
- **prefix** - Prefix caching / shared prefix tasks
- **spsr** - Single-Prompt-Single-Response tasks
- **beam** - Beam search evaluation tasks
- **chat** - Chatbot evaluation tasks

## Usage

The benchmarks module is automatically imported by `bench.py`. All tasks are registered and available through the standard CLI:

```bash
# List all benchmarks
./bench.py list

# Run a specific benchmark
./bench.py run config.json --model MODEL_NAME spsr alpaca

# Run all benchmarks in a category
./bench.py run config.json --model MODEL_NAME beam
```

## Adding New Benchmarks

To add a new benchmark:

1. Create a new file in this directory (e.g., `mybenchmark.py`)
2. Import required classes from `base`, `evaluators`, `loaders`, and `utils`
3. Define your task class(es) inheriting from `Task`
4. Import and register your task in `__init__.py`

Example:

```python
# mybenchmark.py
from typing import Iterable
from .base import Task, Sample, WorkloadOpts
from .evaluators import EMF1Evaluator
from .loaders import _hf_load
from .utils import _to_text

class MyBenchmark(Task):
    name = "mybenchmark"
    def __init__(self, opts: WorkloadOpts):
        super().__init__(opts)
        self.evaluator = EMF1Evaluator()
    
    def load(self) -> Iterable[Sample]:
        # Load and yield samples
        pass
```

Then in `__init__.py`:

```python
from .mybenchmark import MyBenchmark
register("qa", "mybenchmark")(MyBenchmark)
```

