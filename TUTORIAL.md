# The Beginner Tutorial: Build a Looped Transformer

Each step runs on CPU. Everything is deterministic — run it twice, get identical receipts.

## 0. Setup

```bash
cd gpt6-astra-mini
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q              # 30 tests should pass
```

## 1. The one-sentence architecture

A normal transformer runs each block once; a looped transformer runs the **same blocks several times** — extra thinking per token, no extra weights. This repo wraps the shared stack between a first and a last block (MoR-style), re-injects the token's grounded representation on every pass (latent-reasoning style), and lets a gate or router decide how many passes each position uses (Ouro / Universal-Transformer / MoR style).

```python
import torch
from astra_mini import AstraConfig, AstraMiniLM

model = AstraMiniLM(AstraConfig())
print(model.config.effective_depth)          # 16 block applications
print(model.parameter_counts())
# {'total': ..., 'distinct_block_parameters': ...,
#  'block_applications_per_token': 16,
#  'deep_block_equivalent_parameters': 2 x distinct}
```

Same weights, twice the depth. That is the whole trick the Astra reporting is about.

## 2. Train it

```bash
run_cpu.bat scripts\train_pretrain.py --output runs\tutorial --steps 30 --checkpoints 30 --batch-size 8 --randomize-loops --seed 42
```

`--randomize-loops` samples the active loop count per batch (latent-reasoning style), preparing the model to work with different compute budgets at inference. The receipt records `active_loops` and the policy's `mean_depth` per step.

## 3. Watch it think longer at inference

```bash
run_cpu.bat scripts\evaluate_loop_scaling.py --checkpoint runs\tutorial\checkpoint-0030 --split dev --per-family 1 --output runs\loop-scaling.json
```

You get accuracy at 1 loop and at the full loop budget — the test-time compute curve from the article, on your machine.

## 4. The three depth policies

```bash
run_cpu.bat experiments\loop_depth_policy.py --output artifacts\experiments\depth-policy --seeds 11 --steps 60
run_cpu.bat experiments\analyze_loop_experiments.py --results artifacts\experiments\depth-policy\results.json --output artifacts\experiments\depth-policy\analysis.json --higher-is-better
```

`fixed` takes the last pass; `exit_gate` halts each position when its cumulative exit probability crosses 0.5; `router` assigns pass counts up front, token-choice style.

## 5. Looped vs deep, both ways

```bash
run_cpu.bat experiments\looped_vs_deep.py --output artifacts\experiments\looped-vs-deep --seeds 11 --steps 60
```

- **param-matched**: same distinct blocks, looped runs them twice.
- **application-matched**: same compute, deep owns twice the blocks.

## 6. Coding tasks and the verifier

Same fail-closed rules as its sibling repo: one function, no imports/loops/attributes/recursion, allowlisted calls, hidden tests in a restricted namespace. Evaluation:

```bash
run_cpu.bat scripts\evaluate.py --checkpoint runs\tutorial\checkpoint-0030 --split dev --per-family 1 --output runs\eval.json
```

## 7. Read the internals

- `LOOPING.md` — every article concept mapped onto this codebase.
- `astra_mini/depth.py` — 84 lines; the entire policy design space.
- `astra_mini/model.py` — the loop, the caches, the sandwich.
- `tests/test_runtime.py::test_cache_is_per_pass_not_shared` — why the KV cache does not shrink.

## 8. Slides

```bash
python slides/serve_slides.py --port 8765
```
