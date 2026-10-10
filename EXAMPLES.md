# EXAMPLES — real generations from the looped model

Every completion below is a real greedy output of checkpoints in this repository, transcribed from the evaluation payloads and receipts (`runs/confirm-*.json`, `artifacts/receipts/`). No cherry-picking beyond the tasks named in `REPORT.md`.

## The headline: the same task, 1 loop vs 2 loops

Checkpoint `runs/coding-sft-001/checkpoint-0060`, task `confirm-increment-000`, greedy, identical seed — only the test-time loop budget changes:

**1 loop (8 block applications)** — wrong:

```python
    return x + 2
```

**2 loops (16 block applications)** — correct:

```python
    return x + 1
```

Held-out pass@1 across all eight families moves from 2/8 to 7/8 just by letting the model think twice per token. That is recurrent depth doing exactly what the Astra reporting suggested.

## Self-repair by the model's own verified programs

RSI ladder on the one failing family, task `confirm-reverse-000`:

**Before** (after SFT) — invalid, an off-by-one in the slice:

```python
    return text[:::-1]
```

**After** (RSI cycle 1, trained on 4 self-verified programs) — correct:

```python
    return text[::-1]
```

The model wrote its own repair dataset: sampled candidates, kept the ones the executable verifier approved, and fine-tuned on them.

## The interference tax, in plain sight

Same checkpoints, tasks that passed before the ladder:

Task `confirm-square-000`:

**After SFT** — correct:

```python
    return x * x
```

**After RSI** — the reverse pattern leaked everywhere:

```python
    reverse r(text):
    return text):
```

Task `confirm-increment-000`:

**After SFT** — correct:

```python
    return x + 1
```

**After RSI**:

```python
    return reverse text sxtrn f f 1
```

A gentler ladder (learning rate 1e-5) did not fix this. Tiny verified memory + full-model updates = catastrophic forgetting; see `REPORT.md` for the measured scoreboard and the next steps this repo does not implement.

## Scoreboard (greedy pass@1, one unseen task per family)

| Checkpoint | Solved |
|---|---:|
| Pretrained on compositions only | 0/8 |
| + coding SFT (60 steps) | 7/8 |
| + RSI on `reverse` (lr 1e-4) | 1/8 (reverse fixed, rest forgotten) |
| + RSI gentle (lr 1e-5) | 0/8 |
| Same SFT checkpoint, 1 loop instead of 2 | 2/8 |

The last row is the point of this repository: with a looped model, inference-time compute is a real dial.
