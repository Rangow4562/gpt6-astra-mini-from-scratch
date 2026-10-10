# REPORT — a Mini Looped-Transformer "GPT-6 Astra", Measured

All numbers come from real runs of this repository on CPU (torch 2.10.0, Python 3.12, Windows). Every run wrote a versioned receipt; raw receipts live under `runs/` with copies in `artifacts/receipts/`.

## Result in one paragraph

A ~14.6M-parameter looped-transformer language model — a shared 6-block stack applied twice inside a first/last-block sandwich (MoR layout), with anchor re-injection (latent-reasoning style) and three selectable per-token depth policies — was pretrained for 150 steps on compositional multi-step programs (loss 5.62 → 0.21, 229,746 tokens, 338 s, loop counts randomized per batch). After brief supervised adaptation to a synthetic coding benchmark (loss 6.44 → 0.03 in 60 steps), the model greedily solved 7 of 8 held-out families. The flagship measurement: **holding the checkpoint fixed and raising the test-time loop budget from 1 to 2 took held-out pass@1 from 25% (2/8) to 87.5% (7/8)** — including flipping `return x + 2` into the correct `return x + 1` for the same unseen task. Matched-budget depth comparisons and depth-policy comparisons at this tiny scale found no looped advantage, consistent with the literature's smallest-scale behavior.

## Architecture (~14.6M parameters)

| Component | Setting |
|---|---|
| Layout | first block → shared stack ×`loops` → last block (MoR sandwich) |
| Shared stack | 6 blocks, dim 384, 6 heads, SwiGLU hidden 1024, pre-norm RMSNorm, RoPE |
| Loop | 2 passes, weights shared; anchor re-injection per pass |
| Depth policies | `fixed` / `exit_gate` (Ouro-style) / `router` (MoR token-choice) |
| KV cache | separate per (pass, block); never shared across passes |
| Tokenizer | 1 token per UTF-8 byte (vocab 260, tied head) |
| Effective depth | 1 + 6×2 + 1 = 16 block applications |

`test_weight_sharing_loops_do_not_add_parameters` proves `loops=1` and `loops=2` models have identical parameter counts; `deep_block_equivalent_parameters` reports what the unrolled depth would cost conventionally.

## Stage 1 — pretraining with randomized loop counts

`scripts/train_pretrain.py --output runs/pretrain-001 --steps 150 --batch-size 16 --randomize-loops --seed 42`

| Metric | Value |
|---|---|
| Loss (first → last) | 5.618 → 0.212 |
| Tokens seen | 229,746 |
| Elapsed | 338.2 s (CPU) |
| Receipt | `runs/pretrain-001/training-receipt.json` |

The active loop count alternates 1/2 per batch (Geiping-style), preparing the model for variable inference budgets. Pretraining uses three-step compositional programs (`Task: start with x. Add 4. Multiply by 3. Negate it.` → `(-(x + 4) * 3)`) — multi-step targets where per-token compute should matter.

## Stage 2 — supervised coding adaptation

`scripts/train_sft.py --initial-checkpoint runs/pretrain-001/checkpoint-0150 --steps 60` (answer-only loss on coding-task completions; loss 6.44 → 0.028).

Greedy pass@1 on one unseen task per family, `confirm` split: **7/8** — everything except `reverse`, which produced the invalid `return text[:::-1]` (three colons).

## Stage 3 — recursive self-improvement, and the interference tax

RSI ladder on `reverse` (`runs/self-improve-reverse-001`): the policy sampled 16 completions per task, the verifier kept 4 exact passes (`text[::-1]` among them), and one fine-tune cycle early-stopped the ladder (dev pool saturated at 1.0).

| Checkpoint | reverse | other 7 families | Total |
|---|---|---|---|
| After SFT | 0 | 7 passing | 7/8 |
| After RSI cycle 1 | **1** | 0 passing | 1/8 |

The self-repair worked (`text[:::-1]` → `text[::-1]` on the held-out task) but full-model fine-tuning on 4 examples catastrophically interfered with everything else — the `reverse` pattern leaked into `square` (`return reverse r(text):...`) and `increment`. A gentler ladder at learning rate 1e-5 (`runs/self-improve-reverse-002-gentle`) did **not** mitigate: 0/8. At this scale the interference is a property of the recipe (tiny verified memory, full-model updates), not of the learning rate. The actionable next steps — replay buffers, adapters, or frozen-backbone heads — are not implemented here.

## The flagship measurement — test-time loop scaling

`scripts/evaluate_loop_scaling.py --checkpoint runs/coding-sft-001/checkpoint-0060 --split confirm --per-family 1`

| Loop budget | Held-out pass@1 | Valid programs |
|---:|---:|---:|
| 1 loop (8 applications) | 25.0% (2/8) | 5/8 |
| 2 loops (16 applications) | **87.5% (7/8)** | 7/8 |

Same weights, same prompt, same seed — only the loop budget changed. Greedy decoding of the `increment` task flips from `return x + 2` (wrong) to `return x + 1` (correct). This is the latent-reasoning paper's inference-time compute dial, working end to end, enabled by training with randomized loop counts.

## Matched-budget comparisons (tiny scale — honest nulls)

`experiments/looped_vs_deep.py` and `experiments/loop_depth_policy.py`, 120 steps, batch 16, seeds {11, 22}, held-out compositional structures:

| Arm | Seed 11 | Seed 22 |
|---|---:|---:|
| looped-2x (5 blocks, 8+1 applications) | 80.4% | 53.3% |
| deep-same-params (5 blocks, once) | **88.2%** | **88.3%** |
| deep-same-applications (8 blocks, once) | **88.4%** | 86.6% |
| policy: fixed | 80.4% | 53.3% |
| policy: exit-gate | 67.3% | 60.4% |
| policy: router | 55.7% | 76.1% |

No significant differences (paired sign tests, all p ≥ 0.5 with Holm correction) — and the looped arm is markedly less stable across seeds at this width and learning rate. This reproduces, at teaching scale, the smallest-model behavior in MoR Figure 11: "just looking at the smaller model, we would have drawn the opposite conclusion." The loop-scaling result above shows the looped model's depth still converts to accuracy when the budget grows; whether looping *beats* deep at matched budgets requires larger scale than this repo targets.

## Method guarantees

- **Determinism** — explicit seeds everywhere; tasks from pure integer hashes.
- **Fail-closed** — no overwrites; NaN hard-fails; unknown splits/families/loop budgets rejected.
- **Receipts** — versioned JSON with SHA-256 weight hashes for every entry point.
- **Cache correctness** — cached decoding proven bit-for-bit equal to full recomputation; per-pass caches proven unshared.

## Limitations

14.6M parameters, 192-byte context, synthetic programs, single-seed comparisons at teaching scale. Adaptive policies bring no compute savings (all passes run; Ouro's own caveat). RSI interference is documented, not solved. Nothing here establishes that looping explains GPT-6 Astra's capabilities — the rumor remains a rumor.
