# LOOPING.md — the looped transformer, end to end

This repository is a teaching-scale implementation of **recurrent depth** — the "looped transformer" design reported for OpenAI's GPT-6 Astra and studied across a line of public research. Reference: Sebastian Raschka, "GPT-6 Astra, Looped Transformers, and Hidden Reasoning" (Sep 2026) and the papers it surveys. This document maps every concept onto code in this repo.

## 1. The core idea: reuse the weights

A conventional transformer applies each block once. A looped transformer applies the **same blocks multiple times**, so effective depth grows while the parameter count does not:

```text
conventional:  block1 → block2 → ... → blockN          (N applications, N weight sets)
looped:        [block1 → ... → blockN] → [block1 → ... → blockN]
               (2N applications, still N weight sets)
```

The loop is *across depth*, not across time — this is what separates looped transformers from RNNs, which reuse weights across sequence steps. Attention still moves information between positions on every pass.

In this repo: `astra_mini/model.py`, `AstraMiniLM._run_stack` applies `self.stack` exactly `active_loops` times. The test `test_weight_sharing_loops_do_not_add_parameters` proves a `loops=2` model and a `loops=1` model have identical parameter counts.

## 2. The sandwich (Mixture-of-Recursions layout)

Naive looping re-applies *everything*. MoR (2025) sandwiches the shared recursion block between two distinct blocks: **Layer 0** grounds the token in its embedding context once, the **recursion block** does the repeated thinking, and **Layer L−1** prepares the final representation. This repo adopts that layout:

```text
first block (Layer 0)   — distinct weights
shared stack (6 blocks) — applied `loops` times
last block  (Layer L−1) — distinct weights
```

## 3. Anchor re-injection (latent-reasoning style)

The Geiping et al. latent-reasoning model feeds the stack its *original* input on every pass: each loop, the previous loop's hidden state is concatenated with the initial blocks' output and passed through a learned linear projection. Without this, deep loops can drift; with it, every pass stays anchored to the token's grounding.

In this repo: `AstraConfig.inject_input=True` adds `self.reinject = Linear(2·dim, dim)`; each loop runs `state = reinject([anchor ; state])`. Disable with `--no-inject-input` to compare.

## 4. Choosing the depth: three policies

| Policy | Origin | Mechanism | Code |
|---|---|---|---|
| `fixed` | Nanbeige4.2, Astra rumor | every position uses the last pass | `_apply_depth_policy` early return |
| `exit_gate` | Ouro + Universal Transformer | per-pass gate logits → cumulative probability ≥ threshold picks the supplying pass | `depth.ExitGate`, `hard_select` |
| `router` | MoR (token-choice) | one router on the anchor assigns each position its pass count up front | `depth.DepthRouter` |

Design decisions worth understanding:

- **Soft training, hard inference.** Hard pass selection is not differentiable, so training uses the soft mixture `Σᵣ pᵣ · stateᵣ`; inference switches to the hard cumulative-threshold selection. This mirrors Ouro, whose released implementation also computes all passes and selects afterwards.
- **No free compute.** Unlike MoR's expert-choice routing, no tokens are skipped here — all passes always run. The policy changes *which representation is read*, not how much compute is spent. Real compute savings require MoR-style KV sharing and are out of scope.
- **Carry-forward halting** (Universal Transformer Figure 8) is the conceptual ancestor of `hard_select`: positions "halt" at the pass where their cumulative exit probability crosses the threshold.

## 5. Test-time loop scaling

Because loop count is just a runtime parameter, a trained model can spend **more loops at inference** — the latent-reasoning paper's central trick (they train with randomized loop counts, then choose 8/32/64 at inference). This repo supports it directly:

- Train with `--randomize-loops` to sample the active loop count per batch.
- Evaluate with `--loops k` (or `scripts/evaluate_loop_scaling.py`) to trade compute for accuracy.

`active_loops` bounds-checks `1 ≤ k ≤ config.loops` and the exit gate/router handle reduced pass sets gracefully.

## 6. The KV cache is per pass

Looping does **not** shrink the KV cache: pass 1 and pass 2 see different inputs, so block 1's keys/values differ between passes and each pass needs its own cache — a 2-loop model caches like a 2N-block conventional model. Nanbeige tried sharing caches across passes; it halved memory but hurt quality, so they shipped separate caches.

In this repo: `CausalAttention` carries an optional `past_kv` per block; `AstraMiniLM.init_cache()` builds a separate cache bundle for every (pass, block) pair. The test `test_cache_is_per_pass_not_shared` asserts the two passes' caches differ; `test_cached_generation_matches_full_forward_exactly` proves cached decoding is numerically identical to full recomputation.

## 7. What looping buys (and what it doesn't)

From the research the article surveys:

- **Reasoning improves** — multi-step problems benefit from extra per-token compute even when parameter count is fixed (Virtual Logic Depth).
- **Knowledge storage does not** — memorization tracks distinct parameters, not applications; looping adds compute, not capacity.
- **Matched compute favors looping at scale** — SMELT estimates 6.8–18% training-compute savings for the same validation loss; but at the *smallest* model scale vanilla wins (MoR Figure 11).
- **Shorter hidden "thinking"** — full-bandwidth transformer work suggests latent feedback shortens reasoning traces in base models.

This repo reproduces the *measurement methodology* at teaching scale in `experiments/looped_vs_deep.py` and `experiments/loop_depth_policy.py`, and reports tiny-scale results exactly as they land — including null or negative ones.

## 8. Does this hide chains of thought?

The article's answer, which we adopt as documentation rather than speculation: looped transformers add computation *inside* the forward pass, while reasoning traces are *tokens* the model emits. Fewer tokens at fixed accuracy is a capability effect, not an architecture conspiracy. The trace was never a complete description of the computation anyway. See the article's §5 for the full argument.

## Map: article concept → this repository

| Article concept | Where here |
|---|---|
| Shared stack applied N times (Nanbeige) | `model.py :: _run_stack` |
| Layer 0 / recursion / Layer L−1 (MoR) | `first` / `stack` / `last` ModuleLists |
| Exit gate + cumulative threshold (Ouro/UT) | `depth.py :: ExitGate, hard_select` |
| Token-choice router (MoR) | `depth.py :: DepthRouter` |
| Anchor re-injection (Geiping) | `model.py :: reinject` |
| Randomized loop counts in training | `train_pretrain.py --randomize-loops` |
| Test-time loop budgets | `--loops` flags, `evaluate_loop_scaling.py` |
| Per-pass KV caches | `CausalAttention(past_kv)`, `init_cache` |
| Matched-compute comparisons (SMELT/MoR) | `experiments/looped_vs_deep.py` |
| Per-policy comparison (MoR/UT/Ouro) | `experiments/loop_depth_policy.py` |
