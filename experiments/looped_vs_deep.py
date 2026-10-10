#!/usr/bin/env python3
"""Matched study: does looping beat depth at equal parameters, and vice versa?

Two comparisons from the reference article, at teaching scale:

- ``param-matched``: looped (stack applied twice) versus a conventional model
  with the same distinct blocks run once. Same weights count, double the
  per-token compute for the looped arm — the article's core claim.
- ``application-matched``: looped versus a conventional model with as many
  distinct blocks as the looped model has block applications. Same compute,
  twice the parameters for the deep arm — SMELT's framing.

Held-out evaluation uses unseen compositional structures (multi-step programs),
where extra per-token compute should matter most.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra_mini import AstraConfig, AstraMiniLM, ByteTokenizer
from astra_mini.tasks import composition_structures, composition_text


def tiny_config(**overrides) -> AstraConfig:
    values = dict(
        vocab_size=260, dim=96, heads=3, mlp_hidden=256,
        first_blocks=1, stack_blocks=3, last_blocks=1, loops=2,
        depth_policy="fixed", inject_input=True, max_sequence_length=128,
    )
    values.update(overrides)
    return AstraConfig(**values)


def arms() -> dict[str, AstraConfig]:
    return {
        "looped-2x": tiny_config(),
        "deep-same-params": tiny_config(loops=1, inject_input=False),
        "deep-same-applications": tiny_config(loops=1, inject_input=False,
                                              first_blocks=1, stack_blocks=6, last_blocks=1),
    }


def make_batch(tokenizer: ByteTokenizer, structures: list[tuple[str, ...]], *, step: int,
               batch_size: int, sequence_length: int, seed: int, device: torch.device):
    rows = []
    for offset in range(batch_size):
        index = step * batch_size + offset
        text = composition_text(structures[index % len(structures)], index=index, seed=seed)
        ids = tokenizer.encode(text, bos=True, eos=True)
        if len(ids) > sequence_length + 1:
            raise ValueError(f"example is {len(ids)} tokens, limit is {sequence_length + 1}")
        ids += [tokenizer.pad_id] * (sequence_length + 1 - len(ids))
        rows.append(ids)
    values = torch.tensor(rows, dtype=torch.long, device=device)
    labels = values[:, 1:].clone()
    labels[labels == tokenizer.pad_id] = -100
    return values[:, :-1], labels, int((labels != -100).sum())


@torch.no_grad()
def evaluate(model: AstraMiniLM, tokenizer: ByteTokenizer, structures: list[tuple[str, ...]],
             *, seed: int, device: torch.device) -> dict[str, float]:
    model.eval()
    losses, correct, total, exact = [], 0, 0, 0
    for index, structure in enumerate(structures):
        text = composition_text(structure, index=100_000 + index, seed=seed + 999)
        ids = tokenizer.encode(text, bos=True, eos=True)
        values = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
        logits, _ = model(values[:, :-1])
        labels = values[:, 1:]
        target_logits = logits[:, 20:]
        target_labels = labels[:, 20:]
        token_losses = F.cross_entropy(
            target_logits.reshape(-1, model.config.vocab_size), target_labels.reshape(-1), reduction="none",
        )
        losses.extend(token_losses.tolist())
        predicted = target_logits.argmax(dim=-1)
        correct += int((predicted == target_labels).sum())
        total += target_labels.numel()
        exact += int(bool(torch.equal(predicted, target_labels)))
    mean_loss = sum(losses) / len(losses)
    return {
        "target_loss": mean_loss,
        "target_perplexity": math.exp(min(20.0, mean_loss)),
        "target_byte_accuracy": correct / total,
        "target_exact_accuracy": exact / len(structures),
    }


def train_arm(name: str, config: AstraConfig, *, seed: int, steps: int, batch_size: int,
              learning_rate: float, sequence_length: int, device: torch.device,
              train_pool: list, eval_pool: list) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    model = AstraMiniLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    tokenizer = ByteTokenizer()
    curve, tokens_seen = [], 0
    started = time.perf_counter()
    before = evaluate(model, tokenizer, eval_pool, seed=seed, device=device)
    model.train()
    for step in range(steps):
        inputs, labels, tokens = make_batch(
            tokenizer, train_pool, step=step, batch_size=batch_size,
            sequence_length=sequence_length, seed=seed, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += tokens
        if step == 0 or (step + 1) % 10 == 0:
            curve.append({"step": step + 1, "training_loss": float(loss.detach()), "tokens_seen": tokens_seen})
    after = evaluate(model, tokenizer, eval_pool, seed=seed, device=device)
    return {
        "arm": name, "seed": seed, "steps": steps, "batch_size": batch_size,
        "tokens_seen": tokens_seen, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "parameter_counts": model.parameter_counts(), "before": before, "after": after,
        "training_curve": curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seeds", default="11,22")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    structures = composition_structures()
    train_pool, eval_pool = structures[:88], structures[88:120]
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = []
    started = time.perf_counter()
    for name, config in arms().items():
        for seed in seeds:
            row = train_arm(
                name, config, seed=seed, steps=args.steps, batch_size=args.batch_size,
                learning_rate=args.learning_rate, sequence_length=args.sequence_length,
                device=device, train_pool=train_pool, eval_pool=eval_pool,
            )
            rows.append(row)
            print(json.dumps({
                "arm": name, "seed": seed,
                "accuracy": row["after"]["target_byte_accuracy"],
                "loss": row["after"]["target_loss"], "seconds": row["elapsed_seconds"],
            }), flush=True)
    receipt = {
        "schema_version": "1.0", "status": "complete",
        "research_question": "At teaching scale, does looping beat depth at equal parameters, and does depth beat looping at equal applications?",
        "arms": {name: config.to_dict() for name, config in arms().items()},
        "structural_holdout": [list(value) for value in eval_pool],
        "device": device.type, "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seeds": seeds,
        "elapsed_seconds": round(time.perf_counter() - started, 3), "runs": rows,
    }
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
