#!/usr/bin/env python3
"""Supervised adaptation: teach the pretrained model the coding-task format.

Answer-only cross-entropy on procedurally generated coding tasks (prompt and
special tokens are masked out), so the recursive self-improvement ladder starts
from a model that already emits structurally valid completions.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra_mini import ByteTokenizer
from astra_mini.runtime import load_checkpoint, resolve_device, save_checkpoint
from astra_mini.tasks import FAMILIES, FAMILY_BY_NAME, make_task, frozen_tasks


def sft_batch(tokenizer: ByteTokenizer, *, step: int, batch_size: int, seed: int,
              device: torch.device, per_family: int = 8) -> tuple[torch.Tensor, torch.Tensor, int]:
    """One batch of (prompt, reference completion) rows with prompt masked out."""
    rows = []
    prompt_lengths = []
    for offset in range(batch_size):
        index = step * batch_size + offset
        family = FAMILIES[index % len(FAMILIES)]
        task = make_task(family, split="sft", index=index, seed=seed + (index % 7) * 1013)
        prompt_ids = tokenizer.encode(task.prompt, bos=True)
        completion_ids = tokenizer.encode(task.reference_completion, eos=True)
        ids = prompt_ids + completion_ids
        if len(ids) > 192:
            raise ValueError(f"sft example exceeds context: {len(ids)}")
        prompt_lengths.append(len(prompt_ids) - 1)
        rows.append(ids)
    width = max(len(row) for row in rows)
    padded = []
    for row in rows:
        padded.append(row + [tokenizer.pad_id] * (width - len(row)))
    values = torch.tensor(padded, dtype=torch.long, device=device)
    inputs, labels = values[:, :-1], values[:, 1:].clone()
    labels[labels == tokenizer.pad_id] = -100
    # Mask the prompt: loss counts only completion bytes (and EOS).
    for row_index, prompt_length in enumerate(prompt_lengths):
        labels[row_index, :prompt_length] = -100
    return inputs, labels, int((labels != -100).sum().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--checkpoints", default="60")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    checkpoints = {int(value) for value in args.checkpoints.split(",") if value}
    if max(checkpoints) != args.steps:
        raise ValueError("final checkpoint must equal steps")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    tokenizer = ByteTokenizer()
    model = load_checkpoint(args.initial_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    curve = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        inputs, labels, tokens = sft_batch(tokenizer, step=step - 1, batch_size=args.batch_size,
                                           seed=args.seed, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                               labels.reshape(-1), ignore_index=-100)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite SFT loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        curve.append({"step": step, "loss": float(loss.detach()), "tokens": tokens})
        if step == 1 or step % 10 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)
        if step in checkpoints:
            save_checkpoint(model, args.output / f"checkpoint-{step:04d}", {
                "stage": "coding_sft", "seed": args.seed, "step": step,
                "source_checkpoint": str(args.initial_checkpoint),
            })
    receipt = {
        "schema_version": "1.0", "status": "complete", "stage": "coding_sft",
        "source_checkpoint": str(args.initial_checkpoint), "seed": args.seed,
        "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": round(time.perf_counter() - started, 3), "device": device.type,
        "training_curve": curve,
    }
    (args.output / "training-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
