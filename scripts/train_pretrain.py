#!/usr/bin/env python3
"""Pretrain the looped Astra model from random initialization."""
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

from astra_mini import AstraConfig, AstraMiniLM, ByteTokenizer
from astra_mini.runtime import resolve_device, save_checkpoint
from astra_mini.tasks import pretraining_text


def batch_for(tokenizer: ByteTokenizer, *, step: int, batch_size: int, sequence_length: int,
              seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows = []
    for offset in range(batch_size):
        text = pretraining_text(step * batch_size + offset, seed=seed)
        ids = tokenizer.encode(text, bos=True, eos=True)
        if len(ids) > sequence_length + 1:
            raise ValueError(f"training example exceeds sequence length: {len(ids)}")
        ids += [tokenizer.pad_id] * (sequence_length + 1 - len(ids))
        rows.append(ids)
    values = torch.tensor(rows, dtype=torch.long, device=device)
    inputs, labels = values[:, :-1], values[:, 1:]
    labels = labels.masked_fill(labels == tokenizer.pad_id, -100)
    return inputs, labels, int((labels != -100).sum().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--checkpoints", default="50,100,150")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--stack-blocks", type=int, default=6)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--depth-policy", choices=("fixed", "exit_gate", "router"), default="fixed")
    parser.add_argument("--no-inject-input", action="store_true",
                        help="disable the Geiping-style anchor re-injection")
    parser.add_argument("--randomize-loops", action="store_true",
                        help="sample the active loop count per batch (Geiping-style training)")
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
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    tokenizer = ByteTokenizer()
    config = AstraConfig(
        dim=args.dim, stack_blocks=args.stack_blocks, loops=args.loops,
        depth_policy=args.depth_policy, inject_input=not args.no_inject_input,
        max_sequence_length=max(192, args.sequence_length),
    )
    model = AstraMiniLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    counts = model.parameter_counts()
    initial = save_checkpoint(model, args.output / "checkpoint-0000", {
        "stage": "random_initialization", "seed": args.seed, "parameter_counts": counts,
    })
    checkpoint_rows = [dict(step=0, **initial)]
    curve: list[dict] = []
    tokens_seen = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    loop_rng = random.Random(args.seed + 1)
    for step in range(1, args.steps + 1):
        step_started = time.perf_counter()
        active_loops = loop_rng.randint(1, args.loops) if args.randomize_loops else None
        inputs, labels, tokens = batch_for(
            tokenizer, step=step - 1, batch_size=args.batch_size,
            sequence_length=args.sequence_length, seed=args.seed, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, diagnostics = model(inputs, active_loops=active_loops)
            language_loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100
            )
        if not bool(torch.isfinite(language_loss).item()):
            raise FloatingPointError("non-finite loss")
        language_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        tokens_seen += tokens
        row = {
            "step": step,
            "loss": float(language_loss.detach()),
            "active_loops": active_loops or args.loops,
            "mean_depth": round(diagnostics["mean_depth"], 4),
            "gradient_norm": round(float(gradient_norm.detach()), 4),
            "tokens": tokens,
            "tokens_seen": tokens_seen,
            "seconds": round(time.perf_counter() - step_started, 4),
        }
        curve.append(row)
        if step == 1 or step % 10 == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if step in checkpoints:
            saved = save_checkpoint(model, args.output / f"checkpoint-{step:04d}", {
                "stage": "looped_pretraining", "seed": args.seed, "step": step,
                "tokens_seen": tokens_seen, "parameter_counts": counts,
            })
            checkpoint_rows.append(dict(step=step, **saved))
    if device.type == "cuda":
        torch.cuda.synchronize()
    receipt = {
        "schema_version": "1.0",
        "status": "complete",
        "stage": "looped_pretraining",
        "config": config.to_dict(),
        "parameter_counts": counts,
        "randomize_loops": args.randomize_loops,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "learning_rate": args.learning_rate,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "device": device.type,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if device.type == "cuda" else None,
        "checkpoints": checkpoint_rows,
        "training_curve": curve,
    }
    (args.output / "training-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "tokens_seen", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
