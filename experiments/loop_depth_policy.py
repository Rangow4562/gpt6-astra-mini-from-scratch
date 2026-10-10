#!/usr/bin/env python3
"""Matched study: fixed vs exit-gate vs router depth policies on the looped stack.

All three arms share the identical looped architecture and training budget;
only the policy that decides which pass supplies each position's output
differs. Evaluation uses unseen compositional structures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.looped_vs_deep import (
    composition_structures,
    evaluate,
    make_batch,
    tiny_config,
    train_arm,
)
from astra_mini import AstraConfig


def policy_arms() -> dict[str, AstraConfig]:
    return {
        "fixed": tiny_config(depth_policy="fixed"),
        "exit-gate": tiny_config(depth_policy="exit_gate"),
        "router": tiny_config(depth_policy="router"),
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
    for name, config in policy_arms().items():
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
        "research_question": "Do learned depth policies (exit gate, router) beat a fixed loop at matched compute?",
        "arms": {name: config.to_dict() for name, config in policy_arms().items()},
        "structural_holdout": [list(value) for value in eval_pool],
        "device": device.type, "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seeds": seeds,
        "elapsed_seconds": round(time.perf_counter() - started, 3), "runs": rows,
    }
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
