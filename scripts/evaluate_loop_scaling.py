#!/usr/bin/env python3
"""Measure held-out accuracy as the test-time loop budget grows.

The Geiping et al. result: models trained (optionally with randomized loop
counts) can spend more loops at inference. This script evaluates one
checkpoint at every loop count from 1 to the configured maximum and writes
the scaling curve as a receipt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra_mini import ByteTokenizer
from astra_mini.evaluator import evaluate_source
from astra_mini.runtime import generate_group, load_checkpoint, resolve_device
from astra_mini.tasks import frozen_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "final", "confirm"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--families", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = resolve_device(args.device)
    tokenizer = ByteTokenizer()
    model = load_checkpoint(args.checkpoint, device)
    tasks = frozen_tasks(args.split, per_family=args.per_family)
    selected_families = {value.strip() for value in args.families.split(",") if value.strip()}
    if selected_families:
        tasks = [task for task in tasks if task.family in selected_families]
    curve = []
    started = time.perf_counter()
    for loops in range(1, model.config.loops + 1):
        budget_started = time.perf_counter()
        solved = 0
        valid = 0
        for index, task in enumerate(tasks):
            generated = generate_group(
                model, tokenizer, task, group_size=1, max_new_tokens=args.max_new_tokens,
                temperature=1.0, sample=False, seed=args.seed + index, active_loops=loops,
            )[0]
            evaluation = evaluate_source(task, task.prompt + generated["completion"])
            solved += int(evaluation.passed)
            valid += int(evaluation.status != "invalid")
        row = {
            "loops": loops, "solved": solved, "tasks": len(tasks),
            "pass_rate": solved / len(tasks), "valid_programs": valid,
            "seconds": round(time.perf_counter() - budget_started, 3),
        }
        curve.append(row)
        print(json.dumps(row), flush=True)
    receipt = {
        "schema_version": "1.0", "status": "complete", "stage": "loop_scaling",
        "checkpoint": str(args.checkpoint), "max_loops": model.config.loops,
        "split": args.split, "per_family": args.per_family,
        "families": sorted(selected_families) if selected_families else "all",
        "seed": args.seed, "curve": curve,
        "elapsed_seconds": round(time.perf_counter() - started, 3), "device": device.type,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "curve": curve}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
