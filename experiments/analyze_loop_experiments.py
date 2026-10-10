#!/usr/bin/env python3
"""Paired analysis of loop experiments: sign tests against a baseline arm + Holm."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def exact_sign_test(differences: list[float]) -> float:
    wins = sum(d > 0 for d in differences)
    losses = sum(d < 0 for d in differences)
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (len(p_values) - rank)))
        adjusted[index] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metric", choices=("target_byte_accuracy", "target_loss"), default="target_byte_accuracy")
    parser.add_argument("--baseline", default=None, help="baseline arm (default: first arm)")
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.results.read_text())
    rows = receipt["runs"]
    arms = sorted({row["arm"] for row in rows})
    baseline = args.baseline or arms[0]
    baseline_rows = {row["seed"]: row for row in rows if row["arm"] == baseline}
    comparisons = []
    for arm in arms:
        if arm == baseline:
            continue
        arm_rows = {row["seed"]: row for row in rows if row["arm"] == arm}
        seeds = sorted(set(arm_rows) & set(baseline_rows))
        differences = []
        for seed in seeds:
            left = arm_rows[seed]["after"][args.metric]
            right = baseline_rows[seed]["after"][args.metric]
            differences.append(left - right if args.higher_is_better else right - left)
        comparisons.append({
            "arm": arm, "baseline": baseline, "seeds": seeds,
            "arm_values": [arm_rows[s]["after"][args.metric] for s in seeds],
            "baseline_values": [baseline_rows[s]["after"][args.metric] for s in seeds],
            "mean_difference": sum(differences) / len(differences),
            "exact_sign_test_p": exact_sign_test(differences),
        })
    adjusted = holm([c["exact_sign_test_p"] for c in comparisons])
    for comparison, p_value in zip(comparisons, adjusted):
        comparison["holm_adjusted_p"] = p_value
    summary = {
        "schema_version": "1.0", "source": str(args.results), "metric": args.metric,
        "comparisons": comparisons,
        "note": "Seed-level exact sign test; small seed counts make p-values descriptive only.",
    }
    for comparison in comparisons:
        print(json.dumps({
            "pair": f"{comparison['arm']} vs {comparison['baseline']}",
            "mean_difference": round(comparison["mean_difference"], 5),
            "p": round(comparison["exact_sign_test_p"], 4),
            "holm_p": round(comparison["holm_adjusted_p"], 4),
        }))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
