#!/usr/bin/env python3
"""Recursive self-improvement: the looped model trains on its own verified programs."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra_mini import AstraMiniLM, ByteTokenizer
from astra_mini.evaluator import evaluate_source
from astra_mini.runtime import generate_group, load_checkpoint, resolve_device, save_checkpoint
from astra_mini.tasks import CodingTask, frozen_tasks

DISTILL_SEQUENCE_LENGTH = 192


@dataclass
class VerifiedMemory:
    pairs: list[tuple[CodingTask, str]]
    sources: set[tuple[str, str]]

    def __init__(self) -> None:
        self.pairs = []
        self.sources = set()

    def add(self, task: CodingTask, completion: str) -> bool:
        key = (task.task_id, completion)
        if key in self.sources:
            return False
        self.sources.add(key)
        self.pairs.append((task, completion))
        return True

    def __len__(self) -> int:
        return len(self.pairs)


def distill_batch(tokenizer: ByteTokenizer, pairs: list[tuple[CodingTask, str]], *,
                  step: int, batch_size: int, device: torch.device):
    rows = []
    for offset in range(batch_size):
        task, completion = pairs[(step * batch_size + offset) % len(pairs)]
        ids = tokenizer.encode(task.prompt, bos=True) + tokenizer.encode(completion, eos=True)
        if len(ids) > DISTILL_SEQUENCE_LENGTH + 1:
            raise ValueError(f"distilled example exceeds context: {len(ids)}")
        ids += [tokenizer.pad_id] * (DISTILL_SEQUENCE_LENGTH + 1 - len(ids))
        rows.append(ids)
    values = torch.tensor(rows, dtype=torch.long, device=device)
    inputs, labels = values[:, :-1], values[:, 1:]
    labels = labels.masked_fill(labels == tokenizer.pad_id, -100)
    return inputs, labels, int((labels != -100).sum().item())


def propose(model: AstraMiniLM, tokenizer: ByteTokenizer, tasks: list[CodingTask], *,
            proposals_per_task: int, max_new_tokens: int, temperature: float,
            seed: int, active_loops: int | None) -> tuple[list[tuple[CodingTask, str]], list[dict[str, Any]]]:
    accepted: list[tuple[CodingTask, str]] = []
    details: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        generations = generate_group(
            model, tokenizer, task, group_size=proposals_per_task,
            max_new_tokens=max_new_tokens, temperature=temperature,
            sample=True, seed=seed + task_index * 104_729, active_loops=active_loops,
        )
        passed = 0
        for generation in generations:
            evaluation = evaluate_source(task, task.prompt + generation["completion"])
            if evaluation.passed:
                accepted.append((task, generation["completion"]))
                passed += 1
        details.append({
            "task_id": task.task_id, "family": task.family,
            "proposals": proposals_per_task, "passed": passed,
        })
    return accepted, details


def evaluate_pool(model: AstraMiniLM, tokenizer: ByteTokenizer, tasks: list[CodingTask], *,
                  seed: int, active_loops: int | None) -> dict[str, Any]:
    episodes = []
    for index, task in enumerate(tasks):
        generated = generate_group(
            model, tokenizer, task, group_size=1, max_new_tokens=64,
            temperature=1.0, sample=False, seed=seed + index, active_loops=active_loops,
        )[0]
        evaluation = evaluate_source(task, task.prompt + generated["completion"])
        episodes.append({
            "task_id": task.task_id, "family": task.family,
            "completion": generated["completion"], "passed": evaluation.passed,
            "status": evaluation.status, "tests_passed": evaluation.tests_passed,
        })
    solved = sum(episode["passed"] for episode in episodes)
    return {"tasks": len(episodes), "solved": solved, "pass_rate": solved / len(episodes), "episodes": episodes}


def fine_tune(model: AstraMiniLM, tokenizer: ByteTokenizer, memory: VerifiedMemory, *,
              steps: int, batch_size: int, learning_rate: float, device: torch.device) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    curve = []
    for step in range(steps):
        inputs, labels, tokens = distill_batch(tokenizer, memory.pairs, step=step,
                                               batch_size=batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                               labels.reshape(-1), ignore_index=-100)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite self-improvement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        curve.append({"step": step + 1, "loss": float(loss.detach()), "tokens": tokens})
    return {"steps": steps, "curve": curve}


def run_self_improvement(model: AstraMiniLM, tokenizer: ByteTokenizer, config: dict,
                         output: Path, device: torch.device) -> dict[str, Any]:
    output.mkdir(parents=True)
    memory = VerifiedMemory()
    cycles: list[dict[str, Any]] = []
    initial = save_checkpoint(model, output / "checkpoint-cycle-00", {
        "stage": "self_improvement_start", "seed": config["seed"],
    })
    started = time.perf_counter()
    for cycle in range(1, config["cycles"] + 1):
        cycle_started = time.perf_counter()
        rung = (cycle - 1) if config["expand_on_master"] else (config["cycles"] - 1)
        rung = min(rung, len(config["families_ladder"]) - 1)
        ladder = set(config["families_ladder"][: rung + 1])
        train_pool = [t for t in frozen_tasks("rl", per_family=config["per_family"]) if t.family in ladder]
        eval_pool = [t for t in frozen_tasks("dev", per_family=config["eval_per_family"]) if t.family in ladder]
        before = evaluate_pool(model, tokenizer, eval_pool, seed=config["seed"] + cycle * 7919,
                               active_loops=config["active_loops"])
        accepted, proposal_details = propose(
            model, tokenizer, train_pool, proposals_per_task=config["proposals_per_task"],
            max_new_tokens=config["max_new_tokens"], temperature=config["temperature"],
            seed=config["seed"] * 31 + cycle * 1009, active_loops=config["active_loops"],
        )
        added = sum(1 for task, completion in accepted if memory.add(task, completion))
        training = None
        if added > 0:
            training = fine_tune(model, tokenizer, memory, steps=config["fine_tune_steps"],
                                 batch_size=config["fine_tune_batch_size"],
                                 learning_rate=config["learning_rate"], device=device)
        after = evaluate_pool(model, tokenizer, eval_pool, seed=config["seed"] + cycle * 7919,
                              active_loops=config["active_loops"])
        checkpoint = save_checkpoint(model, output / f"checkpoint-cycle-{cycle:02d}", {
            "stage": "self_improvement", "cycle": cycle, "verified_pairs": len(memory),
        })
        cycles.append({
            "cycle": cycle, "families": sorted(ladder),
            "verified_pairs_total": len(memory), "verified_pairs_added": added,
            "proposal_details": proposal_details,
            "pass_rate_before": before["pass_rate"], "pass_rate_after": after["pass_rate"],
            "fine_tune": training, "evaluation": after, "checkpoint": checkpoint,
            "seconds": round(time.perf_counter() - cycle_started, 3),
        })
        print(json.dumps({key: cycles[-1][key] for key in (
            "cycle", "families", "verified_pairs_total", "verified_pairs_added",
            "pass_rate_before", "pass_rate_after", "seconds")}, sort_keys=True), flush=True)
        if after["pass_rate"] >= config["target_pass_rate"]:
            break
    receipt = {
        "schema_version": "1.0", "status": "complete", "stage": "recursive_self_improvement",
        "config": config, "initial_checkpoint": initial,
        "verified_memory_final_size": len(memory), "cycles": cycles,
        "elapsed_seconds": round(time.perf_counter() - started, 3), "device": device.type,
    }
    (output / "training-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--proposals-per-task", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--fine-tune-steps", type=int, default=24)
    parser.add_argument("--fine-tune-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-pass-rate", type=float, default=1.0)
    parser.add_argument("--families-ladder", default="increment")
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--eval-per-family", type=int, default=2)
    parser.add_argument("--expand-on-master", action="store_true")
    parser.add_argument("--active-loops", type=int, default=None)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ladder = tuple(value.strip() for value in args.families_ladder.split(",") if value.strip())
    if not ladder:
        raise ValueError("families ladder must not be empty")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    tokenizer = ByteTokenizer()
    model = load_checkpoint(args.initial_checkpoint, device)
    config = {
        "cycles": args.cycles, "proposals_per_task": args.proposals_per_task,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "fine_tune_steps": args.fine_tune_steps, "fine_tune_batch_size": args.fine_tune_batch_size,
        "learning_rate": args.learning_rate, "target_pass_rate": args.target_pass_rate,
        "families_ladder": list(ladder), "per_family": args.per_family,
        "eval_per_family": args.eval_per_family, "expand_on_master": args.expand_on_master,
        "active_loops": args.active_loops, "seed": args.seed,
    }
    receipt = run_self_improvement(model, tokenizer, config, args.output, device)
    final = receipt["cycles"][-1]
    print(json.dumps({
        "status": receipt["status"], "cycles": len(receipt["cycles"]),
        "verified_memory": receipt["verified_memory_final_size"],
        "final_pass_rate": final["pass_rate_after"],
        "elapsed_seconds": receipt["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
