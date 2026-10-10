"""Procedural corpora: multi-step pretraining documents and frozen coding splits.

Pretraining uses compositional arithmetic-expression programs whose targets
require several chained operations — the regime where extra per-token compute
(looping) is supposed to help. Evaluation uses frozen synthetic coding tasks
with executable hidden tests.
"""
from __future__ import annotations

import itertools
import random
import string
from dataclasses import dataclass
from typing import Any

OPS = ("add", "sub", "mul", "neg", "abs")
PHRASES = {
    "add": ("Add {c}.", "Increase it by {c}."),
    "sub": ("Subtract {c}.", "Decrease it by {c}."),
    "mul": ("Multiply by {c}.", "Scale it by {c}."),
    "neg": ("Negate it.", "Flip its sign."),
    "abs": ("Take its absolute value.", "Make the result nonnegative."),
}


@dataclass(frozen=True)
class Family:
    name: str
    argument: str
    descriptions: tuple[str, ...]
    body: str
    cases: tuple[tuple[tuple[Any, ...], Any], ...]


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    family: str
    entry_point: str
    prompt: str
    reference_completion: str
    cases: tuple[tuple[tuple[Any, ...], Any], ...]

    @property
    def reference_source(self) -> str:
        return self.prompt + self.reference_completion


FAMILIES = (
    Family("increment", "x", ("Return x plus one.", "Increase x by one."), "\n    return x + 1\n",
           (((-3,), -2), ((0,), 1), ((7,), 8))),
    Family("double", "x", ("Return two times x.", "Double the input."), "\n    return x * 2\n",
           (((-4,), -8), ((0,), 0), ((6,), 12))),
    Family("square", "x", ("Return x squared.", "Multiply x by itself."), "\n    return x * x\n",
           (((-4,), 16), ((0,), 0), ((5,), 25))),
    Family("absolute", "x", ("Return the absolute value of x.", "Make a negative x positive."),
           "\n    return -x if x < 0 else x\n", (((-7,), 7), ((0,), 0), ((9,), 9))),
    Family("nonnegative", "x", ("Clamp x to at least zero.", "Return zero when x is negative."),
           "\n    return x if x > 0 else 0\n", (((-5,), 0), ((0,), 0), ((8,), 8))),
    Family("even", "x", ("Return whether x is even.", "Check divisibility by two."),
           "\n    return x % 2 == 0\n", (((-3,), False), ((0,), True), ((8,), True))),
    Family("reverse", "text", ("Return text in reverse order.", "Reverse the string."),
           "\n    return text[::-1]\n", ((("",), ""), (("abc",), "cba"), (("level",), "level"))),
    Family("list_sum", "values", ("Return the sum of values.", "Add every number in the list."),
           "\n    return sum(values)\n", ((([],), 0), (([1, 2, 3],), 6), (([-2, 5],), 3))),
)
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}
FAMILY_ORDINAL = {family.name: index for index, family in enumerate(FAMILIES)}

SPLIT_SEEDS = {"dev": 101, "final": 211, "rl": 307, "confirm": 409}


def composition_structures() -> list[tuple[str, ...]]:
    """All three-step compositions mixing at least two primitives (120 total)."""
    values = [
        structure
        for structure in itertools.product(OPS, repeat=3)
        if len(set(structure)) >= 2
    ]
    random.Random(1701).shuffle(values)
    return values


def composition_text(structure: tuple[str, ...], *, index: int, seed: int) -> str:
    """Render one compositional program; deterministic from (structure, index, seed)."""
    key = sum((position + 1) * (OPS.index(operation) + 1) * 101 for position, operation in enumerate(structure))
    rng = random.Random(seed * 1_000_003 + index * 9176 + key)
    constants = [rng.choice((2, 3, 4, 5)) for _ in structure]
    instructions = []
    expression = "x"
    for operation, constant in zip(structure, constants):
        instructions.append(rng.choice(PHRASES[operation]).format(c=constant))
        if operation == "add":
            expression = f"({expression} + {constant})"
        elif operation == "sub":
            expression = f"({expression} - {constant})"
        elif operation == "mul":
            expression = f"({expression} * {constant})"
        elif operation == "neg":
            expression = f"(-{expression})"
        else:
            expression = f"abs({expression})"
    return f"Task: start with x. {' '.join(instructions)}\nPython: {expression}\n"


def pretraining_text(index: int, *, seed: int = 42, loops: int | None = None) -> str:
    """One deterministic pretraining document; `loops` varies step count Geiping-style."""
    rng = random.Random(seed * 10_000_679 + index * 13)
    pool = composition_structures()
    if loops is not None:
        steps = 2 if (index + loops) % 3 == 0 else 3
    else:
        steps = 3
    while True:
        structure = pool[rng.randrange(len(pool))]
        trimmed = structure[:steps]
        if len(set(trimmed)) >= 2:
            return composition_text(trimmed, index=index, seed=seed)


def _task_seed(seed: int, family_index: int, index: int) -> int:
    return (seed + 1) * 7_717_817 + index * 65_537 + family_index * 3_511


def _random_name(rng: random.Random, prefix: str) -> str:
    return prefix + "_" + "".join(rng.choice(string.ascii_lowercase) for _ in range(7))


def make_task(family: Family, *, split: str, index: int, seed: int) -> CodingTask:
    rng = random.Random(_task_seed(seed, FAMILY_ORDINAL[family.name], index))
    entry = _random_name(rng, family.name)
    description = family.descriptions[rng.randrange(len(family.descriptions))]
    prompt = f"# Complete this Python function.\n# {description}\ndef {entry}({family.argument}):"
    return CodingTask(
        task_id=f"{split}-{family.name}-{index:03d}",
        family=family.name,
        entry_point=entry,
        prompt=prompt,
        reference_completion=family.body,
        cases=family.cases,
    )


def frozen_tasks(split: str, *, per_family: int = 4) -> list[CodingTask]:
    if split not in SPLIT_SEEDS:
        raise ValueError(f"unknown split: {split}")
    tasks: list[CodingTask] = []
    for family_index, family in enumerate(FAMILIES):
        for index in range(per_family):
            tasks.append(
                make_task(family, split=split, index=index, seed=SPLIT_SEEDS[split] + family_index * 101)
            )
    return tasks
