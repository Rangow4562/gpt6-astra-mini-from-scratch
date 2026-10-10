"""Fail-closed AST gate plus executable unit-test verifier for the coding tasks."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any

from .tasks import CodingTask

MAX_SOURCE_BYTES = 2048
BANNED_NODES = (
    ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.With,
    ast.AsyncWith, ast.Lambda, ast.Global, ast.Nonlocal, ast.While, ast.For,
    ast.AsyncFor, ast.Try, ast.Raise, ast.Delete,
)
CALLABLE_BUILTINS = ("sum", "len", "abs", "min", "max", "bool", "int", "str")
_BUILTIN_TABLE = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
SAFE_BUILTINS = {name: _BUILTIN_TABLE[name] for name in CALLABLE_BUILTINS}


@dataclass(frozen=True)
class Evaluation:
    status: str
    passed: bool
    tests_passed: int
    tests_total: int
    pass_fraction: float
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_source(source: str, entry_point: str) -> ast.Module:
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds the byte limit")
    tree = ast.parse(source)
    definitions = []
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) \
                and isinstance(statement.value.value, str):
            continue
        if isinstance(statement, ast.FunctionDef):
            definitions.append(statement)
            continue
        raise ValueError("top level may contain only the requested function")
    if len(definitions) != 1 or definitions[0].name != entry_point:
        raise ValueError("exactly one function with the requested name is required")
    for node in ast.walk(tree):
        if isinstance(node, BANNED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are disallowed")
        if isinstance(node, ast.Attribute):
            raise ValueError("attribute access is disallowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in CALLABLE_BUILTINS:
                raise ValueError("call is not on the allowlist")
            if node.func.id == entry_point:
                raise ValueError("recursion is disallowed")
    return tree


def evaluate_source(task: CodingTask, source: str) -> Evaluation:
    total = len(task.cases)
    try:
        tree = validate_source(source, task.entry_point)
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        exec(compile(tree, "candidate.py", "exec"), namespace)
        candidate = namespace[task.entry_point]
    except (SyntaxError, ValueError, TypeError, KeyError) as error:
        return Evaluation("invalid", False, 0, total, 0.0, str(error))
    passed = 0
    for arguments, expected in task.cases:
        try:
            actual = candidate(*arguments)
            passed += int(type(actual) is type(expected) and actual == expected)
        except Exception:
            pass
    return Evaluation(
        "passed" if passed == total else "failed",
        passed == total,
        passed,
        total,
        passed / total,
    )


def completion_is_parseable(task: CodingTask, completion: str) -> bool:
    try:
        validate_source(task.prompt + completion, task.entry_point)
    except (SyntaxError, ValueError):
        return False
    return completion.endswith("\n")
