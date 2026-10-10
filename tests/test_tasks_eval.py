"""Tokenizer, corpus, and verifier invariants."""
import pytest

from astra_mini.evaluator import evaluate_source
from astra_mini.tasks import composition_structures, composition_text, frozen_tasks, pretraining_text
from astra_mini.tokenizer import ByteTokenizer

SPLITS = ("dev", "rl", "confirm")


def test_tokenizer_round_trip_including_unicode():
    tokenizer = ByteTokenizer()
    text = "héllo αβλ 😀\n    return x + 1\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_reference_completions_pass_on_every_split():
    for split in SPLITS:
        for task in frozen_tasks(split):
            evaluation = evaluate_source(task, task.reference_source)
            assert evaluation.passed, (task.task_id, evaluation.message)


def test_stub_completion_fails_but_stays_valid():
    task = frozen_tasks("dev")[0]
    evaluation = evaluate_source(task, task.prompt + "\n    pass\n")
    assert not evaluation.passed
    assert evaluation.status == "failed"


def test_verifier_rejects_disallowed_sources():
    task = frozen_tasks("dev")[0]
    sources = [
        task.prompt + "\n    import os\n",
        task.prompt + "\n    return __name__\n",
        task.prompt + "\n    for _ in range(3):\n        pass\n",
        task.prompt + "\n    return x.upper()\n",
    ]
    for source in sources:
        evaluation = evaluate_source(task, source)
        assert evaluation.status == "invalid", (task.task_id, source, evaluation.message)


def test_frozen_splits_use_disjoint_entry_points():
    by_split = {split: {task.entry_point for task in frozen_tasks(split)} for split in SPLITS}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            assert by_split[left].isdisjoint(by_split[right])


def test_composition_corpus_is_deterministic_and_multi_step():
    structures = composition_structures()
    assert len(structures) == 120
    assert all(len(set(structure)) >= 2 for structure in structures)
    first = composition_text(structures[0], index=5, seed=42)
    second = composition_text(structures[0], index=5, seed=42)
    assert first == second
    assert "Task: start with x." in first


def test_pretraining_text_varies_steps_when_loops_given():
    base = pretraining_text(3, seed=42)
    assert base == pretraining_text(3, seed=42)
    assert base != pretraining_text(4, seed=42)
    # With loop randomization, some documents use two operations, some three.
    texts = {pretraining_text(index, seed=42, loops=2) for index in range(30)}
    assert any(text.count("Task: start with x.") == 1 for text in texts)
    lengths = {len(text.splitlines()[0]) for text in texts}
    assert len(lengths) >= 2


def test_unknown_split_rejected():
    with pytest.raises(ValueError):
        frozen_tasks("train")
