"""Runtime invariants: cached-vs-uncached equivalence, checkpoints, generation."""
import pytest
import torch

from astra_mini import AstraConfig, AstraMiniLM, ByteTokenizer
from astra_mini.runtime import generate_group, load_checkpoint, save_checkpoint
from astra_mini.tasks import frozen_tasks


def tiny_config(**overrides) -> AstraConfig:
    values = dict(
        vocab_size=260, dim=32, heads=4, mlp_hidden=64,
        first_blocks=1, stack_blocks=2, last_blocks=1, loops=2,
        depth_policy="fixed", inject_input=True,
    )
    values.update(overrides)
    return AstraConfig(**values)


def tiny_model(**overrides) -> AstraMiniLM:
    torch.manual_seed(7)
    return AstraMiniLM(tiny_config(**overrides)).eval()


def test_cached_generation_matches_full_forward_exactly():
    model = tiny_model()
    torch.manual_seed(3)
    ids = torch.randint(4, 260, (1, 6))
    with torch.no_grad():
        full_logits, _ = model(ids)
    cache = model.init_cache()
    cached_logits = []
    with torch.no_grad():
        for position in range(ids.shape[1]):
            step_logits, _ = model.forward_step(ids[:, position: position + 1], cache)
            cached_logits.append(step_logits[:, -1, :])
    cached = torch.stack(cached_logits, dim=1)
    assert torch.allclose(full_logits, cached, atol=1e-5)


def test_cache_is_per_pass_not_shared():
    model = tiny_model()
    cache = model.init_cache()
    assert len(cache["stack"]) == 2
    assert len(cache["stack"][0]) == 2
    ids = torch.randint(4, 260, (1, 4))
    with torch.no_grad():
        model.forward_step(ids, cache)
    first_pass_key = cache["stack"][0][0][0]
    second_pass_key = cache["stack"][1][0][0]
    assert first_pass_key.shape == second_pass_key.shape
    assert not torch.allclose(first_pass_key, second_pass_key)


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model()
    inputs = torch.randint(4, 260, (1, 8))
    with torch.no_grad():
        before, _ = model(inputs)
    first = save_checkpoint(model, tmp_path / "checkpoint", {"stage": "test"})
    assert len(first["weights_sha256"]) == 64
    with pytest.raises(FileExistsError):
        save_checkpoint(model, tmp_path / "checkpoint", {"stage": "test"})
    loaded = load_checkpoint(tmp_path / "checkpoint", torch.device("cpu")).eval()
    with torch.no_grad():
        after, _ = loaded(inputs)
    assert torch.allclose(before, after)


def test_generation_masks_specials_and_is_deterministic():
    model = tiny_model(mlp_hidden=64)
    tokenizer = ByteTokenizer()
    task = frozen_tasks("rl", per_family=1)[0]
    group = generate_group(
        model, tokenizer, task, group_size=3, max_new_tokens=10,
        temperature=1.0, sample=True, seed=99,
    )
    assert len(group) == 3
    for row in group:
        assert len(row["token_ids"]) == row["tokens"]
        assert tokenizer.decode(row["token_ids"]) == row["completion"]
        for special in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.sep_id):
            assert special not in row["token_ids"]
    repeat = generate_group(
        model, tokenizer, task, group_size=3, max_new_tokens=10,
        temperature=1.0, sample=True, seed=99,
    )
    assert [row["token_ids"] for row in group] == [row["token_ids"] for row in repeat]


def test_generation_validates_arguments():
    model = tiny_model()
    tokenizer = ByteTokenizer()
    task = frozen_tasks("rl", per_family=1)[0]
    with pytest.raises(ValueError):
        generate_group(model, tokenizer, task, group_size=1, max_new_tokens=4,
                       temperature=0.0, sample=False, seed=1)
    with pytest.raises(ValueError):
        generate_group(model, tokenizer, task, group_size=1,
                       max_new_tokens=model.config.max_sequence_length,
                       temperature=1.0, sample=False, seed=1)
    with pytest.raises(ValueError):
        generate_group(model, tokenizer, task, group_size=1, max_new_tokens=4,
                       temperature=1.0, sample=False, seed=1, active_loops=9)
