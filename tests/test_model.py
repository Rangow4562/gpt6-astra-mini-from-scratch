"""Looped-model invariants: weight sharing, causality, config validation."""
import pytest
import torch

from astra_mini import AstraConfig, AstraMiniLM


def tiny_config(**overrides) -> AstraConfig:
    values = dict(
        vocab_size=260, dim=32, heads=4, mlp_hidden=64,
        first_blocks=1, stack_blocks=2, last_blocks=1, loops=2,
        depth_policy="fixed", inject_input=True, max_sequence_length=64,
    )
    values.update(overrides)
    return AstraConfig(**values)


def test_weight_sharing_loops_do_not_add_parameters():
    looped = AstraMiniLM(tiny_config(loops=2))
    plain = AstraMiniLM(tiny_config(loops=1))
    assert looped.parameter_counts()["total"] == plain.parameter_counts()["total"]
    assert looped.config.effective_depth == 1 + 2 * 2 + 1
    assert plain.config.effective_depth == 1 + 2 * 1 + 1
    counts = looped.parameter_counts()
    assert counts["deep_block_equivalent_parameters"] == counts["distinct_block_parameters"] * 2


def test_forward_shapes_and_tied_head():
    model = AstraMiniLM(tiny_config()).eval()
    inputs = torch.randint(4, 260, (2, 12))
    with torch.no_grad():
        logits, diagnostics = model(inputs)
    assert logits.shape == (2, 12, 260)
    assert model.output.weight is model.embedding.weight
    assert diagnostics["mean_depth"] == 2.0


def test_prefix_causality_through_the_loop():
    model = AstraMiniLM(tiny_config()).eval()
    torch.manual_seed(0)
    inputs = torch.randint(4, 260, (1, 20))
    with torch.no_grad():
        short_logits, _ = model(inputs[:, :10])
        long_logits, _ = model(inputs)
    assert torch.allclose(short_logits, long_logits[:, :10], atol=1e-4)


def test_active_loops_scales_compute_without_new_parameters():
    model = AstraMiniLM(tiny_config()).eval()
    inputs = torch.randint(4, 260, (1, 8))
    with torch.no_grad():
        one_loop, diagnostics_one = model(inputs, active_loops=1)
        two_loops, diagnostics_two = model(inputs, active_loops=2)
    assert one_loop.shape == two_loops.shape
    assert diagnostics_one["mean_depth"] == 1.0
    assert diagnostics_two["mean_depth"] == 2.0
    assert not torch.allclose(one_loop, two_loops)
    with pytest.raises(ValueError):
        model(inputs, active_loops=0)
    with pytest.raises(ValueError):
        model(inputs, active_loops=3)


def test_config_validation():
    with pytest.raises(ValueError):
        AstraMiniLM(tiny_config(dim=33, heads=4))
    with pytest.raises(ValueError):
        AstraMiniLM(tiny_config(loops=0))
    with pytest.raises(ValueError):
        AstraMiniLM(tiny_config(depth_policy="quantum"))


def test_forward_embeddings_shape_errors():
    model = AstraMiniLM(tiny_config())
    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 16))
    with pytest.raises(ValueError):
        model(torch.randint(0, 260, (1, 65)))
