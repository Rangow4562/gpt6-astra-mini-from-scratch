"""Depth-policy invariants: hard selection, soft mixtures, expected depth."""
import torch

from astra_mini.depth import (
    ExitGate,
    DepthRouter,
    combine_passes,
    expected_depth,
    hard_select,
)


def make_states(loops: int = 3, batch: int = 2, time: int = 5, dim: int = 8):
    torch.manual_seed(0)
    return [torch.randn(batch, time, dim) for _ in range(loops)]


def test_hard_select_picks_first_pass_reaching_threshold():
    states = make_states()
    # One-hot weights on pass 2 (index 1).
    weights = torch.zeros(3, 2, 5)
    weights[1] = 1.0
    selected = hard_select(torch.stack(states), weights, threshold=0.5)
    assert torch.allclose(selected, states[1])


def test_hard_select_cumulative_threshold():
    states = make_states()
    weights = torch.zeros(3, 2, 5)
    weights[0] = 0.4
    weights[1] = 0.35
    weights[2] = 0.25
    # Cumulative [0.40, 0.75, 1.00] -> pass 2 (index 1) is the first past 0.5.
    selected = hard_select(torch.stack(states), weights, threshold=0.5)
    assert torch.allclose(selected, states[1])


def test_soft_mixture_is_the_weighted_sum():
    states = make_states()
    weights = torch.full((3, 2, 5), 1.0 / 3)
    output, depth = combine_passes(torch.stack(states), weights, training=True, threshold=0.5)
    manual = states[0] / 3 + states[1] / 3 + states[2] / 3
    assert torch.allclose(output, manual, atol=1e-6)
    assert abs(depth - 2.0) < 1e-5


def test_expected_depth_readout():
    weights = torch.zeros(3, 2, 5)
    weights[2] = 1.0
    assert expected_depth(weights) == 3.0
    weights = torch.full((3, 1, 1), 1.0 / 3)
    assert abs(expected_depth(weights) - 2.0) < 1e-5


def test_exit_gate_weights_are_a_distribution():
    gate = ExitGate(dim=8, loops=3)
    states = make_states()
    weights = gate.weights(states)
    assert weights.shape == (3, 2, 5)
    assert torch.allclose(weights.sum(dim=0), torch.ones(2, 5), atol=1e-5)


def test_router_weights_read_the_anchor():
    router = DepthRouter(dim=8, loops=3)
    torch.manual_seed(1)
    anchor = torch.randn(2, 5, 8)
    weights = router.weights(anchor)
    assert weights.shape == (3, 2, 5)
    assert torch.allclose(weights.sum(dim=0), torch.ones(2, 5), atol=1e-5)
    # The same anchor must always produce the same routing decision.
    again = router.weights(anchor)
    assert torch.allclose(weights, again)


def test_combine_passes_eval_mode_hard_selects():
    states = make_states()
    weights = torch.zeros(3, 2, 5)
    weights[0] = 0.2
    weights[1] = 0.29
    weights[2] = 0.51
    # Cumulative [0.20, 0.49, 1.00] -> first past 0.5 is the last pass.
    output, _ = combine_passes(torch.stack(states), weights, training=False, threshold=0.5)
    assert torch.allclose(output, states[2])
