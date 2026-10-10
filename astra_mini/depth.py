"""Per-token depth policies for the looped stack.

Three ways to decide which loop supplies a position's output, mirroring the
design space in the reference article:

- ``fixed``: every position takes the final pass (Nanbeige-style).
- ``exit_gate``: a learned gate scores each pass per position; the smallest
  pass whose cumulative probability reaches the threshold supplies the output
  (Ouro's exit gate, Universal-Transformer-style halting).
- ``router``: a single token-choice router on the anchor assigns each position
  a pass count up front (Mixture-of-Recursions, token-choice variant).

Training uses the soft mixture over pass outputs, which is differentiable
everywhere. Inference makes the hard selection. All passes are always computed
— the same documented caveat as Ouro's released implementation; actual
compute savings need MoR-style KV sharing and are out of scope here.
"""
from __future__ import annotations

import torch
from torch import nn

POLICY_FIXED = "fixed"
POLICY_EXIT_GATE = "exit_gate"
POLICY_ROUTER = "router"
POLICY_CHOICES = (POLICY_FIXED, POLICY_EXIT_GATE, POLICY_ROUTER)


def hard_select(stacked: torch.Tensor, weights: torch.Tensor, threshold: float) -> torch.Tensor:
    """Pick, per position, the first pass whose cumulative weight reaches the threshold."""
    cumulative = weights.cumsum(dim=0)
    reached = cumulative >= threshold
    choice = reached.float().argmax(dim=0)
    choice = torch.where(reached.any(dim=0), choice, torch.full_like(choice, stacked.shape[0] - 1))
    index = choice.unsqueeze(0).unsqueeze(-1).expand(1, *choice.shape, stacked.shape[-1])
    return stacked.gather(0, index).squeeze(0)


def expected_depth(weights: torch.Tensor) -> float:
    """Diagnostic: mean selected (train: expected) loop count over all positions."""
    loop_numbers = torch.arange(1, weights.shape[0] + 1, device=weights.device, dtype=weights.dtype)
    return float((weights * loop_numbers[:, None, None]).sum(dim=0).mean().item())


class ExitGate(nn.Module):
    """Ouro-style gate: one logit per pass, read from that pass's own state."""

    def __init__(self, dim: int, loops: int):
        super().__init__()
        self.loops = loops
        self.head = nn.Linear(dim, 1, bias=True)

    def weights(self, pass_states: list[torch.Tensor]) -> torch.Tensor:
        logits = torch.stack([self.head(state).squeeze(-1) for state in pass_states], dim=0)
        return torch.softmax(logits.float(), dim=0).to(pass_states[0].dtype)


class DepthRouter(nn.Module):
    """MoR-style token-choice router: one decision per position, from the anchor."""

    def __init__(self, dim: int, loops: int):
        super().__init__()
        self.loops = loops
        self.head = nn.Linear(dim, loops, bias=True)

    def weights(self, anchor: torch.Tensor) -> torch.Tensor:
        logits = self.head(anchor.float()).permute(2, 0, 1)  # [loops, batch, time]
        return torch.softmax(logits.float(), dim=0).to(anchor.dtype)


def combine_passes(
    stacked: torch.Tensor,
    weights: torch.Tensor,
    *,
    training: bool,
    threshold: float,
) -> tuple[torch.Tensor, float]:
    """Merge stacked pass states [loops, batch, time, dim] under the policy."""
    if training:
        output = torch.einsum("lbtd,lbt->btd", stacked, weights)
    else:
        output = hard_select(stacked, weights, threshold)
    return output, expected_depth(weights)
