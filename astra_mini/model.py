"""AstraMiniLM: a looped-transformer language model built from scratch.

Layout (Mixture-of-Recursions-style sandwich, Nanbeige-style loop):

    bytes -> embedding -> first blocks -> anchor
          -> loops x [ re-inject(anchor, state) -> shared stack ]
          -> depth policy combines per-pass states
          -> last blocks -> RMSNorm -> tied LM head

The shared stack's weights are identical across passes: effective depth grows
with ``loops`` while the parameter count does not. Key/value projections are
computed per pass and are never shared across passes (sharing them halves the
cache but measurably hurts quality, per the Nanbeige report).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth import POLICY_CHOICES, POLICY_EXIT_GATE, POLICY_FIXED, POLICY_ROUTER, DepthRouter, ExitGate, combine_passes


@dataclass(frozen=True)
class AstraConfig:
    vocab_size: int = 260
    dim: int = 384
    heads: int = 6
    mlp_hidden: int = 1024
    first_blocks: int = 1
    stack_blocks: int = 6
    last_blocks: int = 1
    loops: int = 2
    depth_policy: str = POLICY_FIXED
    inject_input: bool = True
    exit_threshold: float = 0.5
    max_sequence_length: int = 192
    rope_base: float = 10000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def effective_depth(self) -> int:
        return self.first_blocks + self.stack_blocks * self.loops + self.last_blocks

    def validate(self) -> None:
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.loops < 1:
            raise ValueError("loops must be at least one")
        if self.depth_policy not in POLICY_CHOICES:
            raise ValueError(f"depth_policy must be one of {POLICY_CHOICES}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


def apply_rotary(q: torch.Tensor, k: torch.Tensor, *, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary position embeddings for [batch, time, heads, head_dim]."""
    if q.shape[-1] % 2:
        raise ValueError("rotary embeddings require an even head dimension")
    length, head_dim = q.shape[1], q.shape[-1]
    positions = torch.arange(offset, offset + length, device=q.device, dtype=torch.float32)
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos()[None, :, None, :].to(q.dtype)
    sin = angles.sin()[None, :, None, :].to(q.dtype)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class CausalAttention(nn.Module):
    """Causal multi-head attention with an optional per-pass KV cache.

    ``past_kv`` carries this block's cached keys/values from previous decoding
    steps. Each loop pass keeps its own cache; caches are never shared between
    passes because the inputs differ on every pass.
    """

    def __init__(self, config: AstraConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, length, dim = x.shape
        offset = 0 if past_kv is None else past_kv[0].shape[2]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        v = v.view(batch, length, self.heads, self.head_dim)
        q, k = apply_rotary(q, k, offset=offset)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) * self.head_dim**-0.5
        total = k.shape[2]
        # Query at absolute position offset+i may attend keys at absolute positions <= offset+i.
        query_positions = torch.arange(offset, offset + length, device=x.device)
        key_positions = torch.arange(total, device=x.device)
        causal = query_positions[:, None] >= key_positions[None, :]
        scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        output = torch.einsum("bhts,bhsd->bhtd", weights, v)
        return self.out(output.transpose(1, 2).reshape(batch, length, dim)), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    """Pre-norm block: attention residual, then SwiGLU residual."""

    def __init__(self, config: AstraConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.attention = CausalAttention(config)
        self.mlp_norm = RMSNorm(config.dim)
        self.mlp = SwiGLU(config.dim, config.mlp_hidden)

    def forward(
        self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, kv = self.attention(self.attention_norm(x), past_kv)
        x = x + attended
        return x + self.mlp(self.mlp_norm(x)), kv


def _make_blocks(config: AstraConfig, count: int) -> nn.ModuleList:
    return nn.ModuleList(TransformerBlock(config) for _ in range(count))


class AstraMiniLM(nn.Module):
    """The full looped language model."""

    def __init__(self, config: AstraConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.first = _make_blocks(config, config.first_blocks)
        self.stack = _make_blocks(config, config.stack_blocks)
        self.last = _make_blocks(config, config.last_blocks)
        self.reinject = nn.Linear(config.dim * 2, config.dim, bias=False) if config.inject_input else None
        if config.depth_policy == POLICY_EXIT_GATE:
            self.exit_gate = ExitGate(config.dim, config.loops)
        elif config.depth_policy == POLICY_ROUTER:
            self.router = DepthRouter(config.dim, config.loops)
        self.final_norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_stack(
        self,
        state: torch.Tensor,
        anchor: torch.Tensor,
        cache: dict | None,
        active_loops: int,
    ) -> list[torch.Tensor]:
        """Apply the shared stack ``active_loops`` times, collecting each pass's output."""
        pass_states = []
        for loop_index in range(active_loops):
            if self.reinject is not None:
                state = self.reinject(torch.cat([anchor, state], dim=-1))
            for block_index, block in enumerate(self.stack):
                past_kv = None
                if cache is not None:
                    past_kv = cache["stack"][loop_index][block_index]
                state, kv = block(state, past_kv)
                if cache is not None:
                    cache["stack"][loop_index][block_index] = kv
            pass_states.append(state)
        return pass_states

    def _apply_depth_policy(
        self, pass_states: list[torch.Tensor], anchor: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        config = self.config
        if config.depth_policy == POLICY_FIXED:
            return pass_states[-1], float(len(pass_states))
        if config.depth_policy == POLICY_EXIT_GATE:
            weights = self.exit_gate.weights(pass_states)
        else:
            weights = self.router.weights(anchor)[: len(pass_states)]
        return combine_passes(
            torch.stack(pass_states, dim=0), weights,
            training=self.training, threshold=config.exit_threshold,
        )

    def forward_full(
        self, input_ids: torch.Tensor, active_loops: int | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Full-sequence forward. ``active_loops`` scales test-time loop compute."""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input exceeds max_sequence_length")
        loops = self.config.loops if active_loops is None else active_loops
        if not 1 <= loops <= self.config.loops:
            raise ValueError(f"active_loops must be within [1, {self.config.loops}]")
        hidden = self.embedding(input_ids)
        for block in self.first:
            hidden, _ = block(hidden)
        anchor = hidden
        pass_states = self._run_stack(anchor, anchor, None, loops)
        combined, mean_depth = self._apply_depth_policy(pass_states, anchor)
        for block in self.last:
            combined, _ = block(combined)
        logits = self.output(self.final_norm(combined))
        return logits, {"mean_depth": mean_depth, "pass_states": pass_states}

    def init_cache(self) -> dict:
        """Fresh per-pass KV cache bundle for incremental decoding."""
        config = self.config
        return {
            "first": [None] * config.first_blocks,
            "stack": [[None] * config.stack_blocks for _ in range(config.loops)],
            "last": [None] * config.last_blocks,
            "length": 0,
        }

    def forward_step(
        self, input_ids: torch.Tensor, cache: dict, active_loops: int | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """One decoding step [batch, 1] with per-pass KV caches."""
        loops = self.config.loops if active_loops is None else active_loops
        if not 1 <= loops <= self.config.loops:
            raise ValueError(f"active_loops must be within [1, {self.config.loops}]")
        hidden = self.embedding(input_ids)
        for index, block in enumerate(self.first):
            hidden, kv = block(hidden, cache["first"][index])
            cache["first"][index] = kv
        anchor = hidden
        pass_states = self._run_stack(anchor, anchor, cache, loops)
        combined, mean_depth = self._apply_depth_policy(pass_states, anchor)
        for index, block in enumerate(self.last):
            combined, kv = block(combined, cache["last"][index])
            cache["last"][index] = kv
        cache["length"] += input_ids.shape[1]
        logits = self.output(self.final_norm(combined))
        return logits, {"mean_depth": mean_depth}

    def forward(self, input_ids: torch.Tensor, active_loops: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.forward_full(input_ids, active_loops)

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        distinct_blocks = sum(
            parameter.numel()
            for module in (*self.first, *self.stack, *self.last)
            for parameter in module.parameters()
        )
        applications = self.config.effective_depth
        deep_equivalent = distinct_blocks * self.config.loops
        return {
            "total": total,
            "distinct_block_parameters": distinct_blocks,
            "block_applications_per_token": applications,
            "deep_block_equivalent_parameters": deep_equivalent,
        }
