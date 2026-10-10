"""Deterministic checkpoints, device resolution, and batched cached generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .evaluator import completion_is_parseable
from .model import AstraConfig, AstraMiniLM
from .tasks import CodingTask
from .tokenizer import ByteTokenizer


def resolve_device(specification: str) -> torch.device:
    if specification == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        name = specification
    return torch.device(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(model: AstraMiniLM, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    weights = path / "model.pt"
    torch.save(model.state_dict(), weights)
    (path / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2, sort_keys=True) + "\n")
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "weights_sha256": sha256_file(weights)}


def load_checkpoint(path: Path, device: torch.device) -> AstraMiniLM:
    config = AstraConfig(**json.loads((path / "config.json").read_text()))
    model = AstraMiniLM(config).to(device)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
    return model


@torch.no_grad()
def generate_group(
    model: AstraMiniLM,
    tokenizer: ByteTokenizer,
    task: CodingTask,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    sample: bool,
    seed: int,
    active_loops: int | None = None,
) -> list[dict[str, Any]]:
    """Batched autoregressive rollout with per-pass KV caches.

    Each new token runs the full loop (all passes) once; attention keys and
    values are cached per pass, so decoding cost does not grow with history.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prompt = tokenizer.encode(task.prompt, bos=True)
    if len(prompt) + max_new_tokens > model.config.max_sequence_length:
        raise ValueError("prompt plus generation allowance exceeds the context")
    device = next(model.parameters()).device
    rows = torch.tensor([prompt] * group_size, dtype=torch.long, device=device)
    completions: list[list[int]] = [[] for _ in range(group_size)]
    finished = [False] * group_size
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.eval()
    cache = model.init_cache()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, _ = model.forward_step(rows, cache, active_loops)
    for _ in range(max_new_tokens - 1):
        next_logits = logits[:, -1, :].float() / temperature
        next_logits[:, [tokenizer.pad_id, tokenizer.bos_id, tokenizer.sep_id]] = -torch.inf
        next_tokens = (
            torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1).squeeze(-1)
            if sample else next_logits.argmax(dim=-1)
        )
        appended = []
        for index, token in enumerate(next_tokens.tolist()):
            if finished[index]:
                token = tokenizer.eos_id
            else:
                completions[index].append(token)
                if token == tokenizer.eos_id or completion_is_parseable(task, tokenizer.decode(completions[index])):
                    finished[index] = True
            appended.append(token)
        if all(finished):
            break
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model.forward_step(
                torch.tensor(appended, dtype=torch.long, device=device)[:, None], cache, active_loops
            )
    results = []
    for tokens in completions:
        results.append({
            "completion": tokenizer.decode(tokens),
            "token_ids": tokens,
            "tokens": len(tokens),
            "hit_token_cap": len(tokens) == max_new_tokens and (not tokens or tokens[-1] != tokenizer.eos_id),
        })
    return results
