"""A tiny from-scratch looped-transformer ("recurrent depth") language model.

A teaching-scale homage to the architecture reported for GPT-6 Astra: a shared
transformer stack applied multiple times, sandwiched between distinct first and
last blocks, with per-token depth policies and optional anchor re-injection.
"""
from .model import AstraConfig, AstraMiniLM
from .tokenizer import ByteTokenizer

__all__ = ["AstraConfig", "AstraMiniLM", "ByteTokenizer"]
