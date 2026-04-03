"""Transformer family package for Chronohorn.

Generic adapter for transformer-based language models (GPT, LLaMA, etc.).
Not tied to any specific implementation -- just defines how transformer
results are detected, validated, and summarized.
"""

__all__ = [
    "TRANSFORMER_TRAINING_ADAPTER",
    "TransformerTrainingAdapter",
]


def __getattr__(name):
    if name in ("TRANSFORMER_TRAINING_ADAPTER", "TransformerTrainingAdapter"):
        from .adapter import TRANSFORMER_TRAINING_ADAPTER, TransformerTrainingAdapter

        return {
            "TRANSFORMER_TRAINING_ADAPTER": TRANSFORMER_TRAINING_ADAPTER,
            "TransformerTrainingAdapter": TransformerTrainingAdapter,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
