"""Shared constants for the Transformer family."""
from __future__ import annotations

TRANSFORMER_FAMILY_ID = "transformer"

# Known architecture strings that should route to this family.
KNOWN_ARCHITECTURES = (
    "transformer", "gpt", "gpt2", "gpt3", "gpt4",
    "llama", "llama2", "llama3", "mistral", "mixtral",
    "nanoGPT", "minGPT", "decoder_only", "causal_lm",
    "bert",  # detected but flagged as potentially illegal (bidirectional)
)

# Config fields commonly found in transformer results.
SUMMARY_KEYS = (
    "n_layers", "n_heads", "n_embd", "d_model", "d_ff",
    "vocab_size", "context_length", "max_seq_len", "block_size",
    "dropout", "attention_type", "activation",
)

# Alternative key names that map to the canonical keys above.
KEY_ALIASES = {
    "num_layers": "n_layers",
    "n_layer": "n_layers",          # nanoGPT singular
    "num_heads": "n_heads",
    "n_head": "n_heads",            # nanoGPT singular
    "hidden_size": "n_embd",
    "embed_dim": "n_embd",
    "intermediate_size": "d_ff",
    "ffn_dim": "d_ff",
    "max_position_embeddings": "max_seq_len",
    "sequence_length": "max_seq_len",
    "context_length": "max_seq_len",
}
