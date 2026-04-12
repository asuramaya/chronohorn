"""Tokenizer pipeline CLI for chronohorn.

Commands:
    chronohorn tokenize difficulty  -- measure per-byte difficulty from a checkpoint
    chronohorn tokenize build       -- build a difficulty-aware vocabulary
    chronohorn tokenize retokenize  -- re-encode shards with a new tokenizer
    chronohorn tokenize report      -- measure tokenizer quality metrics
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chronohorn tokenize",
        description="Tokenizer pipeline: difficulty measurement, vocab building, retokenization.",
    )
    sub = parser.add_subparsers(dest="command")

    diff = sub.add_parser("difficulty", help="Measure per-byte prediction difficulty from a checkpoint")
    diff.add_argument("--checkpoint", required=True, help="Path to .checkpoint.pt")
    diff.add_argument("--result-json", required=True, help="Path to result .json (for config)")
    diff.add_argument("--data-root", required=True, help="Path to tokenized shard directory")
    diff.add_argument("--num-sequences", type=int, default=200, help="Sequences to evaluate (default 200)")
    diff.add_argument("--seq-len", type=int, default=512, help="Sequence length (default 512)")
    diff.add_argument("--output", required=True, help="Output .npy file for difficulty array")
    diff.add_argument("--device", default="cpu", help="Device (default cpu)")

    build = sub.add_parser("build", help="Build difficulty-aware vocabulary")
    build.add_argument("--difficulty", required=True, help="Path to difficulty .npy")
    build.add_argument("--text", required=True, help="Path to raw text file for bigram statistics")
    build.add_argument("--base-model", required=True, help="Path to large sentencepiece .model (candidate pool)")
    build.add_argument("--vocab-size", type=int, default=8192, help="Target vocab size (default 8192)")
    build.add_argument("--difficulty-weight", type=float, default=0.5)
    build.add_argument("--output", required=True, help="Output sentencepiece .model path")

    retok = sub.add_parser("retokenize", help="Re-encode shards with a new tokenizer")
    retok.add_argument("--model", required=True, help="Sentencepiece .model path")
    retok.add_argument("--input-dir", required=True, help="Directory of existing .bin shards")
    retok.add_argument("--input-tokenizer", required=True, help="Original sentencepiece .model")
    retok.add_argument("--output-dir", required=True, help="Output directory for new .bin shards")
    retok.add_argument("--vocab-size", type=int, default=None)

    report = sub.add_parser("report", help="Measure tokenizer quality")
    report.add_argument("--model", required=True, help="Sentencepiece .model path")
    report.add_argument("--text", required=True, help="Raw text file to measure")
    report.add_argument("--difficulty", default=None, help="Optional difficulty .npy")

    args = parser.parse_args(argv)

    if args.command == "difficulty":
        return _cmd_difficulty(args)
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "retokenize":
        return _cmd_retokenize(args)
    if args.command == "report":
        return _cmd_report(args)

    parser.print_help()
    return 0


def _cmd_difficulty(args: argparse.Namespace) -> int:
    import json

    import numpy as np

    print(f"loading checkpoint: {args.checkpoint}", flush=True)

    from decepticons.loader import load_checkpoint
    model = load_checkpoint(args.checkpoint, result_json=args.result_json, device=args.device)

    with open(args.result_json) as f:
        result_json = json.load(f)
    from chronohorn.train.token_shard_dataset import build_token_shard_dataset
    vocab_size = result_json.get("model", {}).get("vocab_size", 1024)
    dataset = build_token_shard_dataset(args.data_root, vocab_size=vocab_size)

    print(f"running {args.num_sequences} sequences of length {args.seq_len}", flush=True)

    from decepticons.tokenizer.difficulty import byte_difficulty_from_model
    difficulty = byte_difficulty_from_model(
        model, dataset,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        device=args.device,
    )

    np.save(args.output, difficulty)
    print(f"saved: {len(difficulty)} bytes, mean={difficulty.mean():.4f}, std={difficulty.std():.4f}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from pathlib import Path

    import numpy as np

    difficulty = np.load(args.difficulty)
    text = Path(args.text).read_bytes()
    print(f"difficulty: {len(difficulty)} bytes, text: {len(text)} bytes", flush=True)

    from decepticons.tokenizer.build_vocab import build_vocab
    vocab = build_vocab(
        difficulty, text,
        vocab_size=args.vocab_size,
        sentencepiece_model_path=args.base_model,
        difficulty_weight=args.difficulty_weight,
    )
    print(f"selected {len(vocab)} pieces", flush=True)

    _write_spm_from_vocab(vocab, args.output, args.vocab_size)
    print(f"wrote: {args.output}")
    return 0


def _write_spm_from_vocab(vocab: list[bytes], output_path: str, vocab_size: int) -> None:
    """Write a vocabulary as a sentencepiece model using the proto format."""
    from sentencepiece import sentencepiece_model_pb2 as model_pb2

    m = model_pb2.ModelProto()

    # Control tokens
    for i, piece in enumerate(["<unk>", "<s>", "</s>"]):
        sp = model_pb2.ModelProto.SentencePiece()
        sp.piece = piece
        sp.score = 0.0
        sp.type = [2, 3, 4][i]  # UNKNOWN, CONTROL, CONTROL
        m.pieces.append(sp)

    # Byte tokens
    for byte_val in range(256):
        sp = model_pb2.ModelProto.SentencePiece()
        sp.piece = f"<0x{byte_val:02X}>"
        sp.score = 0.0
        sp.type = 6  # BYTE
        m.pieces.append(sp)

    # Merge tokens
    added = 0
    for piece_bytes in vocab:
        if len(piece_bytes) <= 1:
            continue
        sp = model_pb2.ModelProto.SentencePiece()
        try:
            text = piece_bytes.decode("utf-8", errors="replace")
        except Exception:  # noqa: S112
            continue  # skip undecodable pieces
        sp.piece = text
        sp.score = -float(added)
        sp.type = 1  # NORMAL
        m.pieces.append(sp)
        added += 1
        if added + 259 >= vocab_size:
            break

    m.trainer_spec.model_type = 1  # UNIGRAM
    m.trainer_spec.vocab_size = len(m.pieces)
    m.trainer_spec.byte_fallback = True
    m.normalizer_spec.add_dummy_prefix = True
    m.normalizer_spec.escape_whitespaces = True

    with open(output_path, "wb") as f:
        f.write(m.SerializeToString())


def _cmd_retokenize(args: argparse.Namespace) -> int:
    import struct
    from pathlib import Path

    import numpy as np
    import sentencepiece as spm

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_sp = spm.SentencePieceProcessor(model_file=args.input_tokenizer)
    new_sp = spm.SentencePieceProcessor(model_file=args.model)
    new_vocab_size = args.vocab_size or new_sp.vocab_size()

    print(f"old vocab: {old_sp.vocab_size()}, new vocab: {new_vocab_size}", flush=True)

    MAGIC = 20240520
    VERSION = 1

    for shard_path in sorted(input_dir.glob("fineweb_*.bin")):
        print(f"  {shard_path.name}", end="", flush=True)

        raw = np.fromfile(shard_path, dtype=np.uint16)
        header = np.frombuffer(raw[:6].tobytes(), dtype=np.int32)
        if header[0] != MAGIC:
            print(" skip (bad magic)", flush=True)
            continue
        old_tokens = raw[6:]

        text = old_sp.decode(old_tokens.tolist())
        new_ids = new_sp.encode(text)
        new_tokens = np.array(new_ids, dtype=np.uint16)

        out_path = output_dir / shard_path.name
        with open(out_path, "wb") as f:
            f.write(struct.pack("<iii", MAGIC, VERSION, len(new_tokens)))
            new_tokens.tofile(f)

        ratio = len(text.encode("utf-8")) / max(len(new_tokens), 1)
        print(f" -> {len(new_tokens)} tokens ({ratio:.2f} B/tok)", flush=True)

    print("done", flush=True)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    import numpy as np
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=args.model)
    text = Path(args.text).read_text(encoding="utf-8", errors="replace")

    ids = sp.encode(text)
    n_tokens = len(ids)
    n_bytes = len(text.encode("utf-8"))
    bytes_per_token = n_bytes / max(n_tokens, 1)
    tokens_per_byte = n_tokens / max(n_bytes, 1)
    byte_tokens = sum(1 for tid in ids if sp.is_byte(tid))
    fallback_rate = byte_tokens / max(n_tokens, 1)

    print(f"vocab size:      {sp.vocab_size()}")
    print(f"tokens:          {n_tokens}")
    print(f"bytes:           {n_bytes}")
    print(f"bytes/token:     {bytes_per_token:.3f}")
    print(f"tokens/byte:     {tokens_per_byte:.4f}")
    print(f"byte fallback:   {byte_tokens} ({fallback_rate:.4%})")

    if args.difficulty:
        difficulty = np.load(args.difficulty)
        from decepticons.tokenizer.build_vocab import _bigram_difficulty, score_piece
        text_bytes = text.encode("utf-8")
        bd = _bigram_difficulty(difficulty[:len(text_bytes)], text_bytes)
        piece_scores = []
        for tid in ids:
            piece = sp.id_to_piece(tid)
            piece_bytes = piece.encode("utf-8").replace(b"\xe2\x96\x81", b" ")
            piece_scores.append(score_piece(piece_bytes, bd))
        scores = np.array(piece_scores)
        print(f"mean difficulty: {scores.mean():.4f}")
        print(f"std difficulty:  {scores.std():.4f}")

    return 0
