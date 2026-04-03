#!/usr/bin/env python3
"""Example: Ingest external training results into Chronohorn.

This shows how to feed results from ANY training framework (nanoGPT,
your custom transformer, etc.) into the Chronohorn tracking system.

Usage:
    python examples/ingest_results.py --result-dir /path/to/results/
    python examples/ingest_results.py --json /path/to/single_result.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from chronohorn.db import ChronohornDB


def create_example_result():
    """Create a minimal example result JSON for a transformer model."""
    return {
        "model": {
            "architecture": "transformer",
            "test_bpb": 1.45,
            "test_bits_per_token": 3.54,
            "params": 25_000_000,
            "n_layers": 6,
            "n_heads": 8,
            "n_embd": 512,
        },
        "config": {
            "train": {
                "steps": 50000,
                "seq_len": 1024,
                "batch_size": 32,
                "learning_rate": 3e-4,
                "weight_decay": 0.1,
            }
        },
        "training": {
            "performance": {
                "tokens_per_second": 150000,
                "elapsed_sec": 3600,
                "steps_completed": 50000,
            },
            "probes": [
                {"step": 1000, "bpb": 3.2},
                {"step": 5000, "bpb": 2.1},
                {"step": 10000, "bpb": 1.8},
                {"step": 25000, "bpb": 1.55},
                {"step": 50000, "bpb": 1.45},
            ]
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest results into Chronohorn")
    parser.add_argument("--result-dir", help="Directory of result JSON files")
    parser.add_argument("--json", help="Single result JSON file")
    parser.add_argument("--db", default="out/chronohorn.db", help="Database path")
    parser.add_argument("--example", action="store_true", help="Create and ingest an example transformer result")
    args = parser.parse_args()

    db = ChronohornDB(args.db)

    if args.example:
        result = create_example_result()
        db.record_result("example-transformer-50k", result)
        print(f"Ingested example transformer result")
        print(f"  bpb: {result['model']['test_bpb']}")
        print(f"  params: {result['model']['params']:,}")
        print(f"  probes: {len(result['training']['probes'])}")
    elif args.result_dir:
        count = db.rebuild_from_archive(args.result_dir)
        print(f"Ingested {count} results from {args.result_dir}")
    elif args.json:
        payload = json.loads(Path(args.json).read_text())
        name = Path(args.json).stem
        db.record_result(name, payload, json_archive=args.json)
        print(f"Ingested {name}: bpb={payload['model']['test_bpb']}")
    else:
        parser.print_help()
        db.close()
        return

    # Show summary
    summary = db.summary()
    print(f"\nDB: {summary['result_count']} results, best={summary['best_bpb']:.4f} bpb")
    print(f"Families: {summary['families']}")
    db.close()


if __name__ == "__main__":
    main()
