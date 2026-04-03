#!/usr/bin/env python3
"""Backfill chronohorn.db from result JSONs with latest detection logic.

Run this after updating family detection, illegal detection, or forecast quality gates
to re-derive all stored metadata from the source result JSONs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from chronohorn.db import ChronohornDB


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild chronohorn.db from result JSONs")
    parser.add_argument("--result-dir", default="out/results", help="Result JSON directory")
    parser.add_argument("--db", default=None, help="DB path (default: <result-dir>/../chronohorn.db)")
    parser.add_argument("--manifests-dir", default="manifests", help="Manifests directory")
    parser.add_argument("--force", action="store_true", help="Delete existing DB and rebuild")
    args = parser.parse_args()

    db_path = args.db or str(Path(args.result_dir).parent / "chronohorn.db")

    if args.force and Path(db_path).exists():
        Path(db_path).unlink()
        print(f"Deleted {db_path}")

    db = ChronohornDB(db_path)

    # Rebuild results
    count = db.rebuild_from_archive(args.result_dir)
    print(f"Ingested {count} results from {args.result_dir}")

    # Ingest manifests
    mdir = Path(args.manifests_dir)
    if mdir.exists():
        manifest_count = 0
        for p in sorted(mdir.glob("frontier_*.jsonl")):
            try:
                manifest_count += db.ingest_manifest(str(p))
            except Exception as e:
                print(f"  warning: {p.name}: {e}")
        print(f"Ingested {manifest_count} manifest jobs from {args.manifests_dir}")

    # Show stats
    summary = db.summary()
    print(f"\nDB summary:")
    print(f"  Results: {summary['result_count']}")
    print(f"  Best bpb: {summary['best_bpb']}")
    print(f"  Families: {summary['families']}")
    print(f"  Manifests: {summary['manifest_count']}")

    # Show illegal count
    illegal = db.query("SELECT COUNT(*) as c FROM results WHERE illegal = 1")
    print(f"  Illegal: {illegal[0]['c']}")

    # Show forecast coverage
    forecasts = db.query("SELECT COUNT(*) as c FROM forecasts WHERE asymptote IS NOT NULL")
    print(f"  Forecasted: {forecasts[0]['c']}")

    db.close()
    print(f"\nSaved to {db_path}")


if __name__ == "__main__":
    main()
