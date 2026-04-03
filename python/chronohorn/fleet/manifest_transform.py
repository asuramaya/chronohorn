"""Filter and mutate manifest rows without editing scan code."""
from __future__ import annotations

import copy
import fnmatch
import json
import re
from pathlib import Path
from typing import Any, Sequence


def filter_manifest(
    rows: list[dict[str, Any]],
    *,
    name_pattern: str | None = None,
    family: str | None = None,
    resource_class: str | None = None,
) -> list[dict[str, Any]]:
    result = rows
    if name_pattern:
        result = [r for r in result if fnmatch.fnmatch(r.get("name", ""), name_pattern)]
    if family:
        result = [r for r in result if r.get("family") == family]
    if resource_class:
        result = [r for r in result if r.get("resource_class") == resource_class]
    return result


def mutate_manifest(
    rows: list[dict[str, Any]],
    *,
    steps: int | None = None,
    seed: int | None = None,
    learning_rate: float | None = None,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        row = copy.deepcopy(row)
        cmd = row.get("command", "")

        if steps is not None:
            old_steps = row.get("steps", 1000)
            row["steps"] = steps
            cmd = re.sub(r"--steps \S+", f"--steps {steps}", cmd)
            row["work_tokens"] = steps * row.get("seq_len", 256) * row.get("batch_size", 16)
            old_name = row["name"]
            row["name"] = f"{old_name}-s{steps}"
            cmd = cmd.replace(f"{old_name}.json", f"{row['name']}.json")

        if seed is not None:
            old_seed = row.get("seed", 42)
            row["seed"] = seed
            cmd = re.sub(r"--seed \S+", f"--seed {seed}", cmd)
            old_name = row["name"]
            if f"seed{old_seed}" in old_name:
                row["name"] = old_name.replace(f"seed{old_seed}", f"seed{seed}")
            else:
                row["name"] = f"{old_name}-seed{seed}"
            cmd = cmd.replace(f"{old_name}.json", f"{row['name']}.json")

        if learning_rate is not None:
            row["learning_rate"] = learning_rate
            if "--learning-rate" in cmd:
                cmd = re.sub(r"--learning-rate \S+", f"--learning-rate {learning_rate}", cmd)
            else:
                cmd = re.sub(r"--lr \S+", f"--lr {learning_rate}", cmd)

        row["command"] = cmd
        result.append(row)
    return result


def load_and_transform(
    manifest_path: Path,
    *,
    name_pattern: str | None = None,
    steps: int | None = None,
    seed: int | None = None,
    learning_rate: float | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    from chronohorn.fleet.dispatch import load_manifest

    rows = load_manifest(manifest_path)
    if name_pattern:
        rows = filter_manifest(rows, name_pattern=name_pattern)
    rows = mutate_manifest(rows, steps=steps, seed=seed, learning_rate=learning_rate)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(f"# Transformed from {manifest_path.name}\n")
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    return rows
