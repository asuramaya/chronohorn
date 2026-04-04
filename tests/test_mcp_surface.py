from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from chronohorn.mcp import TOOLS, ToolServer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_result(
    path: Path,
    *,
    arch: str,
    bpb: float,
    train_bpb: float,
    params: int,
    steps: int,
    seq_len: int,
    seed: int,
    readout: str,
    extra: dict | None = None,
) -> None:
    payload = {
        "model": {
            "test_bpb": bpb,
            "train_bpb": train_bpb,
            "params": params,
            "architecture": arch,
            "readout": readout,
            "overfit_pct": round((bpb - train_bpb) / train_bpb * 100, 2),
        },
        "training": {
            "performance": {
                "steps_completed": steps,
                "tokens_per_second": 1000.0 + seed,
                "estimated_sustained_tflops": 1.0 + seed * 0.1,
                "elapsed_sec": 60.0 + seed,
            },
            "probes": [
                {"step": max(100, steps // 4), "bpb": bpb + 0.20, "tflops": 0.2, "elapsed_sec": 15.0},
                {"step": max(200, steps // 2), "bpb": bpb + 0.10, "tflops": 0.5, "elapsed_sec": 30.0},
                {"step": steps, "bpb": bpb, "tflops": 1.0, "elapsed_sec": 60.0},
            ],
        },
        "config": {
            "train": {
                "steps": steps,
                "seq_len": seq_len,
                "batch_size": 4,
                "seed": seed,
                "architecture": arch,
                "readout": readout,
            }
        },
    }
    if extra:
        payload["config"]["train"].update(extra)
        payload["model"].update({k: v for k, v in extra.items() if k in {"num_heads", "patch_size", "num_blocks"}})
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(path: Path, *, local_only: bool) -> None:
    launcher = "local_command" if local_only else "managed_command"
    backend = "metal" if local_only else "cuda"
    command_prefix = "out/results" if local_only else "/run/results"
    jobs = [
        {
            "name": "group-alpha-1000",
            "family": "causal-bank",
            "backend": backend,
            "resource_class": "cpu_serial",
            "launcher": launcher,
            "command": f"python train.py --steps 1000 --json {command_prefix}/group-alpha-1000.json",
            "scale": 14.0,
            "steps": 1000,
            "learning_rate": 0.001,
            "seq_len": 128,
            "readout": "mlp",
            "seed": 1,
        },
        {
            "name": "group-alpha-2000",
            "family": "causal-bank",
            "backend": backend,
            "resource_class": "cpu_serial",
            "launcher": launcher,
            "command": f"python train.py --steps 2000 --json {command_prefix}/group-alpha-2000.json",
            "scale": 14.0,
            "steps": 2000,
            "learning_rate": 0.001,
            "seq_len": 256,
            "readout": "mlp",
            "seed": 1,
        },
        {
            "name": "group-alpha-seed2",
            "family": "causal-bank",
            "backend": backend,
            "resource_class": "cpu_serial",
            "launcher": launcher,
            "command": f"python train.py --steps 1000 --json {command_prefix}/group-alpha-seed2.json",
            "scale": 14.0,
            "steps": 1000,
            "learning_rate": 0.001,
            "seq_len": 128,
            "readout": "mlp",
            "seed": 2,
        },
        {
            "name": "branch-beta-1000",
            "family": "polyhash",
            "backend": backend,
            "resource_class": "cpu_serial",
            "launcher": launcher,
            "command": f"python train.py --steps 1000 --json {command_prefix}/branch-beta-1000.json",
            "scale": 9.0,
            "steps": 1000,
            "learning_rate": 0.002,
            "seq_len": 192,
            "readout": "linear",
            "seed": 3,
        },
    ]
    path.write_text("\n".join(json.dumps(job) for job in jobs) + "\n", encoding="utf-8")


def _build_fixture(tmp_path: Path, *, local_only: bool) -> dict[str, Path]:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    manifest_path = manifests_dir / "sample.jsonl"
    _write_manifest(manifest_path, local_only=local_only)
    empty_manifest_path = manifests_dir / "empty.jsonl"
    empty_manifest_path.write_text("", encoding="utf-8")

    _make_result(
        results_dir / "group-alpha-1000.json",
        arch="causal-bank",
        bpb=2.30,
        train_bpb=2.22,
        params=1_000_000,
        steps=1000,
        seq_len=128,
        seed=1,
        readout="mlp",
        extra={"num_heads": 2},
    )
    _make_result(
        results_dir / "group-alpha-2000.json",
        arch="causal-bank",
        bpb=2.18,
        train_bpb=2.10,
        params=1_200_000,
        steps=2000,
        seq_len=256,
        seed=1,
        readout="mlp",
        extra={"num_heads": 4},
    )
    _make_result(
        results_dir / "group-alpha-seed2.json",
        arch="causal-bank",
        bpb=2.33,
        train_bpb=2.24,
        params=1_000_000,
        steps=1000,
        seq_len=128,
        seed=2,
        readout="mlp",
        extra={"num_heads": 2},
    )
    _make_result(
        results_dir / "branch-beta-1000.json",
        arch="polyhash",
        bpb=2.46,
        train_bpb=2.35,
        params=900_000,
        steps=1000,
        seq_len=192,
        seed=3,
        readout="linear",
        extra={"patch_size": 4},
    )

    token_path = tmp_path / "tokens.bin"
    np.array([1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 6, 7, 1, 2, 3, 8], dtype=np.uint16).tofile(token_path)

    return {
        "manifest": manifest_path,
        "empty_manifest": empty_manifest_path,
        "results_dir": results_dir,
        "token_path": token_path,
    }


def _surface_tool_args(paths: dict[str, Path]) -> dict[str, dict]:
    return {
        "chronohorn_manifests": {"manifest_paths": [str(paths["manifest"])]},
        "chronohorn_results": {"result_paths": [str(paths["results_dir"])]},
        "chronohorn_forecast": {},
        "chronohorn_runtime_status": {},
        "chronohorn_launches": {"top_k": 10},
        "chronohorn_records": {"kind": "results", "top_k": 10},
        "chronohorn_status": {},
        "chronohorn_frontier": {"top_k": 5, "format": "text"},
        "chronohorn_control_recommend": {
            "manifest_paths": [str(paths["manifest"])],
            "result_paths": [str(paths["results_dir"])],
            "max_launches": 0,
        },
        "chronohorn_control_act": {"manifest_paths": [], "result_paths": [], "max_launches": 0, "allow_stop": False},
        "chronohorn_reset": {},
        "chronohorn_fleet_dispatch": {"manifest_path": str(paths["manifest"]), "dry_run": True},
        "chronohorn_fleet_drain_tick": {"manifest_path": str(paths["empty_manifest"])},
        "chronohorn_fleet_status": {"manifest_path": str(paths["empty_manifest"])},
        "chronohorn_learning_curve": {"name": "group-alpha-1000", "format": "text"},
        "chronohorn_compare": {"names": ["group-alpha-1000", "group-alpha-2000"]},
        "chronohorn_marginal_rank": {"top_k": 10},
        "chronohorn_saturation": {"name": "group-alpha-1000"},
        "chronohorn_saturation_frontier": {"top_k": 10},
        "chronohorn_auto_deepen": {"top_n": 2, "dry_run": True, "target_steps": 4000},
        "chronohorn_artifact_check": {"name": "group-alpha-1000"},
        "chronohorn_subscribe": {},
        "chronohorn_query": {"sql": "SELECT name, bpb FROM results ORDER BY bpb"},
        "chronohorn_build_table": {
            "data_path": str(paths["token_path"]),
            "output_path": str(paths["token_path"].parent / "out" / "table.npz"),
        },
        "chronohorn_events": {"limit": 10},
        "chronohorn_drain_status": {},
        "chronohorn_list_manifests": {},
        "chronohorn_flag_illegal": {"name": "branch-beta-1000", "illegal": True},
        "chronohorn_config_diff": {"name1": "group-alpha-1000", "name2": "group-alpha-2000"},
        "chronohorn_what_varied": {"limit": 10},
        "chronohorn_cost": {},
        "chronohorn_terminal_dashboard": {"top_k": 5},
        "chronohorn_changelog": {"hours": 24},
        "chronohorn_journal_write": {
            "kind": "observation",
            "content": "stress test note",
            "run_name": "group-alpha-1000",
            "tags": ["test"],
        },
        "chronohorn_journal_read": {"limit": 10},
        "chronohorn_predict": {"name": "group-alpha-1000", "steps": 4000},
        "chronohorn_prediction_audit": {},
        "chronohorn_emit_matrix": {
            "name_template": "matrix-{lr}-{heads}",
            "base": {"backend": "cuda", "steps": 1000},
            "sweep": {"lr": [0.001, 0.002], "heads": [2, 4]},
        },
        "chronohorn_seed_analysis": {},
        "chronohorn_interpret": {"name": "group-alpha-1000"},
        "chronohorn_frontier_velocity": {},
        "chronohorn_branch_health": {"prefix": "group-alpha"},
        "chronohorn_experiment_groups": {},
        "chronohorn_suggest_next": {},
        "chronohorn_axis_analysis": {},
        "chronohorn_architecture_boundary": {},
    }


class _MCPClient:
    def __init__(self, cwd: Path) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "chronohorn", "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

    def call(self, message: dict) -> dict:
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(frame)
        self._proc.stdin.flush()

        header = b""
        while b"\r\n\r\n" not in header:
            chunk = self._proc.stdout.read(1)
            if not chunk:
                raise RuntimeError("EOF waiting for MCP response header")
            header += chunk
        head, _, rest = header.partition(b"\r\n\r\n")
        headers = {}
        for line in head.decode("ascii").split("\r\n"):
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
        length = int(headers["content-length"])
        body = rest
        if len(body) < length:
            body += self._proc.stdout.read(length - len(body))
        return json.loads(body.decode("utf-8"))

    def notify(self, message: dict) -> None:
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        assert self._proc.stdin is not None
        self._proc.stdin.write(frame)
        self._proc.stdin.flush()

    def close(self) -> str:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.terminate()
        self._proc.wait(timeout=5)
        assert self._proc.stderr is not None
        return self._proc.stderr.read().decode("utf-8", "replace")


def _run_surface_pass(
    client: _MCPClient,
    paths: dict[str, Path],
    *,
    start_id: int,
) -> tuple[int, list[tuple[str, str]]]:
    tool_names = set(TOOLS)
    tools_list = client.call({"jsonrpc": "2.0", "id": start_id, "method": "tools/list", "params": {}})
    listed_names = {tool["name"] for tool in tools_list["result"]["tools"]}
    assert listed_names == tool_names

    ping = client.call({"jsonrpc": "2.0", "id": start_id + 1, "method": "ping", "params": {}})
    assert ping["result"] == {}

    failures: list[tuple[str, str]] = []
    next_id = start_id + 2
    for tool_name, arguments in _surface_tool_args(paths).items():
        response = client.call(
            {
                "jsonrpc": "2.0",
                "id": next_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        )
        next_id += 1
        if "error" in response:
            failures.append((tool_name, json.dumps(response["error"], sort_keys=True)))
            continue
        result = response["result"]
        if result.get("isError"):
            failures.append((tool_name, result["content"][0]["text"]))
    return next_id, failures


def test_mcp_full_surface_smoke(tmp_path: Path):
    paths = _build_fixture(tmp_path, local_only=True)
    client = _MCPClient(tmp_path)

    initialize = client.call({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert initialize["result"]["protocolVersion"] == "2024-11-05"
    client.notify({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    _, failures = _run_surface_pass(client, paths, start_id=2)

    stderr_text = client.close()

    assert failures == []
    assert "Traceback" not in stderr_text


def test_mcp_full_surface_stress_repeated_round_trips(tmp_path: Path):
    paths = _build_fixture(tmp_path, local_only=True)
    client = _MCPClient(tmp_path)

    initialize = client.call({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert initialize["result"]["protocolVersion"] == "2024-11-05"
    client.notify({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    next_id = 2
    failures: list[tuple[int, str, str]] = []
    for round_index in range(3):
        next_id, round_failures = _run_surface_pass(client, paths, start_id=next_id)
        failures.extend((round_index, tool_name, detail) for tool_name, detail in round_failures)

    stderr_text = client.close()

    assert failures == []
    assert "Traceback" not in stderr_text


def test_control_recommend_skips_runtime_probe_and_zero_launches(
    tmp_path: Path,
    monkeypatch,
):
    paths = _build_fixture(tmp_path, local_only=False)
    monkeypatch.chdir(tmp_path)

    def _probe_should_not_run(*args, **kwargs):
        raise AssertionError("probe_fleet_state should not run when probe_runtime is false")

    monkeypatch.setattr("chronohorn.control.policy.probe_fleet_state", _probe_should_not_run)

    server = ToolServer()
    server.call_tool("chronohorn_manifests", {"manifest_paths": [str(paths["manifest"])]})
    server.call_tool("chronohorn_results", {"result_paths": [str(paths["results_dir"])]})

    result = server.call_tool(
        "chronohorn_control_recommend",
        {
            "manifest_paths": [str(paths["manifest"])],
            "result_paths": [str(paths["results_dir"])],
            "probe_runtime": False,
            "max_launches": 0,
        },
    )

    assert "error" not in result
    # All 4 manifest jobs have matching results, so pending_count should be 0
    # (the planner correctly recognizes completed runs)
    assert result["summary"]["pending_count"] == 0
    assert result["summary"]["action_counts"]["launch"] == 0
