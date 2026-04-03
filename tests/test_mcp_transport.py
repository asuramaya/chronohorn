from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_mcp(payload: bytes, *, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "chronohorn", "mcp"],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        check=False,
    )


def _frame(message: dict) -> bytes:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def _parse_framed_responses(stream: bytes) -> list[dict]:
    responses = []
    cursor = 0
    while cursor < len(stream):
        boundary = stream.index(b"\r\n\r\n", cursor)
        header_block = stream[cursor:boundary].decode("ascii")
        headers = {}
        for line in header_block.split("\r\n"):
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
        length = int(headers["content-length"])
        body_start = boundary + 4
        body_end = body_start + length
        responses.append(json.loads(stream[body_start:body_end].decode("utf-8")))
        cursor = body_end
    return responses


def test_mcp_transport_accepts_content_length_initialize(tmp_path: Path):
    proc = _run_mcp(
        _frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        cwd=tmp_path,
    )

    assert proc.returncode == 0
    assert proc.stderr == b""

    responses = _parse_framed_responses(proc.stdout)
    assert len(responses) == 1
    assert "error" not in responses[0]
    assert responses[0]["id"] == 1
    assert responses[0]["result"]["protocolVersion"] == "2024-11-05"
    assert responses[0]["result"]["capabilities"]["tools"]["listChanged"] is False


def test_mcp_transport_tools_list_and_call_are_framed(tmp_path: Path):
    payload = b"".join(
        [
            _frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            _frame({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
            _frame(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "chronohorn_reset", "arguments": {}},
                }
            ),
        ]
    )
    proc = _run_mcp(payload, cwd=tmp_path)

    assert proc.returncode == 0
    assert proc.stderr == b""

    responses = _parse_framed_responses(proc.stdout)
    assert [response["id"] for response in responses] == [1, 2, 3]

    tools = responses[1]["result"]["tools"]
    assert any(tool["name"] == "chronohorn_reset" for tool in tools)

    call_content = responses[2]["result"]["content"][0]["text"]
    call_result = json.loads(call_content)
    assert call_result["status"] == "no-op"


def test_mcp_transport_preserves_line_mode_compatibility(tmp_path: Path):
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}).encode("utf-8") + b"\n"
    proc = _run_mcp(request, cwd=tmp_path)

    assert proc.returncode == 0
    assert proc.stderr == b""
    assert b"Content-Length:" not in proc.stdout

    responses = [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines() if line.strip()]
    assert responses == [{"jsonrpc": "2.0", "id": 1, "result": {}}]


def test_module_entrypoint_help_uses_sys_argv(tmp_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "chronohorn.mcp_transport", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
        check=False,
    )

    assert proc.returncode == 0
    assert "usage: chronohorn mcp" in proc.stdout.decode("utf-8")
    assert proc.stderr == b""
