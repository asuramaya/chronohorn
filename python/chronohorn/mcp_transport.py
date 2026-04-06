"""Chronohorn MCP stdio transport."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO, Literal

from .mcp import TOOLS, ToolServer

_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class _Message:
    payload: dict[str, Any]
    transport: Literal["framed", "line"]


def _build_tools_list() -> list[dict[str, Any]]:
    tools_list = []
    for name, definition in TOOLS.items():
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param_info in definition.get("parameters", {}).items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if param_info.get("required"):
                required.append(param_name)
        tools_list.append(
            {
                "name": name,
                "description": definition["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        )
    return tools_list


class _Transport:
    def __init__(self, stdin: BinaryIO, stdout: BinaryIO) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._reply_mode: Literal["framed", "line"] = "framed"

    def read_message(self) -> _Message | None:
        while True:
            first_line = self._stdin.readline()
            if not first_line:
                return None
            if first_line in {b"\n", b"\r\n"}:
                continue
            if first_line.lstrip().startswith((b"{", b"[")):
                return _Message(self._read_line_message(first_line), "line")
            if b":" in first_line:
                return _Message(self._read_framed_message(first_line), "framed")
            raise ValueError("Unsupported message framing")

    def remember_mode(self, transport: Literal["framed", "line"]) -> None:
        self._reply_mode = transport

    def send(self, message: dict[str, Any]) -> None:
        encoded = json.dumps(message, separators=(",", ":"), default=str).encode("utf-8")
        if self._reply_mode == "line":
            self._stdout.write(encoded + b"\n")
        else:
            header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
            self._stdout.write(header + encoded)
        self._stdout.flush()

    def _read_framed_message(self, first_line: bytes) -> dict[str, Any]:
        headers = self._parse_headers(first_line)
        length_text = headers.get("content-length")
        if length_text is None:
            raise ValueError("Missing Content-Length header")
        try:
            content_length = int(length_text)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc
        body = self._stdin.read(content_length)
        if len(body) != content_length:
            raise EOFError("Incomplete message body")
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON-RPC payload must be an object")
        return payload

    def _parse_headers(self, first_line: bytes) -> dict[str, str]:
        headers: dict[str, str] = {}
        line = first_line
        while True:
            stripped = line.strip()
            if not stripped:
                break
            name, _, value = stripped.partition(b":")
            if not _:
                raise ValueError("Malformed header line")
            headers[name.decode("ascii").strip().lower()] = value.decode("ascii").strip()
            line = self._stdin.readline()
            if not line:
                raise EOFError("Unexpected EOF while reading headers")
        return headers

    @staticmethod
    def _read_line_message(line: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON request") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON-RPC payload must be an object")
        return payload


def run_stdio_server() -> None:
    server = ToolServer()
    tools_list = _build_tools_list()
    transport = _Transport(sys.stdin.buffer, sys.stdout.buffer)

    while True:
        try:
            message = transport.read_message()
        except EOFError:
            return
        except ValueError as exc:
            _send_error(transport, None, -32700, str(exc))
            continue
        if message is None:
            return

        transport.remember_mode(message.transport)
        request = message.payload
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params", {})

        if not isinstance(method, str):
            _send_error(transport, req_id, -32600, "Invalid request: method must be a string")
            continue
        if params is None:
            params = {}
        if not isinstance(params, dict):
            _send_error(transport, req_id, -32602, "Invalid params: params must be an object")
            continue

        if method == "initialize":
            _send_result(
                transport,
                req_id,
                {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "chronohorn", "version": "0.1.0"},
                },
            )
        elif method == "ping":
            _send_result(transport, req_id, {})
        elif method == "tools/list":
            _send_result(transport, req_id, {"tools": tools_list})
        elif method == "tools/call":
            _handle_tool_call(transport, server, req_id, params)
        elif method in {"notifications/initialized", "notifications/cancelled"}:
            continue
        else:
            _send_error(transport, req_id, -32601, f"Method not found: {method}")


def _handle_tool_call(transport: _Transport, server: ToolServer, req_id: Any, params: dict[str, Any]) -> None:
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    if not isinstance(tool_name, str) or not tool_name:
        _send_error(transport, req_id, -32602, "Invalid params: tool name must be a non-empty string")
        return
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        _send_error(transport, req_id, -32602, "Invalid params: arguments must be an object")
        return

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            result = server.call_tool(tool_name, arguments)
    except Exception as exc:  # noqa: BLE001
        _flush_captured_output(captured_stdout, captured_stderr)
        _send_result(
            transport,
            req_id,
            {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True},
        )
        return

    _flush_captured_output(captured_stdout, captured_stderr)
    if isinstance(result, dict) and result.get("error") and set(result) == {"error"}:
        _send_result(
            transport,
            req_id,
            {"content": [{"type": "text", "text": str(result["error"])}], "isError": True},
        )
        return
    _send_result(
        transport,
        req_id,
        {"content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]},
    )


def _flush_captured_output(stdout_buffer: io.StringIO, stderr_buffer: io.StringIO) -> None:
    stdout_text = stdout_buffer.getvalue().strip()
    stderr_text = stderr_buffer.getvalue().strip()
    if stdout_text:
        sys.stderr.write(stdout_text + "\n")
    if stderr_text:
        sys.stderr.write(stderr_text + "\n")
    if stdout_text or stderr_text:
        sys.stderr.flush()


def _send_result(transport: _Transport, req_id: Any, result: dict[str, Any]) -> None:
    transport.send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _send_error(transport: _Transport, req_id: Any, code: int, message: str) -> None:
    transport.send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help", "help"}:
        print("usage: chronohorn mcp")
        print("")
        print("Run the Chronohorn MCP stdio server for runtime observation, forecasting, and closed-loop control.")
        return 0
    run_stdio_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
