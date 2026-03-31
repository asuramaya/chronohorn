"""Chronohorn MCP stdio transport — JSON-RPC over stdin/stdout."""

from __future__ import annotations

import json
import sys
from typing import Any, Sequence

from .mcp import TOOLS, ToolServer


def run_stdio_server() -> None:
    server = ToolServer()
    tools_list = []
    for name, definition in TOOLS.items():
        tool_schema = {
            "name": name,
            "description": definition["description"],
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }
        for param_name, param_info in definition.get("parameters", {}).items():
            tool_schema["inputSchema"]["properties"][param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if param_info.get("required"):
                tool_schema["inputSchema"]["required"].append(param_name)
        tools_list.append(tool_schema)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _send_error(None, -32700, "Parse error")
            continue

        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            _send_result(
                req_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "chronohorn", "version": "0.1.0"},
                },
            )
        elif method == "tools/list":
            _send_result(req_id, {"tools": tools_list})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            try:
                result = server.call_tool(tool_name, arguments)
                _send_result(
                    req_id,
                    {"content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]},
                )
            except Exception as exc:  # noqa: BLE001
                _send_result(
                    req_id,
                    {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True},
                )
        elif method == "notifications/initialized":
            continue
        else:
            _send_error(req_id, -32601, f"Method not found: {method}")


def _send_result(req_id: Any, result: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
    sys.stdout.flush()


def _send_error(req_id: Any, code: int, message: str) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}) + "\n")
    sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or [])
    if args and args[0] in {"-h", "--help", "help"}:
        print("usage: chronohorn mcp")
        print("")
        print("Run the Chronohorn MCP stdio server for runtime observation, forecasting, and closed-loop control.")
        return 0
    run_stdio_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
