"""CLI entry point for the drain daemon.

Usage:
    python -m chronohorn.fleet.drain_cli start --manifest path.jsonl
    python -m chronohorn.fleet.drain_cli stop
    python -m chronohorn.fleet.drain_cli status
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from chronohorn.fleet.daemon import (
    DEFAULT_OUT_DIR,
    LOG_FILE,
    PID_FILE,
    read_pid,
    run_daemon,
)


def cmd_start(args: argparse.Namespace) -> int:
    """Start the drain daemon."""
    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"error: manifest not found: {manifest}", file=sys.stderr)
        return 1

    pid_path = Path(args.pid_file) if args.pid_file else PID_FILE
    log_path = Path(args.log_file) if args.log_file else LOG_FILE

    # Check for existing daemon
    existing = read_pid(pid_path)
    if existing is not None:
        print(f"error: daemon already running (pid {existing})", file=sys.stderr)
        return 1

    if args.daemonize:
        # Fork into background
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable, "-m", "chronohorn.fleet.drain_cli",
            "start",
            "--manifest", str(manifest),
            "--interval", str(args.interval),
            "--pid-file", str(pid_path),
            "--log-file", str(log_path),
        ]
        if args.kill_stale:
            argv.append("--kill-stale")
        if args.job_names:
            for name in args.job_names:
                argv.extend(["--job", name])
        if args.classes:
            for cls in args.classes:
                argv.extend(["--class", cls])
        if args.result_dir:
            argv.extend(["--result-dir", args.result_dir])
        # Do NOT pass --daemonize to the child — it runs in foreground
        # Redirect stdout/stderr to log file so nothing leaks to terminal
        log_fd = open(str(log_path), "a")
        proc = subprocess.Popen(
            argv,
            stdout=log_fd,
            stderr=log_fd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        log_fd.close()
        # Wait for PID file to appear (condition-based, not arbitrary sleep)
        import time
        for _ in range(20):
            child_pid = read_pid(pid_path)
            if child_pid is not None:
                print(f"drain daemon started in background (pid {child_pid})", file=sys.stderr)
                return 0
            time.sleep(0.1)
        print(f"drain daemon forked (pid {proc.pid}) but PID file not written after 2s", file=sys.stderr)
        print(f"check log: {log_path}", file=sys.stderr)
        return 0

    # Foreground mode
    result_out_dir = Path(args.result_dir) if args.result_dir else None
    return run_daemon(
        manifest_path=str(manifest),
        poll_interval=args.interval,
        job_names=args.job_names or (),
        classes=args.classes or (),
        result_out_dir=result_out_dir,
        kill_stale=args.kill_stale,
        pid_path=pid_path,
        log_path=log_path,
    )


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop the drain daemon via SIGTERM."""
    pid_path = Path(args.pid_file) if args.pid_file else PID_FILE
    pid = read_pid(pid_path)
    if pid is None:
        print("no running daemon found", file=sys.stderr)
        return 1

    print(f"sending SIGTERM to pid {pid}", file=sys.stderr)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"process {pid} not found — removing stale PID file", file=sys.stderr)
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass
        return 1
    except PermissionError:
        print(f"permission denied sending signal to pid {pid}", file=sys.stderr)
        return 1

    # Wait for process to exit (condition-based polling)
    import time
    for _ in range(150):  # 15s at 0.1s intervals
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print("daemon stopped", file=sys.stderr)
            return 0
        time.sleep(0.1)

    print(f"daemon (pid {pid}) did not stop within 15s — try SIGKILL", file=sys.stderr)
    return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Print daemon status and recent log lines."""
    pid_path = Path(args.pid_file) if args.pid_file else PID_FILE
    log_path = Path(args.log_file) if args.log_file else LOG_FILE

    pid = read_pid(pid_path)
    if pid is not None:
        print(f"daemon running (pid {pid})")
    else:
        if pid_path.exists():
            print("daemon NOT running (stale PID file exists)")
        else:
            print("daemon NOT running (no PID file)")

    # Show last N log lines
    n_lines = args.lines
    if log_path.exists():
        print(f"\n--- last {n_lines} lines of {log_path} ---")
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lines = text.rstrip("\n").split("\n")
            for line in lines[-n_lines:]:
                print(line)
        except OSError as exc:
            print(f"  (error reading log: {exc})")
    else:
        print(f"\n(no log file at {log_path})")

    return 0 if pid is not None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chronohorn.fleet.drain_cli",
        description="Drain daemon CLI — start, stop, or check status.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- start ---
    p_start = sub.add_parser("start", help="Start the drain daemon")
    p_start.add_argument("--manifest", required=True, help="Path to manifest JSONL file")
    p_start.add_argument("--interval", type=int, default=30, help="Poll interval in seconds (default 30)")
    p_start.add_argument("--daemonize", "-d", action="store_true", help="Fork into background")
    p_start.add_argument("--kill-stale", action="store_true", help="Kill containers exceeding 4x expected duration")
    p_start.add_argument("--job", dest="job_names", action="append", help="Filter to specific job name (repeatable)")
    p_start.add_argument("--class", dest="classes", action="append", help="Filter to resource class (repeatable)")
    p_start.add_argument("--result-dir", default=None, help="Local result output directory")
    p_start.add_argument("--pid-file", default=None, help=f"PID file path (default {PID_FILE})")
    p_start.add_argument("--log-file", default=None, help=f"Log file path (default {LOG_FILE})")
    p_start.set_defaults(func=cmd_start)

    # --- stop ---
    p_stop = sub.add_parser("stop", help="Stop the running daemon")
    p_stop.add_argument("--pid-file", default=None, help=f"PID file path (default {PID_FILE})")
    p_stop.set_defaults(func=cmd_stop)

    # --- status ---
    p_status = sub.add_parser("status", help="Show daemon status and recent logs")
    p_status.add_argument("--lines", "-n", type=int, default=20, help="Number of log lines to show (default 20)")
    p_status.add_argument("--pid-file", default=None, help=f"PID file path (default {PID_FILE})")
    p_status.add_argument("--log-file", default=None, help=f"Log file path (default {LOG_FILE})")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
