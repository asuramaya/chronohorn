#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
import sqlite3
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "out" / "chronohorn.db"
HANDOFF_JSON = REPO_ROOT / "state" / "next_instance_handoff.json"
HANDOFF_MD = REPO_ROOT / "state" / "NEXT_INSTANCE_HANDOFF.md"
FRONTIER_STATUS_JSON = REPO_ROOT / "state" / "frontier_status.json"
MAX_CELL_LEN = 32767

sys.path[:0] = [str(REPO_ROOT / "python"), str(REPO_ROOT.parent / "decepticons" / "src")]

from chronohorn.db import ChronohornDB  # noqa: E402


@dataclass
class Sheet:
    name: str
    rows: list[dict[str, Any]]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else str(value)
    text = "".join(ch for ch in text if ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) >= 32)
    if len(text) > MAX_CELL_LEN:
        text = text[: MAX_CELL_LEN - 12] + " [truncated]"
    return text


def _flatten_json(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if not value and prefix:
            rows.append({"path": prefix, "value": "{}"})
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_json(item, path))
        return rows
    if isinstance(value, list):
        if not value:
            rows.append({"path": prefix, "value": "[]"})
            return rows
        if all(not isinstance(item, (dict, list)) for item in value):
            rows.append({"path": prefix, "value": json.dumps(value, ensure_ascii=False)})
            return rows
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.extend(_flatten_json(item, path))
        return rows
    rows.append({"path": prefix, "value": value})
    return rows


def _sheet_name(name: str, used: set[str]) -> str:
    cleaned = "".join("_" if ch in '[]:*?/\\' else ch for ch in name).strip() or "Sheet"
    cleaned = cleaned[:31]
    base = cleaned
    suffix = 2
    while cleaned in used:
        tail = f"_{suffix}"
        cleaned = f"{base[:31 - len(tail)]}{tail}"
        suffix += 1
    used.add(cleaned)
    return cleaned


def _col_letter(index: int) -> str:
    letters: list[str] = []
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters.append(chr(65 + rem))
    return "".join(reversed(letters))


def _xml_cell(ref: str, value: Any, style: int = 0) -> str:
    if value is None or value == "":
        return f'<c r="{ref}" s="{style}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" s="{style}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            text = _clean_text(value)
            return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'
        return f'<c r="{ref}" s="{style}" t="n"><v>{value}</v></c>'
    text = _clean_text(value)
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'


def _render_sheet(rows: list[dict[str, Any]]) -> tuple[list[str], str]:
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)

    xml_rows: list[str] = []
    header_cells = []
    for col_idx, header in enumerate(headers, start=1):
        header_cells.append(_xml_cell(f"{_col_letter(col_idx)}1", header, style=1))
    xml_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_idx, row in enumerate(rows, start=2):
        cells: list[str] = []
        for col_idx, header in enumerate(headers, start=1):
            cells.append(_xml_cell(f"{_col_letter(col_idx)}{row_idx}", row.get(header), style=0))
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    last_col = _col_letter(max(1, len(headers)))
    auto_filter = f'<autoFilter ref="A1:{last_col}{max(1, len(rows) + 1)}"/>'
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        f"{auto_filter}"
        "</worksheet>"
    )
    return headers, xml


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for index, name in enumerate(sheet_names, start=1):
        sheets.append(
            f'<sheet name="{escape(name)}" sheetId="{index}" '
            f'r:id="rId{index}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<bookViews><workbookView/></bookViews>'
        f'<sheets>{"".join(sheets)}</sheets>'
        "</workbook>"
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
        '</fonts>'
        '<fills count="2">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )


def _content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(overrides)}'
        "</Types>"
    )


def _root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def _workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(rels)}'
        "</Relationships>"
    )


def _core_props_xml(created_at: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>Chronohorn Snapshot</dc:title>"
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{created_at}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{created_at}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def _app_props_xml(sheet_count: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Codex</Application>"
        f"<Worksheets>{sheet_count}</Worksheets>"
        "</Properties>"
    )


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _json_file_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return [{"path": str(path), "status": "missing"}]
    return _flatten_json(json.loads(path.read_text()), prefix="")


def _markdown_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return [{"line": 1, "text": f"missing: {path}"}]
    return [{"line": idx, "text": line.rstrip("\n")} for idx, line in enumerate(path.read_text().splitlines(), start=1)]


def _table_rows(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    allowed_tables = {
        "configs",
        "events",
        "fleet",
        "forecasts",
        "jobs",
        "journal",
        "predictions",
        "probes",
        "results",
        "schema_version",
    }
    if table not in allowed_tables:
        raise ValueError(f"unsupported table export: {table}")
    cols = [row[1] for row in conn.execute(f"pragma table_info({table})").fetchall()]
    rows = conn.execute(f"select * from {table}").fetchall()
    return [dict(zip(cols, row, strict=False)) for row in rows]


def _git_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    code, stdout, stderr = _run(["git", "status", "--short"])
    if code == 0:
        for line in stdout.splitlines():
            rows.append({"section": "status", "value": line})
    else:
        rows.append({"section": "status_error", "value": stderr.strip()})
    code, stdout, stderr = _run(["git", "log", "--oneline", "-20"])
    if code == 0:
        for line in stdout.splitlines():
            rows.append({"section": "recent_commits", "value": line})
    else:
        rows.append({"section": "log_error", "value": stderr.strip()})
    return rows


def _file_rows(paths: Iterable[Path], base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        stat = path.stat()
        rows.append(
            {
                "relative_path": str(path.relative_to(base)),
                "absolute_path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def _running_pod_rows() -> list[dict[str, Any]]:
    code, stdout, stderr = _run(["kubectl", "get", "pods", "-A", "-o", "json"])
    if code != 0:
        return [{"error": stderr.strip() or "kubectl failed"}]
    payload = json.loads(stdout)
    rows: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        name = item.get("metadata", {}).get("name", "")
        if "cb-" not in name:
            continue
        status = item.get("status", {})
        spec = item.get("spec", {})
        rows.append(
            {
                "namespace": item.get("metadata", {}).get("namespace"),
                "name": name,
                "phase": status.get("phase"),
                "reason": status.get("reason"),
                "node_name": spec.get("nodeName"),
                "pod_ip": status.get("podIP"),
                "start_time": status.get("startTime"),
                "host_ip": status.get("hostIP"),
            }
        )
    if not rows:
        rows.append({"status": "no matching pods"})
    return rows


def _active_manifest_rows(conn: sqlite3.Connection, manifest: str | None) -> list[dict[str, Any]]:
    if not manifest:
        return [{"status": "no active manifest"}]
    cols = [row[1] for row in conn.execute("pragma table_info(jobs)").fetchall()]
    rows = conn.execute(
        "select * from jobs where manifest = ? order by case state when 'running' then 0 when 'pending' then 1 when 'completed' then 2 else 3 end, name",
        (manifest,),
    ).fetchall()
    return [dict(zip(cols, row, strict=False)) for row in rows]


def _select_mutation_rows(board: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "name",
        "best_bpb",
        "best_tok_s",
        "mutation_score",
        "median_bpb_delta_vs_base",
        "best_bpb_delta_vs_base",
        "median_speed_ratio_vs_base",
        "artifact_budget_ok",
        "compute_budget_ok",
        "constant_state_inference",
        "scale_survived",
        "context_survived",
        "next_action",
        "gates_remaining",
        "mutation_label",
        "run_count",
        "lane_count",
        "signature",
    ]
    rows: list[dict[str, Any]] = []
    for entry in board:
        row = {key: entry.get(key) for key in keep}
        if isinstance(row.get("gates_remaining"), list):
            row["gates_remaining"] = ", ".join(str(x) for x in row["gates_remaining"])
        rows.append(row)
    return rows


def _overview_rows(
    snapshot_at: str,
    handoff: dict[str, Any],
    db_summary: dict[str, Any],
    trust_summary: dict[str, Any],
    population_summary: dict[str, Any],
    cost_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"section": "snapshot", "key": "generated_at", "value": snapshot_at},
        {"section": "snapshot", "key": "repo_root", "value": str(REPO_ROOT)},
        {"section": "snapshot", "key": "db_path", "value": str(DB_PATH)},
        {"section": "repo", "key": "branch", "value": handoff.get("branch")},
        {"section": "repo", "key": "head_commit", "value": handoff.get("head_commit")},
        {"section": "repo", "key": "dirty_worktree", "value": handoff.get("dirty_worktree")},
        {"section": "queue", "key": "active_manifest", "value": handoff.get("active_manifest")},
        {"section": "queue", "key": "running", "value": handoff.get("queue", {}).get("running")},
        {"section": "queue", "key": "pending", "value": handoff.get("queue", {}).get("pending")},
        {"section": "queue", "key": "completed", "value": handoff.get("queue", {}).get("completed")},
        {"section": "frontier", "key": "best_bpb_any", "value": db_summary.get("best_bpb_any")},
        {"section": "frontier", "key": "provisional_best_bpb", "value": db_summary.get("provisional_best_bpb")},
        {"section": "frontier", "key": "admissible_best_bpb", "value": db_summary.get("admissible_best_bpb")},
        {"section": "trust", "key": "admissible", "value": trust_summary.get("counts", {}).get("admissible")},
        {"section": "trust", "key": "provisional", "value": trust_summary.get("counts", {}).get("provisional")},
        {"section": "trust", "key": "dataset_anchored", "value": trust_summary.get("metric_states", {}).get("dataset_anchored")},
        {"section": "trust", "key": "legacy_result_schema", "value": trust_summary.get("metric_states", {}).get("legacy_result_schema")},
        {"section": "cost", "key": "total_runs", "value": cost_summary.get("total_runs")},
        {"section": "cost", "key": "total_gpu_hours", "value": cost_summary.get("total_gpu_hours")},
        {"section": "population", "key": "result_count", "value": population_summary.get("result_count")},
        {"section": "population", "key": "controlled_count", "value": population_summary.get("populations", {}).get("controlled", {}).get("count")},
    ]
    return rows


def _write_xlsx(path: Path, sheets: list[Sheet], created_at: str) -> None:
    used_names: set[str] = set()
    rendered: list[tuple[str, str]] = []
    for sheet in sheets:
        name = _sheet_name(sheet.name, used_names)
        _, xml = _render_sheet(sheet.rows or [{"status": "empty"}])
        rendered.append((name, xml))

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml(len(rendered)))
        zf.writestr("_rels/.rels", _root_rels_xml())
        zf.writestr("xl/workbook.xml", _workbook_xml([name for name, _ in rendered]))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(rendered)))
        zf.writestr("xl/styles.xml", _styles_xml())
        zf.writestr("docProps/core.xml", _core_props_xml(created_at))
        zf.writestr("docProps/app.xml", _app_props_xml(len(rendered)))
        for idx, (_, xml) in enumerate(rendered, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", xml)


def build_workbook(output_path: Path) -> None:
    snapshot_at = datetime.now().astimezone().isoformat(timespec="seconds")
    handoff_json = json.loads(HANDOFF_JSON.read_text()) if HANDOFF_JSON.exists() else {}

    db = ChronohornDB(str(DB_PATH))
    db_summary = db.summary()
    trust_summary = db.trust_summary()
    population_summary = db.population_summary()
    cost_summary = db.cost_summary()
    frontier_rows = db.frontier(50, trust="provisional")
    mutation_rows = _select_mutation_rows(db.mutation_leaderboard(top_k=50))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    active_manifest = handoff_json.get("active_manifest")

    sheets: list[Sheet] = [
        Sheet("Overview", _overview_rows(snapshot_at, handoff_json, db_summary, trust_summary, population_summary, cost_summary)),
        Sheet("DB Summary", _flatten_json(db_summary)),
        Sheet("Trust Summary", _flatten_json(trust_summary)),
        Sheet("Population Summary", _flatten_json(population_summary)),
        Sheet("Cost Summary", _flatten_json(cost_summary)),
        Sheet("Handoff JSON", _json_file_rows(HANDOFF_JSON)),
        Sheet("Handoff MD", _markdown_rows(HANDOFF_MD)),
        Sheet("Frontier Status JSON", _json_file_rows(FRONTIER_STATUS_JSON)),
        Sheet("Frontier Top 50", frontier_rows),
        Sheet("Mutation Leaderboard", mutation_rows),
        Sheet("Active Manifest Jobs", _active_manifest_rows(conn, active_manifest)),
        Sheet("Running Pods", _running_pod_rows()),
        Sheet("Git State", _git_rows()),
        Sheet("Manifest Files", _file_rows((REPO_ROOT / "manifests").glob("*.jsonl"), REPO_ROOT)),
        Sheet("Result Files", _file_rows((REPO_ROOT / "out" / "results").glob("*.json"), REPO_ROOT)),
    ]

    for table in [
        "configs",
        "events",
        "fleet",
        "forecasts",
        "jobs",
        "journal",
        "predictions",
        "probes",
        "results",
        "schema_version",
    ]:
        sheets.append(Sheet(f"table_{table}", _table_rows(conn, table)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_xlsx(output_path, sheets, created_at=snapshot_at)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Chronohorn repo state and DB data to a single .xlsx workbook.")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "out" / "chronohorn_snapshot.xlsx"),
        help="Output workbook path.",
    )
    args = parser.parse_args()
    output_path = Path(args.output).resolve()
    build_workbook(output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
