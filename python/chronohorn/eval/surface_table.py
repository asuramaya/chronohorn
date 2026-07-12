"""Assemble the kNN surface table from run logs — with the smoothing/knowledge split.

Ruling 2c260e7c (the relevance square) fixed the reading rule: a fold's raw delta
CONFLATES two different things, and quoting it bare is a category error.

    delta(fold store -> fold queries)      what the RELEVANT archive buys
  - delta(enwik8 store -> fold queries)    what ANY archive buys (calibration repair)
  ------------------------------------------------------------------
  = KNOWLEDGE                              what relevance itself buys

The control is not a nicety. At 64M, enwik8->world (-0.62) BEAT world->world (-0.42):
an irrelevant English archive out-earned the relevant multilingual one, because most
of the lift was the base's calibration being repaired by diffuse fallback votes, not
knowledge being retrieved. A table without controls would have read that as "the world
store knows a lot", which is backwards.

The independent discriminator (free, needs no control): kNN-ALONE vs BASE.
  alone >> base  -> smoothing fold (archive cannot predict; it only softens the base)
  alone << base  -> KNOWLEDGE fold (archive out-predicts the body outright)

This module parses the eval's logs read-only. It does not re-run anything, and it
refuses to invent rows: a missing control prints as MISSING, never as zero — an
absent measurement is not a measurement of zero.

Run:  python -m chronohorn.eval.surface_table out/results
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

RE_BANK = re.compile(r"bank lifted:.*?store_bytes=(\d+), corpus=([\w:.]+)@(\d+), queries=([\w:.]+)")
RE_BASE = re.compile(r"base \(windowed logits\): test ([\d.]+)")
RE_ALONE = re.compile(r"kNN-alone .*?-> TEST ([\d.]+)")
RE_MARGINAL = re.compile(r"marginal-alone: TEST ([\d.]+)")
RE_MIX = re.compile(
    r"kNN-LM mix: lam=([\d.]+) -> TEST ([\d.]+) \(base [\d.]+, paired delta ([-\d.]+)\s*"
    r"\+/-\s*([\d.]+), windows improved (\d+)/(\d+)\)")
RE_ORACLE = re.compile(
    r"oracle gate: TEST ([\d.]+) \(delta ([-\d.]+) vs base; kNN wins ([\d.]+)% of positions\)")


@dataclass
class Row:
    log: str
    store: str
    queries: str
    store_bytes: int
    base: float | None = None
    alone: float | None = None
    lam: float | None = None
    mix: float | None = None
    delta: float | None = None
    delta_err: float | None = None
    wins: str | None = None
    oracle: float | None = None
    oracle_delta: float | None = None
    oracle_win_frac: float | None = None
    complete: bool = False

    @property
    def fold(self) -> str:
        """The domain being PREDICTED — the table's row key."""
        return self.queries.split(":")[-1]

    @property
    def is_control(self) -> bool:
        """enwik8 store against a foreign fold: the smoothing (no-relevance) arm."""
        return self.store.startswith("enwik8") and not self.queries.startswith("enwik8")


def parse_log(path: Path) -> Row | None:
    text = path.read_text(errors="replace")
    m = RE_BANK.search(text)
    if not m:
        return None
    row = Row(log=path.name, store_bytes=int(m.group(1)), store=m.group(2), queries=m.group(4))

    if (b := RE_BASE.search(text)):
        row.base = float(b.group(1))
    # MARGINAL FIRST. A marginal-mix run logs BOTH a kNN-alone line (it still searches,
    # it just doesn't MIX with the result) and a marginal-alone line. Matching kNN-alone
    # first silently filed the no-retrieval arm as an enwik8 CONTROL, where it overwrote
    # the real 64M control and rewrote world's knowledge column. An arm mislabelled as a
    # control is worse than a missing one: it answers with a number.
    if (a := RE_MARGINAL.search(text)):          # the no-retrieval arm
        row.alone = float(a.group(1))
        row.store = "marginal"                   # relabel: there is no archive here
    elif (a := RE_ALONE.search(text)):
        row.alone = float(a.group(1))
    if (x := RE_MIX.search(text)):
        row.lam, row.mix = float(x.group(1)), float(x.group(2))
        row.delta, row.delta_err = float(x.group(3)), float(x.group(4))
        row.wins = f"{x.group(5)}/{x.group(6)}"
    if (o := RE_ORACLE.search(text)):
        row.oracle, row.oracle_delta = float(o.group(1)), float(o.group(2))
        row.oracle_win_frac = float(o.group(3))
    # "complete" means the run reached its final line, not merely that it started.
    row.complete = row.delta is not None and row.oracle is not None
    return row


def render(rows: list[Row]) -> str:
    out: list[str] = []
    done = [r for r in rows if r.complete]
    partial = [r for r in rows if not r.complete]

    out.append("=" * 108)
    out.append("kNN SURFACE — raw rows (every completed run)")
    out.append("=" * 108)
    out.append(f"{'store':<16}{'queries':<16}{'MB':>6}  {'base':>7}{'alone':>8}"
               f"{'lam':>6}{'mix':>8}{'delta':>9}{'+/-':>7}  {'win':>6} {'oracle':>8}{'purse':>8}{'kNN%':>6}")
    out.append("-" * 108)
    for r in sorted(done, key=lambda r: (r.fold, r.store)):
        out.append(
            f"{r.store:<16}{r.queries:<16}{r.store_bytes // 1_000_000:>5}M  "
            f"{r.base:>7.4f}{r.alone:>8.4f}{r.lam:>6.2f}{r.mix:>8.4f}"
            f"{r.delta:>9.4f}{r.delta_err:>7.4f}  {r.wins:>6} "
            f"{r.oracle:>8.4f}{r.oracle_delta:>8.4f}{r.oracle_win_frac:>6.1f}")

    # ---- the decomposition: fold minus control ----
    folds = {}
    for r in done:
        if r.store == "marginal":
            folds.setdefault(r.fold, {})["marginal"] = r
        elif r.is_control:
            folds.setdefault(r.fold, {})["control"] = r
        elif r.fold == r.store.split(":")[-1]:
            folds.setdefault(r.fold, {})["fold"] = r

    out.append("")
    out.append("=" * 108)
    out.append("THE SPLIT — per ruling 2c260e7c: quote fold-minus-control, never the raw fold delta")
    out.append("=" * 108)
    out.append(f"{'fold':<10}{'discriminator':<26}{'total':>9}{'smoothing':>11}{'knowledge':>11}"
               f"{'free?':>9}   verdict")
    out.append("-" * 108)
    for name in sorted(folds):
        f = folds[name].get("fold")
        c = folds[name].get("control")
        g = folds[name].get("marginal")
        if f is None:
            out.append(f"{name:<10}{'(no fold run)':<26}{'MISSING — not zero':>31}")
            continue

        disc = (f"alone {f.alone:.2f} vs base {f.base:.2f}"
                if f.alone is not None and f.base is not None else "?")
        kind = ("KNOWLEDGE" if (f.alone is not None and f.base is not None and f.alone < f.base)
                else "smoothing")

        # The HOME fold has no split to make: the store IS the domain, so there is no
        # relevant-vs-irrelevant contrast to draw and no control to wait for. Its delta
        # is in-domain retrieval by definition. Demanding a control here would be a
        # category error dressed as rigour.
        if not f.queries.startswith("pentad:"):
            out.append(f"{name:<10}{disc:<26}{f.delta:>9.4f}{'n/a':>11}{'n/a':>11}{'—':>9}"
                       f"   home fold — store IS the domain; delta is in-domain retrieval, no split applies")
            continue

        if c is None:
            out.append(f"{name:<10}{disc:<26}{f.delta:>9.4f}{'MISSING':>11}{'MISSING':>11}"
                       f"{'—':>9}   {kind} fold; CONTROL PENDING — the split cannot be quoted")
            continue

        smoothing = c.delta
        knowledge = f.delta - c.delta          # what relevance itself bought
        free = f"{g.delta:>+.4f}" if g is not None else "—"
        note = kind + " fold"
        if knowledge > 0:
            note += "; relevance SUBTRACTS (control beats the relevant store)"
        out.append(f"{name:<10}{disc:<26}{f.delta:>9.4f}{smoothing:>11.4f}{knowledge:>+11.4f}"
                   f"{free:>9}   {note}")

    out.append("")
    out.append("  total     = delta(fold store -> fold queries)      what the relevant archive buys")
    out.append("  smoothing = delta(enwik8 store -> fold queries)    what ANY archive buys (calibration repair)")
    out.append("  knowledge = total - smoothing                      what RELEVANCE itself buys (negative = real)")
    out.append("  free?     = delta(marginal, NO retrieval)          how much smoothing costs no archive at all")

    if partial:
        out.append("")
        out.append("IN FLIGHT / INCOMPLETE (reached no final line — reported, never silently dropped):")
        for r in partial:
            got = [n for n, v in (("base", r.base), ("alone", r.alone),
                                  ("mix", r.delta), ("oracle", r.oracle)) if v is not None]
            out.append(f"  {r.log:<44} {r.store} -> {r.queries}: has {got or ['nothing']}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", nargs="?", default="out/results")
    p.add_argument("--glob", default="knn_*.log")
    a = p.parse_args(argv)
    rows = [r for r in (parse_log(f) for f in sorted(Path(a.results_dir).glob(a.glob))) if r]
    if not rows:
        print(f"no kNN logs matched {a.glob} in {a.results_dir}")
        return 1
    print(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
