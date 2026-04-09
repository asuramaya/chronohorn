"""Extended DB tests — journal, predictions, config_diff, what_varied, branch_health, frontier_velocity."""
from __future__ import annotations

from chronohorn.db import ChronohornDB


def _seed_branched(db):
    """Seed with results that have branch-like naming."""
    configs = [
        ("v12-base-seed42", 1.85, "polyhash_v12"),
        ("v12-base-seed43", 1.87, "polyhash_v12"),
        ("v12-base-seed44", 1.86, "polyhash_v12"),
        ("v12-pkm-seed42", 1.63, "polyhash_v12"),
        ("v12-pkm-seed43", 1.65, "polyhash_v12"),
        ("v8-full-seed42", 1.90, "polyhash_v8"),
    ]
    for name, bpb, arch in configs:
        db.record_result(name, {
            "model": {"test_bpb": bpb, "params": 11000000, "architecture": arch},
            "config": {
                "train": {"steps": 10000, "seq_len": 512, "learning_rate": 0.005},
                "architecture": {"hidden_dim": 320, "num_layers": 6},
            },
            "training": {
                "performance": {"tokens_per_second": 350000, "elapsed_sec": 900},
                "probes": [
                    {"step": 1000, "bpb": bpb + 0.5},
                    {"step": 5000, "bpb": bpb + 0.2},
                    {"step": 10000, "bpb": bpb},
                ],
            },
        })


def test_journal_write_and_read(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_journal("observation", "scan half-life is 8.5 tokens", tags=["scan", "geometry"])
    db.record_journal("decision", "skip rotation scan", tags=["scan"])
    entries = db.journal_entries(limit=10)
    assert len(entries) == 2
    db.close()


def test_journal_filter_by_kind(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    db.record_journal("observation", "note 1")
    db.record_journal("decision", "note 2")
    entries = db.journal_entries(kind="decision", limit=10)
    assert len(entries) == 1
    db.close()


def test_config_diff(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    diff = db.config_diff("v12-base-seed42", "v12-pkm-seed42")
    assert isinstance(diff, dict)
    db.close()


def test_what_varied(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    result = db.what_varied(["v12-base-seed42", "v12-base-seed43", "v12-base-seed44"])
    assert isinstance(result, dict)
    db.close()


def test_branch_health(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    health = db.branch_health("v12-base")
    assert isinstance(health, dict)
    db.close()


def test_frontier_velocity(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    velocity = db.frontier_velocity()
    assert isinstance(velocity, dict)
    db.close()


def test_predict_at_steps(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    pred = db.predict_at_steps("v12-base-seed42", 50000)
    assert isinstance(pred, dict)
    db.close()


def test_find_similar(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    similar = db.find_similar({"hidden_dim": 320, "num_layers": 6}, threshold=0.5)
    assert isinstance(similar, list)
    db.close()


def test_seed_groups(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    result = db.seed_groups()
    assert isinstance(result, (dict, list))
    db.close()


def test_detect_groups(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    _seed_branched(db)
    groups = db.detect_groups()
    assert isinstance(groups, (dict, list))
    db.close()


def test_ablation_board_prefers_context_screen_after_scale_screen(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    base_cfg = {
        "family": "causal-bank",
        "variant": "base",
        "linear_readout_kind": "routed_sqrelu_experts",
        "scale": 8.0,
        "seq_len": 256,
        "profile": "pilot",
    }
    db.record_job(
        "cb-ablate-s8",
        manifest="ablation.jsonl",
        family="causal-bank",
        config=base_cfg,
        steps=4000,
        seed=42,
        batch_size=8,
        job_spec={"work_tokens": 200_000_000},
    )
    db.record_result(
        "cb-ablate-s8",
        {
            "model": {"test_bpb": 1.91, "params": 7_000_000, "architecture": "causal-bank"},
            "config": {"train": {"steps": 4000, "seq_len": 256, "scale": 8.0, "profile": "pilot"}},
            "training": {
                "performance": {"tokens_per_second": 10_000, "elapsed_sec": 100.0},
                "probes": [
                    {"step": 250, "bpb": 2.40, "tflops": 0.05, "elapsed_sec": 10.0},
                    {"step": 500, "bpb": 2.20, "tflops": 0.10, "elapsed_sec": 20.0},
                    {"step": 1000, "bpb": 2.00, "tflops": 0.25, "elapsed_sec": 40.0},
                    {"step": 4000, "bpb": 1.91, "tflops": 1.00, "elapsed_sec": 100.0},
                ],
            },
        },
    )
    db.record_job(
        "cb-ablate-s12",
        manifest="ablation.jsonl",
        family="causal-bank",
        config={**base_cfg, "scale": 12.0},
        steps=4000,
        seed=42,
        batch_size=8,
        job_spec={"work_tokens": 200_000_000},
    )
    db.record_result(
        "cb-ablate-s12",
        {
            "model": {"test_bpb": 1.84, "params": 12_000_000, "architecture": "causal-bank"},
            "config": {"train": {"steps": 4000, "seq_len": 256, "scale": 12.0, "profile": "pilot"}},
            "training": {
                "performance": {"tokens_per_second": 8_000, "elapsed_sec": 120.0},
                "probes": [
                    {"step": 250, "bpb": 2.32, "tflops": 0.08, "elapsed_sec": 12.0},
                    {"step": 500, "bpb": 2.12, "tflops": 0.16, "elapsed_sec": 24.0},
                    {"step": 1000, "bpb": 1.95, "tflops": 0.35, "elapsed_sec": 48.0},
                    {"step": 4000, "bpb": 1.84, "tflops": 1.20, "elapsed_sec": 120.0},
                ],
            },
        },
    )

    board = db.ablation_board(population="controlled", legality="legal", trust="all")
    assert board
    assert board[0]["next_action"] == "test_longer_context"
    assert board[0]["tested_scales"] == [8.0, 12.0]
    assert board[0]["tested_seq_lens"] == [256]
    assert board[0]["trajectory_direction"] in {"improving", "slowing", "accelerating"}
    db.close()


def test_ablation_board_blocks_overbudget_rows_from_promotion(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    cfg = {
        "family": "causal-bank",
        "variant": "base",
        "scale": 12.0,
        "seq_len": 512,
        "profile": "pilot",
    }
    db.record_job(
        "cb-overbudget-s12",
        manifest="ablation.jsonl",
        family="causal-bank",
        config=cfg,
        steps=4000,
        seed=42,
        batch_size=8,
        job_spec={"work_tokens": 200_000_000},
    )
    db.record_result(
        "cb-overbudget-s12",
        {
            "model": {"test_bpb": 1.84, "params": 30_000_000, "architecture": "causal-bank"},
            "config": {"train": {"steps": 4000, "seq_len": 512, "scale": 12.0, "profile": "pilot"}},
            "training": {
                "performance": {"tokens_per_second": 8_000, "elapsed_sec": 120.0},
                "probes": [
                    {"step": 250, "bpb": 2.32, "tflops": 0.08, "elapsed_sec": 12.0},
                    {"step": 500, "bpb": 2.12, "tflops": 0.16, "elapsed_sec": 24.0},
                    {"step": 1000, "bpb": 1.95, "tflops": 0.35, "elapsed_sec": 48.0},
                    {"step": 4000, "bpb": 1.84, "tflops": 1.20, "elapsed_sec": 120.0},
                ],
            },
        },
    )

    board = db.ablation_board(population="controlled", legality="legal", trust="all")
    row = next(item for item in board if item["name"] == "cb-overbudget-s12")
    assert row["artifact_budget_ok"] is False
    assert row["scaling_viable"] is False
    assert row["constant_state_inference"] is True
    assert row["scale_survived"] is True
    assert row["context_survived"] is True
    assert row["next_action"] == "shrink_under_budget"
    assert "artifact_budget" in row["gates_remaining"]
    db.close()


def test_ablation_board_blocks_overcompute_rows_from_promotion(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")
    cfg = {
        "family": "causal-bank",
        "variant": "base",
        "scale": 12.0,
        "seq_len": 512,
        "profile": "full",
    }
    db.record_job(
        "cb-overcompute-s12",
        manifest="ablation.jsonl",
        family="causal-bank",
        config=cfg,
        steps=4000,
        seed=42,
        batch_size=8,
        job_spec={"work_tokens": 2_000_000_000},
    )
    db.record_result(
        "cb-overcompute-s12",
        {
            "model": {"test_bpb": 1.84, "params": 12_000_000, "architecture": "causal-bank"},
            "config": {"train": {"steps": 4000, "seq_len": 512, "scale": 12.0, "profile": "full"}},
            "training": {
                "performance": {
                    "tokens_per_second": 8_000,
                    "elapsed_sec": 120.0,
                    "estimated_sustained_tflops": 100_000.0,
                },
                "probes": [
                    {"step": 250, "bpb": 2.32, "tflops": 0.08, "elapsed_sec": 12.0},
                    {"step": 500, "bpb": 2.12, "tflops": 0.16, "elapsed_sec": 24.0},
                    {"step": 1000, "bpb": 1.95, "tflops": 0.35, "elapsed_sec": 48.0},
                    {"step": 4000, "bpb": 1.84, "tflops": 1.20, "elapsed_sec": 120.0},
                ],
            },
        },
    )

    board = db.ablation_board(population="controlled", legality="legal", trust="all")
    row = next(item for item in board if item["name"] == "cb-overcompute-s12")
    assert row["artifact_budget_ok"] is True
    assert row["compute_budget_ok"] is False
    assert row["scaling_viable"] is False
    assert row["next_action"] == "reduce_compute"
    assert "compute_budget" in row["gates_remaining"]
    assert row["compute_budget_fraction"] > 1.0
    db.close()


def test_mutation_leaderboard_summarizes_matched_base_deltas(tmp_path):
    db = ChronohornDB(tmp_path / "test.db")

    def _record(name: str, *, scale: float, bpb: float, tok_s: float, extra_cfg: dict | None = None) -> None:
        cfg = {
            "family": "causal-bank",
            "scale": scale,
            "seq_len": 256,
            "profile": "pilot",
            **(extra_cfg or {}),
        }
        db.record_job(
            name,
            manifest="mutation.jsonl",
            family="causal-bank",
            config=cfg,
            steps=4000,
            seed=42,
            batch_size=8,
            job_spec={"work_tokens": 200_000_000},
        )
        db.record_result(
            name,
            {
                "model": {"test_bpb": bpb, "params": 12_000_000, "architecture": "causal-bank"},
                "config": {"train": {"steps": 4000, "seq_len": 256, "scale": scale, "profile": "pilot", **(extra_cfg or {})}},
                "training": {
                    "performance": {"tokens_per_second": tok_s, "elapsed_sec": 120.0},
                    "probes": [
                        {"step": 250, "bpb": bpb + 0.32, "tflops": 0.08, "elapsed_sec": 12.0},
                        {"step": 500, "bpb": bpb + 0.20, "tflops": 0.16, "elapsed_sec": 24.0},
                        {"step": 1000, "bpb": bpb + 0.10, "tflops": 0.35, "elapsed_sec": 48.0},
                        {"step": 4000, "bpb": bpb, "tflops": 1.20, "elapsed_sec": 120.0},
                    ],
                },
            },
        )

    _record("cb-base-s8", scale=8.0, bpb=1.90, tok_s=10_000)
    _record("cb-base-s12", scale=12.0, bpb=1.85, tok_s=8_000)
    _record("cb-bands4-s8", scale=8.0, bpb=1.86, tok_s=11_000, extra_cfg={"readout_bands": 4})
    _record("cb-bands4-s12", scale=12.0, bpb=1.82, tok_s=8_800, extra_cfg={"readout_bands": 4})

    leaderboard = db.mutation_leaderboard(population="controlled", legality="legal", trust="all")
    mutation = next(row for row in leaderboard if row["mutation_label"] == "readout_bands=4")

    assert mutation["best_name"] == "cb-bands4-s12"
    assert mutation["matched_base_lane_count"] == 2
    assert mutation["median_bpb_delta_vs_base"] == -0.035
    assert mutation["best_bpb_delta_vs_base"] == -0.04
    assert mutation["median_speed_ratio_vs_base"] == 1.1
    assert mutation["lane_count"] == 2
    assert mutation["run_count"] == 2
    assert mutation["next_action"] == "test_longer_context"
    db.close()
