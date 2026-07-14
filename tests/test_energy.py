from __future__ import annotations

from pathlib import Path

import pytest

from chronohorn.eval.energy import (
    EnergyReading,
    discover_rapl_domains,
    format_energy_line,
    is_package_domain,
    parse_power_draw_line,
    rapl_delta_uj,
    rapl_package_total,
    rapl_totals,
    trapezoid_integrate_j,
)


def _write_domain(root: Path, id_: str, name: str, energy_uj: int,
                   max_range_uj: int = 1_000_000_000) -> None:
    d = root / id_
    d.mkdir()
    (d / "name").write_text(name + "\n")
    (d / "energy_uj").write_text(str(energy_uj) + "\n")
    (d / "max_energy_range_uj").write_text(str(max_range_uj) + "\n")


# --- wraparound arithmetic --------------------------------------------------

def test_rapl_delta_uj_no_wrap():
    assert rapl_delta_uj(1000, 1500, max_range_uj=1_000_000) == 500


def test_rapl_delta_uj_wraps():
    max_range = 1_000_000
    start = max_range - 100
    end = 50
    # counter climbed the last 100 to the ceiling, wrapped, then climbed 50 more.
    assert rapl_delta_uj(start, end, max_range) == 150


def test_rapl_delta_uj_real_world_range():
    # this machine's actual intel-rapl:0/max_energy_range_uj
    max_range = 262_143_328_850
    start = max_range - 1
    end = 0
    assert rapl_delta_uj(start, end, max_range) == 1


# --- trapezoid integration ---------------------------------------------------

def test_trapezoid_integration_constant_power():
    # 10 W held for 2s -> exactly 20 J; trapezoid is exact for a constant signal.
    ts = [0.0, 1.0, 2.0]
    watts = [10.0, 10.0, 10.0]
    assert trapezoid_integrate_j(ts, watts) == pytest.approx(20.0)


def test_trapezoid_integration_ramp():
    # linear ramp 0 -> 10W over 2s: triangle area = 0.5 * base * height = 10 J.
    ts = [0.0, 1.0, 2.0]
    watts = [0.0, 5.0, 10.0]
    assert trapezoid_integrate_j(ts, watts) == pytest.approx(10.0)


def test_trapezoid_integration_uneven_spacing():
    # 5W for 0.5s then 15W for 1.5s (jittery sampling): 0.5*(5+5)*0.5 wrong -- use
    # actual trapezoid segments: (0,5)->(0.5,5)=2.5J, (0.5,5)->(2.0,15)=0.5*(5+15)*1.5=15J
    ts = [0.0, 0.5, 2.0]
    watts = [5.0, 5.0, 15.0]
    assert trapezoid_integrate_j(ts, watts) == pytest.approx(2.5 + 15.0)


def test_trapezoid_integration_needs_two_points():
    assert trapezoid_integrate_j([0.0], [10.0]) == 0.0
    assert trapezoid_integrate_j([], []) == 0.0


# --- N/A sample parsing -------------------------------------------------------

def test_parse_power_draw_line_valid():
    assert parse_power_draw_line("9.71\n") == pytest.approx(9.71)


def test_parse_power_draw_line_na_variants_skipped():
    assert parse_power_draw_line("N/A") is None
    assert parse_power_draw_line("[N/A]") is None
    assert parse_power_draw_line("") is None
    assert parse_power_draw_line("   \n") is None
    assert parse_power_draw_line("garbage\n") is None


def test_na_lines_counted_not_zeroed_in_stream():
    # simulate what GpuPowerSampler's read loop does per line, without the
    # subprocess/thread machinery: N/A must be counted, never contribute a 0 W sample.
    lines = ["9.7\n", "N/A\n", "9.8\n", "[N/A]\n", "10.1\n"]
    watts, na = [], 0
    for line in lines:
        w = parse_power_draw_line(line)
        if w is None:
            na += 1
        else:
            watts.append(w)
    assert na == 2
    assert watts == [9.7, 9.8, 10.1]


# --- EnergyReading / format_energy_line --------------------------------------

def test_format_energy_line_missing_cpu_prints_literal_missing():
    r = EnergyReading(duration_s=12.34, gpu_j=145.2, gpu_avg_w=11.8, gpu_samples=61,
                       cpu_domains={}, missing=["cpu-rapl: permission denied"])
    line = format_energy_line(r)
    assert "cpu_J=MISSING" in line
    assert "cpu_J=0" not in line
    assert "cpu_J=0.0" not in line


def test_format_energy_line_all_fields_present():
    r = EnergyReading(duration_s=12.34, gpu_j=145.2, gpu_avg_w=11.8, gpu_samples=61,
                       cpu_domains={}, missing=[], baseline_gpu_w=5.0)
    line = format_energy_line(r)
    assert line.startswith("ENERGY: ")
    assert "dur_s=12.34" in line
    assert "gpu_J=145.2" in line
    assert "gpu_avg_W=11.8" in line
    assert "samples=61" in line


def test_energy_reading_cpu_j_none_when_no_domains():
    r = EnergyReading(duration_s=1.0, gpu_j=None, gpu_avg_w=None, gpu_samples=0,
                       cpu_domains={}, missing=[])
    assert r.cpu_j is None


def test_energy_reading_cpu_j_excludes_subdomains_and_psys():
    r = EnergyReading(duration_s=1.0, gpu_j=None, gpu_avg_w=None, gpu_samples=0,
                       cpu_domains={"package-0": 50.0, "package-0/core": 30.0, "psys": 120.0},
                       missing=[])
    assert r.cpu_j == pytest.approx(50.0)


def test_net_gpu_j_is_none_without_baseline():
    r = EnergyReading(duration_s=10.0, gpu_j=200.0, gpu_avg_w=20.0, gpu_samples=50,
                       cpu_domains={}, missing=[])
    assert r.net_gpu_j is None


def test_net_cpu_j_is_none_without_baseline():
    r = EnergyReading(duration_s=10.0, gpu_j=None, gpu_avg_w=None, gpu_samples=0,
                       cpu_domains={"package-0": 80.0}, missing=[])
    assert r.net_cpu_j is None


def test_net_gpu_j_subtracts_baseline():
    r = EnergyReading(duration_s=10.0, gpu_j=200.0, gpu_avg_w=20.0, gpu_samples=50,
                       cpu_domains={}, missing=[], baseline_gpu_w=9.7)
    assert r.net_gpu_j == pytest.approx(200.0 - 97.0)


def test_net_gpu_j_floors_at_zero():
    # baseline*duration (97 J) exceeds raw (50 J) -- must clamp to 0, never negative.
    r = EnergyReading(duration_s=10.0, gpu_j=50.0, gpu_avg_w=5.0, gpu_samples=50,
                       cpu_domains={}, missing=[], baseline_gpu_w=9.7)
    assert r.net_gpu_j == 0.0


def test_net_cpu_j_floors_at_zero():
    r = EnergyReading(duration_s=10.0, gpu_j=None, gpu_avg_w=None, gpu_samples=0,
                       cpu_domains={"package-0": 10.0}, missing=[], baseline_cpu_w=5.0)
    assert r.net_cpu_j == 0.0


def test_j_per_byte_uses_net_when_baseline_present():
    r = EnergyReading(duration_s=10.0, gpu_j=200.0, gpu_avg_w=20.0, gpu_samples=50,
                       cpu_domains={}, missing=[], baseline_gpu_w=9.7)
    line = format_energy_line(r, n_bytes=1000)
    expected = (200.0 - 97.0) / 1000
    assert f"J_per_byte={expected:.3g}" in line


def test_j_per_byte_uses_raw_without_baseline():
    r = EnergyReading(duration_s=10.0, gpu_j=200.0, gpu_avg_w=20.0, gpu_samples=50,
                       cpu_domains={}, missing=[])
    line = format_energy_line(r, n_bytes=1000)
    expected = 200.0 / 1000
    assert f"J_per_byte={expected:.3g}" in line


def test_j_per_byte_omitted_without_n_bytes():
    r = EnergyReading(duration_s=10.0, gpu_j=200.0, gpu_avg_w=20.0, gpu_samples=50,
                       cpu_domains={}, missing=[])
    assert "J_per_byte" not in format_energy_line(r)


# --- RAPL discovery / double-counting protection ------------------------------

def test_discover_rapl_domains_empty_root(tmp_path):
    assert discover_rapl_domains(tmp_path / "does-not-exist") == []


def test_discover_rapl_domains_classifies_package_subdomain_mirror(tmp_path):
    _write_domain(tmp_path, "intel-rapl:0", "package-0", energy_uj=100_000_000)
    _write_domain(tmp_path, "intel-rapl:0:0", "core", energy_uj=40_000_000)
    _write_domain(tmp_path, "intel-rapl-mmio:0", "package-0", energy_uj=100_000_000)

    domains = {d.id: d for d in discover_rapl_domains(tmp_path)}
    assert set(domains) == {"intel-rapl:0", "intel-rapl:0:0", "intel-rapl-mmio:0"}

    pkg = domains["intel-rapl:0"]
    assert pkg.is_subdomain is False and pkg.mirror is False and pkg.readable is True

    core = domains["intel-rapl:0:0"]
    assert core.is_subdomain is True and core.parent_id == "intel-rapl:0" and core.mirror is False

    mirror = domains["intel-rapl-mmio:0"]
    assert mirror.mirror is True


def test_discover_rapl_domains_marks_unreadable_when_energy_uj_absent(tmp_path):
    d = tmp_path / "intel-rapl:0"
    d.mkdir()
    (d / "name").write_text("package-0\n")
    (d / "max_energy_range_uj").write_text("1000000\n")
    # no energy_uj file -- stands in for a permission-denied read. readable requires
    # BOTH energy_uj and max_energy_range_uj; max_range_uj itself is still captured
    # since (unlike energy_uj on this machine) it's typically world-readable metadata.
    domains = discover_rapl_domains(tmp_path)
    assert len(domains) == 1
    assert domains[0].readable is False
    assert domains[0].max_range_uj == 1_000_000


def test_is_package_domain():
    assert is_package_domain("package-0")
    assert is_package_domain("package-1")
    assert not is_package_domain("psys")
    assert not is_package_domain("core")
    assert not is_package_domain("package-0/core")


def test_rapl_double_counting_package_counted_once(tmp_path):
    """The scenario the task exists to prevent: MSR package-0, its core subdomain,
    and the MMIO mirror of the SAME package-0 -- the summed total must count
    package-0's energy exactly once (not doubled by the mirror, not inflated by
    adding the core subdomain on top).
    """
    _write_domain(tmp_path, "intel-rapl:0", "package-0", energy_uj=100_000_000)
    _write_domain(tmp_path, "intel-rapl:0:0", "core", energy_uj=40_000_000)
    _write_domain(tmp_path, "intel-rapl-mmio:0", "package-0", energy_uj=100_000_000)

    domains = discover_rapl_domains(tmp_path)
    assert len(domains) == 3

    # simulate a wraparound-free reading interval: joules == raw uj / 1e6 for each.
    deltas_j = {d.id: d.read_uj() / 1e6 for d in domains}
    by_name = rapl_totals(domains, deltas_j)

    assert rapl_package_total(by_name) == pytest.approx(100.0)   # not 200 (mirror), not 140 (+core)
    assert by_name["package-0"] == pytest.approx(100.0)
    assert by_name["package-0/core"] == pytest.approx(40.0)
    # the mirror contributes nothing under its own key -- it must not appear at all.
    assert len(by_name) == 2


def test_rapl_psys_not_summed_with_package(tmp_path):
    _write_domain(tmp_path, "intel-rapl:0", "package-0", energy_uj=100_000_000)
    _write_domain(tmp_path, "intel-rapl:1", "psys", energy_uj=250_000_000)

    domains = discover_rapl_domains(tmp_path)
    deltas_j = {d.id: d.read_uj() / 1e6 for d in domains}
    by_name = rapl_totals(domains, deltas_j)

    assert rapl_package_total(by_name) == pytest.approx(100.0)   # psys excluded from the total
    assert by_name["psys"] == pytest.approx(250.0)
