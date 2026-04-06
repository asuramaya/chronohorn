"""Tests for fleet/k8s.py — manifest building, tolerations, runtime class, naming."""
from __future__ import annotations


def test_build_job_manifest_basic():
    from chronohorn.fleet.k8s import build_job_manifest

    spec = {
        "name": "test-job",
        "image": "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
        "command": "echo hello",
        "gpu": True,
        "host": "slop-01",
    }
    manifest = build_job_manifest(spec)

    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"

    pod_spec = manifest["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    assert container["image"] == spec["image"]
    assert "bash" in container["command"][0]


def test_build_job_manifest_gpu_resources():
    from chronohorn.fleet.k8s import build_job_manifest

    spec = {
        "name": "gpu-test",
        "image": "pytorch/pytorch:latest",
        "command": "nvidia-smi",
        "gpu": True,
    }
    manifest = build_job_manifest(spec)
    container = manifest["spec"]["template"]["spec"]["containers"][0]

    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"


def test_build_job_manifest_no_gpu():
    from chronohorn.fleet.k8s import build_job_manifest

    spec = {
        "name": "cpu-test",
        "image": "python:3.12",
        "command": "echo hi",
        "gpu": False,
    }
    manifest = build_job_manifest(spec)
    container = manifest["spec"]["template"]["spec"]["containers"][0]

    assert container["resources"] == {}


def test_build_job_manifest_has_toleration():
    from chronohorn.fleet.k8s import build_job_manifest

    spec = {
        "name": "tol-test",
        "image": "pytorch/pytorch:latest",
        "command": "echo hi",
        "gpu": True,
    }
    manifest = build_job_manifest(spec)
    tolerations = manifest["spec"]["template"]["spec"]["tolerations"]

    # Must tolerate chronohorn-gpu=reserved:NoSchedule
    found = any(
        t.get("key") == "chronohorn-gpu" and t.get("effect") == "NoSchedule"
        for t in tolerations
    )
    assert found, f"Missing chronohorn-gpu toleration in {tolerations}"


def test_build_job_manifest_has_nvidia_runtime_class():
    from chronohorn.fleet.k8s import build_job_manifest

    spec = {
        "name": "rtc-test",
        "image": "pytorch/pytorch:latest",
        "command": "echo hi",
        "gpu": True,
    }
    manifest = build_job_manifest(spec)
    assert manifest["spec"]["template"]["spec"]["runtimeClassName"] == "nvidia"


def test_build_job_manifest_no_runtime_class_without_gpu():
    from chronohorn.fleet.k8s import build_job_manifest

    spec = {
        "name": "no-rtc",
        "image": "python:3.12",
        "command": "echo hi",
        "gpu": False,
    }
    manifest = build_job_manifest(spec)
    assert "runtimeClassName" not in manifest["spec"]["template"]["spec"]


def test_build_job_manifest_node_selector():
    from chronohorn.fleet.k8s import build_job_manifest

    spec = {
        "name": "node-test",
        "image": "pytorch/pytorch:latest",
        "command": "echo hi",
        "host": "slop-02",
    }
    manifest = build_job_manifest(spec)
    selector = manifest["spec"]["template"]["spec"].get("nodeSelector", {})

    assert selector.get("kubernetes.io/hostname") == "slop-02"


def test_build_job_manifest_empty_name_raises():
    import pytest

    from chronohorn.fleet.k8s import build_job_manifest

    with pytest.raises(ValueError, match="non-empty name"):
        build_job_manifest({"name": "", "image": "x", "command": "y"})


def test_build_job_manifest_empty_command_raises():
    import pytest

    from chronohorn.fleet.k8s import build_job_manifest

    with pytest.raises(ValueError, match="non-empty command"):
        build_job_manifest({"name": "x", "image": "y", "command": ""})


def test_job_name_sanitization():
    from chronohorn.fleet.k8s import default_runtime_job_name

    # K8s job names must be DNS-safe
    name = default_runtime_job_name({"name": "My_Weird.Job Name!!"})
    assert all(c.isalnum() or c in "-" for c in name)
    assert len(name) <= 63  # k8s name limit
