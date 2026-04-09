"""Tests for security hardening: shell injection, path traversal, env key validation."""
from __future__ import annotations

import pytest


class TestEnvKeyValidation:
    def test_valid_keys(self):
        from chronohorn.fleet.dispatch import _validate_env_key

        for key in ["PATH", "PYTHONPATH", "MY_VAR_123", "_private", "A"]:
            assert _validate_env_key(key) == key

    def test_rejects_semicolon_injection(self):
        from chronohorn.fleet.dispatch import _validate_env_key

        with pytest.raises(ValueError, match="Invalid environment variable key"):
            _validate_env_key("foo;rm -rf /")

    def test_rejects_spaces(self):
        from chronohorn.fleet.dispatch import _validate_env_key

        with pytest.raises(ValueError):
            _validate_env_key("BAD KEY")

    def test_rejects_numeric_start(self):
        from chronohorn.fleet.dispatch import _validate_env_key

        with pytest.raises(ValueError):
            _validate_env_key("123start")

    def test_rejects_equals(self):
        from chronohorn.fleet.dispatch import _validate_env_key

        with pytest.raises(ValueError):
            _validate_env_key("a=b")

    def test_rejects_empty(self):
        from chronohorn.fleet.dispatch import _validate_env_key

        with pytest.raises(ValueError):
            _validate_env_key("")


class TestRenderRemoteExports:
    def test_basic_export(self):
        from chronohorn.fleet.dispatch import render_remote_exports

        result = render_remote_exports({"PATH": "/usr/bin", "HOME": "/root"})
        assert "export HOME=" in result
        assert "export PATH=" in result

    def test_rejects_bad_key(self):
        from chronohorn.fleet.dispatch import render_remote_exports

        with pytest.raises(ValueError):
            render_remote_exports({"good": "val", "bad;key": "val"})


class TestFleetPathValidation:
    def test_launch_managed_command_rejects_remote_cwd_escape(self, tmp_path):
        from chronohorn.fleet.dispatch import launch_managed_command

        source_dir = tmp_path / "src"
        source_dir.mkdir()
        with pytest.raises(ValueError, match="remote_cwd_rel"):
            launch_managed_command(
                {
                    "name": "safe-name",
                    "host": "local",
                    "source_dir": str(source_dir),
                    "remote_cwd_rel": "../etc",
                    "command": "echo hi",
                }
            )

    def test_record_remote_run_rejects_name_escape(self):
        from chronohorn.fleet.dispatch import record_remote_run

        with pytest.raises(ValueError, match="job name"):
            record_remote_run({}, "../escape")

    def test_pull_remote_result_rejects_name_escape(self, tmp_path):
        from chronohorn.fleet.results import pull_remote_result

        with pytest.raises(ValueError, match="job name"):
            pull_remote_result(
                host="slop-01",
                remote_run="/tmp/chronohorn-runs/safe",
                job_name="../escape",
                local_out_dir=tmp_path,
            )

    def test_write_launch_record_rejects_name_escape(self):
        from chronohorn.fleet.dispatch import write_launch_record

        with pytest.raises(ValueError, match="job name"):
            write_launch_record("../escape", {"name": "../escape"})

    def test_load_manifest_rejects_name_with_path_separator(self, tmp_path):
        from chronohorn.fleet.dispatch import load_manifest

        manifest = tmp_path / "bad.jsonl"
        manifest.write_text('{"name":"../escape","launcher":"local_command","cwd":".","command":"echo hi"}\n')
        with pytest.raises(ValueError, match="job name"):
            load_manifest(manifest)

    def test_expand_matrix_rejects_name_template_escape(self):
        from chronohorn.fleet.experiment_matrix import expand_matrix

        with pytest.raises(ValueError, match="job name"):
            expand_matrix(
                {
                    "name_template": "../escape-{_index}",
                    "base": {"steps": 1000},
                    "sweep": {"lr": [0.001]},
                }
            )

    def test_matrix_to_commands_quotes_result_path(self):
        from chronohorn.fleet.experiment_matrix import matrix_to_commands

        commands = matrix_to_commands(
            [{"name": "safe name", "steps": 1000}],
            script="scripts/train.py",
        )

        assert commands[0]["command"].endswith("--json '/results/safe name.json'")


class TestMcpRequiredParams:
    def test_required_raises_on_missing(self):
        from chronohorn.mcp import _required

        with pytest.raises(ValueError, match="required parameter 'name' is missing"):
            _required({}, "name")

    def test_required_raises_on_none(self):
        from chronohorn.mcp import _required

        with pytest.raises(ValueError):
            _required({"name": None}, "name")

    def test_required_returns_value(self):
        from chronohorn.mcp import _required

        assert _required({"name": "test"}, "name") == "test"

    def test_required_returns_falsy_non_none(self):
        from chronohorn.mcp import _required

        assert _required({"count": 0}, "count") == 0
        assert _required({"flag": False}, "flag") is False
        assert _required({"text": ""}, "text") == ""


class TestSshQuoting:
    def test_results_ssh_cat_quoted(self):
        """Verify _ssh_cat_file uses shlex.quote on the path."""
        import inspect

        from chronohorn.fleet.results import _ssh_cat_file

        source = inspect.getsource(_ssh_cat_file)
        assert "shlex.quote" in source

    def test_cli_pull_ls_quoted(self):
        """Verify _do_one_pull uses shlex.quote on remote_dir."""
        import inspect

        from chronohorn.fleet.cli import _do_one_pull

        source = inspect.getsource(_do_one_pull)
        assert "shlex.quote" in source
