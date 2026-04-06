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
