"""Config loading and project-root resolution."""

import pytest

from xray_fl.config import load_config, resolve_path


def test_load_config_returns_the_project_root_above_configs(tmp_path):
    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs" / "config.yaml"
    config_path.write_text("seed: 7\npaths:\n  results_dir: results\n", encoding="utf-8")

    config, project_root = load_config(config_path)

    assert config["seed"] == 7
    assert project_root == tmp_path


def test_resolve_path_keeps_absolute_paths_and_anchors_relative_ones(tmp_path):
    absolute = tmp_path / "elsewhere"

    assert resolve_path(tmp_path, absolute) == absolute
    assert resolve_path(tmp_path, "results") == tmp_path / "results"


def test_load_config_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
