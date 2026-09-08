"""Configuration loading and project-root resolution."""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[dict[str, Any], Path]:
    """Load a YAML config and return it together with the project root it belongs to.

    The project root is the directory holding ``configs/``, so relative paths such as
    ``data/``, ``outputs/`` and ``results/`` resolve against the repository root.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    project_root = path.parent.parent if path.parent.name == "configs" else path.parent
    return config, project_root


def resolve_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a config path value against the project root."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path
