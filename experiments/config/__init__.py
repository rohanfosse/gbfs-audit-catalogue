"""Experiment configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULTS = _CONFIG_DIR / "defaults.yaml"


def load_config(override_path: Path | None = None) -> dict[str, Any]:
    """Load experiment configuration with optional overrides.

    Reads defaults.yaml, then deep-merges any override file on top.
    """
    with open(_DEFAULTS, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if override_path is not None:
        with open(override_path, encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        config = _deep_merge(config, overrides)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged
