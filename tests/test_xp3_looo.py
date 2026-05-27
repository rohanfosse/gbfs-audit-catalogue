"""Tests for XP3 LOOO cross-validation protocol."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.xp3_looo_validation.protocol import (
    _apply_rules,
    _learn_thresholds,
    run_looo_fold,
    summarise_looo,
)


def _make_multi_operator_catalogue(n_per_op: int = 100) -> pd.DataFrame:
    """Create a synthetic catalogue with 4 operators, each with 2 systems."""
    rng = np.random.default_rng(42)
    rows = []
    operators = ["OpA", "OpB", "OpC", "OpD"]
    for i, op in enumerate(operators):
        for sys_idx in range(2):
            sys_id = f"{op}_sys{sys_idx}"
            center_lat = 45 + i * 2
            center_lon = 2 + sys_idx * 0.5
            for stn_idx in range(n_per_op):
                rows.append({
                    "system_id": sys_id,
                    "station_id": f"{sys_id}_stn{stn_idx}",
                    "station_type": "docked_bike",
                    "capacity": float(rng.integers(10, 30)),
                    "lat": center_lat + rng.normal(0, 0.01),
                    "lon": center_lon + rng.normal(0, 0.01),
                    "operator_name": op,
                    "system_name": sys_id,
                })
    return pd.DataFrame(rows)


class TestApplyRules:
    def test_schema_rules_work_on_synthetic_data(self):
        df = _make_multi_operator_catalogue()
        thresholds = _learn_thresholds(df)
        flagged = _apply_rules(df, thresholds)
        for rule in [f"flag_A{i}" for i in range(1, 8)]:
            assert rule in flagged.columns


class TestLOOOFold:
    def test_fold_runs_without_error(self):
        df = _make_multi_operator_catalogue()
        result = run_looo_fold(df, "OpA", operator_col="operator_name")
        assert result.held_out_operator == "OpA"
        assert result.n_test > 0
        assert result.n_train > 0
        assert result.n_train + result.n_test == len(df)

    def test_train_test_no_overlap(self):
        df = _make_multi_operator_catalogue()
        result = run_looo_fold(df, "OpB", operator_col="operator_name")
        assert result.n_test == len(df[df["operator_name"] == "OpB"])


class TestSummarise:
    def test_cv_is_finite(self):
        df = _make_multi_operator_catalogue()
        results = [
            run_looo_fold(df, op, operator_col="operator_name")
            for op in ["OpA", "OpB", "OpC", "OpD"]
        ]
        summary = summarise_looo(results, n_bootstrap=100)
        for rule, cv in summary.per_rule_cv.items():
            assert np.isfinite(cv), f"CV for {rule} is not finite"
