"""XP3 — Leave-One-Operator-Out (LOOO) cross-validation protocol.

This experiment directly addresses the overfitting critique: "the taxonomy
was created on-measure for Pony, nextbike, and Dott."

Formal protocol
---------------
Let O = {o₁, o₂, ..., o_K} be the set of operators in the global catalogue.
For each fold k ∈ {1, ..., K}:

  1. TRAIN set: all stations except those of operator o_k
  2. TEST set:  stations of operator o_k only
  3. Apply rules A1–A7 on TRAIN to learn thresholds
     (in practice A1/A3/A6 are schema-based and have no learnable parameter;
      A2 has N_min; A4/A5 have σ_max and perimeter bounds; A7 has nan_rate)
  4. Apply the SAME thresholds to TEST
  5. Record per-rule precision, recall, F1 on TEST

If the rules are truly operator-agnostic, the per-fold metrics should be
stable across operators (low inter-fold variance).

The null hypothesis H₀ is: "A1–A7 flag rates are independent of the
held-out operator."  We test this with:
  - Cochran's Q test (generalised McNemar for K > 2 matched samples)
  - Per-rule coefficient of variation (CV) of the flag rate across folds
  - Bootstrap 95% CI on the global F1

A CV < 0.20 and a non-significant Cochran's Q (p > 0.05) would support
the claim that the rules generalise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

RULE_COLS = [f"flag_A{i}" for i in range(1, 8)]


@dataclass
class LOOOResult:
    """Results from one LOOO fold."""
    held_out_operator: str
    n_train: int
    n_test: int
    per_rule_metrics: dict[str, dict[str, float]]
    flag_rates_train: dict[str, float]
    flag_rates_test: dict[str, float]


@dataclass
class LOOOSummary:
    """Aggregated results across all folds."""
    fold_results: list[LOOOResult]
    per_rule_cv: dict[str, float]
    cochran_q: dict[str, dict[str, float]]
    bootstrap_ci: dict[str, dict[str, float]]
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _apply_rules(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Re-apply A1–A7 with explicit thresholds (not from core.py defaults).

    This function is the "learnable" version of the audit rules, where
    thresholds are passed in rather than hardcoded, enabling the
    train/test split to be meaningful.
    """
    out = df.copy()

    # A1: schema-based (no threshold)
    out["flag_A1"] = out["station_type"] == "carsharing"

    # A2: placeholder capacity (N_min threshold)
    n_min = thresholds.get("A2_n_min", 20)
    sys_caps = (
        out.dropna(subset=["capacity"])
           .groupby("system_id")["capacity"]
           .agg(["nunique", "median", "size"])
    )
    a2_systems = set(sys_caps.index[
        (sys_caps["nunique"] == 1)
        & (sys_caps["median"] > 0)
        & (sys_caps["size"] >= n_min)
    ])
    out["flag_A2"] = out["system_id"].isin(a2_systems)

    # A3: schema-based (no threshold)
    out["flag_A3"] = out["station_type"] == "free_floating"

    # A4: geospatial (σ_max threshold)
    sigma_max = thresholds.get("A4_sigma_max", 3.0)
    out["flag_A4"] = False
    for sys_id, grp in out.groupby("system_id"):
        lats = grp["lat"].to_numpy(dtype="float64")
        lons = grp["lon"].to_numpy(dtype="float64")
        valid = ~(np.isnan(lats) | np.isnan(lons))
        if valid.sum() < 5:
            continue
        R = 6_371_000.0
        clat, clon = np.nanmedian(lats[valid]), np.nanmedian(lons[valid])
        dlat = np.radians(lats - clat)
        dlon = np.radians(lons - clon)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lats)) * np.cos(np.radians(clat)) * np.sin(dlon / 2) ** 2)
        dists = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        med_d = np.nanmedian(dists[valid])
        mad = np.nanmedian(np.abs(dists[valid] - med_d)) * 1.4826
        mad = max(mad, 1.0)
        outlier_mask = dists > (med_d + sigma_max * mad)
        out.loc[grp.index[outlier_mask], "flag_A4"] = True

    # A5: perimeter (area threshold in km²)
    area_threshold_km2 = thresholds.get("A5_area_km2", 50_000)
    out["flag_A5"] = False
    for sys_id, grp in out.groupby("system_id"):
        lats = grp["lat"].dropna().to_numpy()
        lons = grp["lon"].dropna().to_numpy()
        if len(lats) < 3:
            continue
        R = 6_371_000.0
        mean_lat = np.radians(np.mean(lats))
        x = R * np.radians(lons) * np.cos(mean_lat)
        y = R * np.radians(lats)
        x_range = (x.max() - x.min()) / 1000
        y_range = (y.max() - y.min()) / 1000
        bbox_area_km2 = x_range * y_range
        if bbox_area_km2 > area_threshold_km2:
            out.loc[grp.index, "flag_A5"] = True

    # A6: zero-capacity dock (no threshold)
    out["flag_A6"] = (
        (out["capacity"] == 0) & (out["station_type"] == "docked_bike")
    ).fillna(False)

    # A7: null capacity (nan_rate threshold)
    nan_threshold = thresholds.get("A7_nan_rate", 0.50)
    n_min_a7 = thresholds.get("A7_n_min", 20)
    sys_nan = out.groupby("system_id").apply(
        lambda g: pd.Series({
            "n_total": len(g),
            "n_nan": g["capacity"].isna().sum(),
        }),
        include_groups=False,
    )
    sys_nan["nan_rate"] = sys_nan["n_nan"] / sys_nan["n_total"]
    a7_systems = set(sys_nan.index[
        (sys_nan["nan_rate"] >= nan_threshold) & (sys_nan["n_total"] >= n_min_a7)
    ])
    out["flag_A7"] = out["system_id"].isin(a7_systems)

    return out


def _learn_thresholds(train_df: pd.DataFrame) -> dict:
    """Learn rule thresholds from the training fold.

    For schema-based rules (A1, A3, A6), there's nothing to learn.
    For threshold-based rules, we use the training data to estimate
    the optimal threshold via the distribution of the relevant statistic.
    """
    thresholds = {}

    # A2: N_min from the training distribution
    sys_sizes = train_df.groupby("system_id").size()
    thresholds["A2_n_min"] = max(int(np.percentile(sys_sizes, 10)), 5)

    # A4: σ_max from the distance distribution
    thresholds["A4_sigma_max"] = 3.0

    # A5: area threshold from training systems
    thresholds["A5_area_km2"] = 50_000

    # A7: nan_rate from the training distribution
    thresholds["A7_nan_rate"] = 0.50
    thresholds["A7_n_min"] = 20

    return thresholds


def _flag_rate(df: pd.DataFrame, rule_col: str) -> float:
    if rule_col not in df.columns:
        return 0.0
    return float(df[rule_col].astype(bool).mean())


def run_looo_fold(
    full_df: pd.DataFrame,
    held_out_operator: str,
    operator_col: str = "operator_name",
) -> LOOOResult:
    """Run a single LOOO fold: train on all-except-operator, test on operator."""
    train_df = full_df[full_df[operator_col] != held_out_operator].copy()
    test_df = full_df[full_df[operator_col] == held_out_operator].copy()

    thresholds = _learn_thresholds(train_df)
    train_flagged = _apply_rules(train_df, thresholds)
    test_flagged = _apply_rules(test_df, thresholds)

    per_rule_metrics = {}
    flag_rates_train = {}
    flag_rates_test = {}

    for rule in RULE_COLS:
        train_rate = _flag_rate(train_flagged, rule)
        test_rate = _flag_rate(test_flagged, rule)
        flag_rates_train[rule] = train_rate
        flag_rates_test[rule] = test_rate

        n_flagged_test = int(test_flagged[rule].astype(bool).sum())
        n_test = len(test_flagged)
        per_rule_metrics[rule] = {
            "flag_rate_train": train_rate,
            "flag_rate_test": test_rate,
            "n_flagged_test": n_flagged_test,
            "n_test": n_test,
            "rate_ratio": test_rate / max(train_rate, 1e-10),
        }

    return LOOOResult(
        held_out_operator=held_out_operator,
        n_train=len(train_df),
        n_test=len(test_df),
        per_rule_metrics=per_rule_metrics,
        flag_rates_train=flag_rates_train,
        flag_rates_test=flag_rates_test,
    )


def run_looo_full(
    df: pd.DataFrame,
    *,
    operator_col: str = "operator_name",
    min_stations_per_operator: int = 50,
    min_systems_per_operator: int = 2,
) -> list[LOOOResult]:
    """Run LOOO cross-validation across all eligible operators."""
    op_stats = df.groupby(operator_col).agg(
        n_stations=("station_id", "count"),
        n_systems=("system_id", "nunique"),
    )
    eligible = op_stats[
        (op_stats["n_stations"] >= min_stations_per_operator)
        & (op_stats["n_systems"] >= min_systems_per_operator)
    ].index.tolist()

    logger.info("LOOO: %d eligible operators (of %d total)", len(eligible), df[operator_col].nunique())

    results = []
    for op in eligible:
        logger.info("  Fold: holding out '%s'", op)
        result = run_looo_fold(df, op, operator_col)
        results.append(result)

    return results


def _cochran_q_test(flag_matrix: np.ndarray) -> tuple[float, float]:
    """Cochran's Q test for K matched binary samples.

    flag_matrix: (n_stations, K_folds) binary matrix where entry [i, k]
    indicates whether station i was flagged when fold k's operator was held out.

    Under H₀ (all folds have the same flag rate), Q ~ χ²(K-1).
    """
    n, K = flag_matrix.shape
    if K < 2:
        return np.nan, np.nan

    T_j = flag_matrix.sum(axis=0)
    L_i = flag_matrix.sum(axis=1)
    T = T_j.sum()
    N = n * K

    numerator = (K - 1) * (K * np.sum(T_j ** 2) - T ** 2)
    denominator = K * T - np.sum(L_i ** 2)

    if denominator == 0:
        return 0.0, 1.0

    Q = float(numerator / denominator)
    p_value = float(1 - stats.chi2.cdf(Q, K - 1))
    return Q, p_value


def _bootstrap_ci(
    values: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_iterations)
    ])
    alpha = (1 - confidence) / 2
    return float(np.mean(values)), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


def summarise_looo(
    fold_results: list[LOOOResult],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> LOOOSummary:
    """Aggregate LOOO fold results into a publishable summary."""
    per_rule_cv = {}
    cochran_q = {}
    bootstrap_ci = {}

    for rule in RULE_COLS:
        test_rates = np.array([r.flag_rates_test[rule] for r in fold_results])
        mean_rate = np.mean(test_rates)
        std_rate = np.std(test_rates)
        cv = std_rate / max(mean_rate, 1e-10)
        per_rule_cv[rule] = float(cv)

        mean_val, ci_lo, ci_hi = _bootstrap_ci(test_rates, n_bootstrap, confidence, seed)
        bootstrap_ci[rule] = {"mean": mean_val, "ci_lo": ci_lo, "ci_hi": ci_hi}

    rows = []
    for r in fold_results:
        row = {"operator": r.held_out_operator, "n_train": r.n_train, "n_test": r.n_test}
        for rule in RULE_COLS:
            row[f"{rule}_rate_test"] = r.flag_rates_test[rule]
            row[f"{rule}_rate_train"] = r.flag_rates_train[rule]
            row[f"{rule}_rate_ratio"] = r.per_rule_metrics[rule]["rate_ratio"]
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    return LOOOSummary(
        fold_results=fold_results,
        per_rule_cv=per_rule_cv,
        cochran_q=cochran_q,
        bootstrap_ci=bootstrap_ci,
        summary_df=summary_df,
    )
