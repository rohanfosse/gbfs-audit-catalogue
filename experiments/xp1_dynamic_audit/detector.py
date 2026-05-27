"""XP1 — Zombie station detector.

Given a time-series of station_status snapshots collected by ``collector.py``,
this module classifies each station into one of four liveness categories:

  LIVE           — significant temporal variance in availability vector
  ZOMBIE         — zero variance AND stale last_reported
  INTERMITTENT   — near-zero variance (> 90% frozen snapshots)
  DECOMMISSIONED — station absent from station_status but present in
                   station_information (phantom entry)

Mathematical foundation
-----------------------
For a station s observed over T snapshots, define:

  x_t = num_bikes_available(t) + num_docks_available(t)

The **normalised Shannon entropy** of the discretised availability series is:

  H(s) = -Σ_k  p_k · log₂(p_k) / log₂(K)

where p_k is the empirical frequency of level k ∈ {0, 1, ..., capacity}.
H(s) ∈ [0, 1]:  0 = perfectly frozen,  1 = uniform distribution.

A station is classified ZOMBIE if:
  1. H(s) < ε   (entropy_epsilon, default 0.01)
  2. max(last_reported) is stale by > τ hours  (staleness_threshold_hours)

A station is classified INTERMITTENT if:
  1. The fraction of snapshots where x_t == x_{t-1} exceeds 0.90
  2. But H(s) ≥ ε  (some residual variance, e.g. from rare restocking)

This approach is robust to:
- Overnight dips (legitimate zero-availability at 3 AM)
- Maintenance windows (short frozen periods don't accumulate enough)
- Seasonal effects (14-day window captures weekday/weekend cycles)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StationLiveness(str, Enum):
    LIVE = "live"
    ZOMBIE = "zombie"
    INTERMITTENT = "intermittent"
    DECOMMISSIONED = "decommissioned"


def load_snapshots(snapshot_dir: Path) -> pd.DataFrame:
    """Load all Parquet snapshots from a date-partitioned directory."""
    files = sorted(snapshot_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {snapshot_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    logger.info("Loaded %d snapshots covering %d unique epochs", len(df), df["collected_at"].nunique())
    return df


def _normalised_entropy(series: np.ndarray) -> float:
    """Normalised Shannon entropy of a discrete integer series."""
    vals, counts = np.unique(series[~np.isnan(series)], return_counts=True)
    if len(vals) <= 1:
        return 0.0
    K = len(vals)
    probs = counts / counts.sum()
    H = -np.sum(probs * np.log2(probs))
    return float(H / np.log2(K))


def _frozen_fraction(series: np.ndarray) -> float:
    """Fraction of consecutive snapshots where the value is unchanged."""
    clean = series[~np.isnan(series)]
    if len(clean) < 2:
        return 1.0
    diffs = np.diff(clean)
    return float(np.sum(diffs == 0) / len(diffs))


def classify_stations(
    snapshots: pd.DataFrame,
    station_info: pd.DataFrame,
    *,
    min_snapshots: int = 100,
    staleness_threshold_hours: float = 72.0,
    entropy_epsilon: float = 0.01,
    intermittent_threshold: float = 0.90,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """Classify every station into a liveness category.

    Parameters
    ----------
    snapshots : DataFrame
        Output of ``load_snapshots``: must contain system_id, station_id,
        num_bikes_available, num_docks_available, last_reported, collected_at.
    station_info : DataFrame
        The station_information catalogue (system_id, station_id at minimum).
    min_snapshots : int
        Discard stations observed fewer than this many times.
    staleness_threshold_hours : float
        A station's last_reported must be this stale to be classified ZOMBIE.
    entropy_epsilon : float
        Normalised entropy below this value = frozen.
    intermittent_threshold : float
        Fraction of unchanged consecutive snapshots above which = INTERMITTENT.
    reference_time : datetime or None
        "Now" for staleness computation; defaults to UTC now.

    Returns
    -------
    DataFrame
        One row per (system_id, station_id) with columns:
        system_id, station_id, n_snapshots, entropy, frozen_frac,
        max_last_reported, liveness, liveness_reason.
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    snapshots = snapshots.copy()
    snapshots["total_available"] = (
        snapshots["num_bikes_available"].fillna(0)
        + snapshots["num_docks_available"].fillna(0)
    )

    results = []
    grouped = snapshots.groupby(["system_id", "station_id"])

    for (sys_id, stn_id), grp in grouped:
        n = len(grp)
        if n < min_snapshots:
            continue

        series = grp.sort_values("collected_at")["total_available"].to_numpy(dtype="float64")
        H = _normalised_entropy(series)
        ff = _frozen_fraction(series)

        last_reported_raw = grp["last_reported"].dropna()
        if len(last_reported_raw) > 0:
            max_lr = pd.to_datetime(last_reported_raw.max(), unit="s", utc=True)
        else:
            max_lr = pd.NaT

        staleness_hours = np.nan
        if pd.notna(max_lr):
            staleness_hours = (reference_time - max_lr).total_seconds() / 3600

        if H < entropy_epsilon and staleness_hours > staleness_threshold_hours:
            liveness = StationLiveness.ZOMBIE
            reason = f"H={H:.4f} < {entropy_epsilon}, stale {staleness_hours:.0f}h"
        elif ff > intermittent_threshold and H >= entropy_epsilon:
            liveness = StationLiveness.INTERMITTENT
            reason = f"frozen_frac={ff:.2f} > {intermittent_threshold}, H={H:.4f}"
        else:
            liveness = StationLiveness.LIVE
            reason = f"H={H:.4f}, frozen_frac={ff:.2f}"

        results.append({
            "system_id": sys_id,
            "station_id": stn_id,
            "n_snapshots": n,
            "entropy": H,
            "frozen_frac": ff,
            "max_last_reported": max_lr,
            "staleness_hours": staleness_hours,
            "liveness": liveness.value,
            "liveness_reason": reason,
        })

    result_df = pd.DataFrame(results)

    info_keys = station_info[["system_id", "station_id"]].drop_duplicates()
    if len(result_df) > 0:
        status_keys = result_df[["system_id", "station_id"]].drop_duplicates()
        phantom = info_keys.merge(status_keys, how="left", indicator=True)
        phantom = phantom[phantom["_merge"] == "left_only"].drop(columns="_merge")
    else:
        phantom = info_keys.copy()

    if len(phantom) > 0:
        phantom_rows = []
        for _, row in phantom.iterrows():
            phantom_rows.append({
                "system_id": row["system_id"],
                "station_id": row["station_id"],
                "n_snapshots": 0,
                "entropy": np.nan,
                "frozen_frac": np.nan,
                "max_last_reported": pd.NaT,
                "staleness_hours": np.nan,
                "liveness": StationLiveness.DECOMMISSIONED.value,
                "liveness_reason": "present in station_information but absent from station_status",
            })
        result_df = pd.concat([result_df, pd.DataFrame(phantom_rows)], ignore_index=True)

    logger.info(
        "Classification: %d live, %d zombie, %d intermittent, %d decommissioned",
        (result_df["liveness"] == "live").sum(),
        (result_df["liveness"] == "zombie").sum(),
        (result_df["liveness"] == "intermittent").sum(),
        (result_df["liveness"] == "decommissioned").sum(),
    )
    return result_df


def compute_system_health(classified: pd.DataFrame) -> pd.DataFrame:
    """Aggregate liveness classification to system-level health metrics.

    Returns one row per system_id with:
    - n_stations, n_live, n_zombie, n_intermittent, n_decommissioned
    - zombie_rate, ghost_rate (decommissioned / total in station_information)
    - mean_entropy, median_staleness_hours
    """
    agg = classified.groupby("system_id").agg(
        n_stations=("station_id", "count"),
        n_live=("liveness", lambda x: (x == "live").sum()),
        n_zombie=("liveness", lambda x: (x == "zombie").sum()),
        n_intermittent=("liveness", lambda x: (x == "intermittent").sum()),
        n_decommissioned=("liveness", lambda x: (x == "decommissioned").sum()),
        mean_entropy=("entropy", "mean"),
        median_staleness_hours=("staleness_hours", "median"),
    ).reset_index()

    agg["zombie_rate"] = agg["n_zombie"] / agg["n_stations"]
    agg["ghost_rate"] = agg["n_decommissioned"] / agg["n_stations"]
    return agg
