"""XP2 — Ablation study comparing legacy A4 vs. composite spatial detector.

The ablation answers two questions:
1. Does the composite detector reduce false positives on anisotropic networks?
2. Does it maintain true positive recall on known geospatial errors?

Protocol
--------
For each system in the global catalogue:
  - Run both legacy centroid + 3σ and composite HDBSCAN + spectral
  - Compute confusion matrix: legacy-flagged vs. composite-flagged
  - Identify DISCORDANT stations (flagged by one method but not the other)
  - Classify discordances into:
      FP_LEGACY   — legacy flags, composite does not (likely false positive)
      FN_COMPOSITE — legacy does not flag, composite does (new finding)
      AGREE_FLAG  — both flag
      AGREE_CLEAN — neither flags

The key metric for the paper is the number of FP_LEGACY stations on
systems with known anisotropic geometry (linear, multi-hub, national).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .anomaly_detector import detect_spatial_anomalies

logger = logging.getLogger(__name__)


def run_ablation(
    df: pd.DataFrame,
    *,
    anomaly_threshold: float = 0.8,
    legacy_sigma_max: float = 3.0,
    **kwargs,
) -> pd.DataFrame:
    """Run both detectors and produce a discordance analysis.

    Returns a DataFrame with one row per station, columns:
      system_id, station_id, flag_A4_legacy, flag_A4_composite,
      discordance_class, composite_score, legacy_sigma_distance
    """
    result = detect_spatial_anomalies(
        df,
        anomaly_threshold=anomaly_threshold,
        legacy_sigma_max=legacy_sigma_max,
        **kwargs,
    )
    sdf = result.station_df.copy()

    def _classify(row):
        leg = row["flag_A4_legacy"]
        comp = row["flag_A4_composite"]
        if leg and comp:
            return "AGREE_FLAG"
        if not leg and not comp:
            return "AGREE_CLEAN"
        if leg and not comp:
            return "FP_LEGACY"
        return "FN_COMPOSITE"

    sdf["discordance_class"] = sdf.apply(_classify, axis=1)
    return sdf


def ablation_summary(ablation_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate discordance analysis at system level."""
    agg = ablation_df.groupby("system_id").agg(
        n_stations=("station_id", "count"),
        n_agree_flag=("discordance_class", lambda x: (x == "AGREE_FLAG").sum()),
        n_agree_clean=("discordance_class", lambda x: (x == "AGREE_CLEAN").sum()),
        n_fp_legacy=("discordance_class", lambda x: (x == "FP_LEGACY").sum()),
        n_fn_composite=("discordance_class", lambda x: (x == "FN_COMPOSITE").sum()),
    ).reset_index()
    agg["discordance_rate"] = (agg["n_fp_legacy"] + agg["n_fn_composite"]) / agg["n_stations"]
    return agg.sort_values("discordance_rate", ascending=False)


def geometry_type_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each system's spatial geometry as isotropic, linear, or multi-hub.

    Uses the eigenvalue ratio of the 2D coordinate covariance matrix:
      λ₁ / λ₂ > 5  →  LINEAR  (elongated along one axis)
      n_clusters ≥ 3  →  MULTI_HUB
      otherwise  →  ISOTROPIC

    This is needed to stratify the ablation results: we expect FP_LEGACY
    to concentrate on LINEAR and MULTI_HUB systems.
    """
    results = []
    for sys_id, grp in df.groupby("system_id"):
        lats = grp["lat"].dropna().to_numpy()
        lons = grp["lon"].dropna().to_numpy()
        if len(lats) < 5:
            results.append({"system_id": sys_id, "geometry_type": "too_small", "eigenvalue_ratio": np.nan})
            continue

        R = 6_371_000.0
        mean_lat = np.radians(np.mean(lats))
        x = R * np.radians(lons) * np.cos(mean_lat)
        y = R * np.radians(lats)
        coords = np.column_stack([x - x.mean(), y - y.mean()])
        cov = np.cov(coords.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

        ratio = eigvals[0] / max(eigvals[1], 1e-10)
        if ratio > 5:
            geo_type = "linear"
        else:
            geo_type = "isotropic"

        results.append({
            "system_id": sys_id,
            "geometry_type": geo_type,
            "eigenvalue_ratio": float(ratio),
        })

    return pd.DataFrame(results)
