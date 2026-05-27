"""XP2 — End-to-end runner for the spatial topology experiment.

Usage
-----
# Full pipeline: detect + ablation + figures
python -m experiments.xp2_spatial_topology.run_xp2 \
    --catalogue catalogue/stations_gold_standard_final.parquet \
    --output results/xp2/

# Ablation only (if detection already ran)
python -m experiments.xp2_spatial_topology.run_xp2 \
    --catalogue catalogue/stations_gold_standard_final.parquet \
    --output results/xp2/ --ablation-only
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ablation import ablation_summary, geometry_type_heuristic, run_ablation
from .anomaly_detector import detect_spatial_anomalies

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _plot_ablation(ablation_df: pd.DataFrame, geo_df: pd.DataFrame, out_dir: Path) -> None:
    merged = ablation_df.merge(geo_df, on="system_id", how="left")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Discordance class distribution
    counts = merged["discordance_class"].value_counts()
    colors_map = {
        "AGREE_CLEAN": "#2ecc71", "AGREE_FLAG": "#e67e22",
        "FP_LEGACY": "#e74c3c", "FN_COMPOSITE": "#3498db",
    }
    axes[0].bar(counts.index, counts.values,
                color=[colors_map.get(c, "#999") for c in counts.index])
    axes[0].set_title("Discordance classification (all stations)")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    # Panel B: FP_LEGACY by geometry type
    fp = merged[merged["discordance_class"] == "FP_LEGACY"]
    if len(fp) > 0:
        fp_by_geo = fp["geometry_type"].value_counts()
        axes[1].bar(fp_by_geo.index, fp_by_geo.values, color="#e74c3c")
    axes[1].set_title("Legacy false positives by geometry type")
    axes[1].set_ylabel("Stations flagged by legacy only")

    # Panel C: Composite vs. legacy sigma-distance scatter
    sample = merged.dropna(subset=["composite_score", "legacy_sigma_distance"])
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    if len(sample) > 0:
        c = sample["discordance_class"].map(colors_map).fillna("#999")
        axes[2].scatter(sample["legacy_sigma_distance"], sample["composite_score"],
                        c=c, alpha=0.3, s=5)
        axes[2].axhline(0.8, color="#333", linestyle="--", linewidth=0.5, label="composite threshold")
        axes[2].axvline(3.0, color="#333", linestyle=":", linewidth=0.5, label="legacy 3σ threshold")
        axes[2].set_xlabel("Legacy σ-distance from centroid")
        axes[2].set_ylabel("Composite anomaly score")
        axes[2].set_title("Legacy vs. composite scoring")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "xp2_ablation_overview.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="XP2 — Spatial topology experiment")
    parser.add_argument("--catalogue", required=True, help="Parquet catalogue")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5, help="HDBSCAN/spectral mixing weight")
    parser.add_argument("--anomaly-threshold", type=float, default=0.8)
    parser.add_argument("--legacy-sigma-max", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.catalogue)

    logger.info("Loaded %d stations across %d systems", len(df), df["system_id"].nunique())

    if not args.ablation_only:
        result = detect_spatial_anomalies(
            df,
            alpha=args.alpha,
            anomaly_threshold=args.anomaly_threshold,
            legacy_sigma_max=args.legacy_sigma_max,
        )
        result.station_df.to_parquet(out_dir / "xp2_station_scores.parquet", index=False)
        result.system_summary.to_csv(out_dir / "xp2_system_summary.csv", index=False)
        logger.info("Detection complete; wrote station scores and system summary")

    # Ablation
    ablation_df = run_ablation(
        df,
        anomaly_threshold=args.anomaly_threshold,
        legacy_sigma_max=args.legacy_sigma_max,
        alpha=args.alpha,
    )
    ablation_df.to_parquet(out_dir / "xp2_ablation.parquet", index=False)

    abl_summary = ablation_summary(ablation_df)
    abl_summary.to_csv(out_dir / "xp2_ablation_summary.csv", index=False)

    geo_df = geometry_type_heuristic(df)
    geo_df.to_csv(out_dir / "xp2_geometry_types.csv", index=False)

    _plot_ablation(ablation_df, geo_df, out_dir)

    n_fp = int((ablation_df["discordance_class"] == "FP_LEGACY").sum())
    n_fn = int((ablation_df["discordance_class"] == "FN_COMPOSITE").sum())
    n_agree = int((ablation_df["discordance_class"].isin(["AGREE_FLAG", "AGREE_CLEAN"])).sum())
    logger.info(
        "Ablation: %d FP_LEGACY, %d FN_COMPOSITE, %d agreement (of %d total)",
        n_fp, n_fn, n_agree, len(ablation_df),
    )


if __name__ == "__main__":
    main()
