"""Run all experiments end-to-end.

Usage
-----
# Full suite (XP2 + XP3 on the static catalogue; XP1 requires prior collection)
python -m experiments.run_all --catalogue catalogue/stations_gold_standard_final.parquet \
                              --output results/

# Individual experiments via their own runners:
#   python -m experiments.xp1_dynamic_audit.run_xp1 ...
#   python -m experiments.xp2_spatial_topology.run_xp2 ...
#   python -m experiments.xp3_looo_validation.run_xp3 ...
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all GBFS audit experiments")
    parser.add_argument("--catalogue", required=True, help="Path to certified parquet")
    parser.add_argument("--output", required=True, help="Root output directory")
    parser.add_argument("--skip-xp1", action="store_true",
                        help="Skip XP1 (requires pre-collected station_status data)")
    parser.add_argument("--xp1-snapshots", help="Path to XP1 snapshot directory (if running XP1 detect)")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_parquet(args.catalogue)
    logger.info("Loaded catalogue: %d stations, %d systems", len(df), df["system_id"].nunique())

    # ── XP2: Spatial Topology ──
    logger.info("=" * 60)
    logger.info("XP2 — Spatial Topology V2")
    logger.info("=" * 60)
    from experiments.xp2_spatial_topology.anomaly_detector import detect_spatial_anomalies
    from experiments.xp2_spatial_topology.ablation import run_ablation, ablation_summary, geometry_type_heuristic

    xp2_dir = out / "xp2"
    xp2_dir.mkdir(exist_ok=True)

    result = detect_spatial_anomalies(df)
    result.station_df.to_parquet(xp2_dir / "xp2_station_scores.parquet", index=False)
    result.system_summary.to_csv(xp2_dir / "xp2_system_summary.csv", index=False)

    ablation_df = run_ablation(df)
    ablation_df.to_parquet(xp2_dir / "xp2_ablation.parquet", index=False)
    abl_sum = ablation_summary(ablation_df)
    abl_sum.to_csv(xp2_dir / "xp2_ablation_summary.csv", index=False)

    geo_df = geometry_type_heuristic(df)
    geo_df.to_csv(xp2_dir / "xp2_geometry_types.csv", index=False)
    logger.info("XP2 complete")

    # ── XP3: LOOO Validation ──
    logger.info("=" * 60)
    logger.info("XP3 — Leave-One-Operator-Out Validation")
    logger.info("=" * 60)
    from audit_pipeline.core import enrich
    from experiments.xp3_looo_validation.protocol import run_looo_full, summarise_looo

    xp3_dir = out / "xp3"
    xp3_dir.mkdir(exist_ok=True)

    df_enriched = enrich(df) if "operator_name" not in df.columns else df
    fold_results = run_looo_full(df_enriched)

    if fold_results:
        summary = summarise_looo(fold_results)
        summary.summary_df.to_csv(xp3_dir / "xp3_looo_per_operator.csv", index=False)
        import json
        report = {
            "n_folds": len(fold_results),
            "per_rule_cv": summary.per_rule_cv,
            "bootstrap_ci": summary.bootstrap_ci,
        }
        with open(xp3_dir / "xp3_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("XP3 complete: %d folds", len(fold_results))
    else:
        logger.warning("XP3: no eligible operators found")

    # ── XP1: Note ──
    if not args.skip_xp1:
        if args.xp1_snapshots:
            logger.info("=" * 60)
            logger.info("XP1 — Dynamic Audit (detect phase)")
            logger.info("=" * 60)
            from experiments.xp1_dynamic_audit.detector import classify_stations, load_snapshots, compute_system_health

            xp1_dir = out / "xp1"
            xp1_dir.mkdir(exist_ok=True)

            snapshots = load_snapshots(Path(args.xp1_snapshots))
            classified = classify_stations(snapshots, df)
            classified.to_parquet(xp1_dir / "xp1_liveness.parquet", index=False)
            health = compute_system_health(classified)
            health.to_csv(xp1_dir / "xp1_system_health.csv", index=False)
            logger.info("XP1 complete")
        else:
            logger.info(
                "XP1 skipped: requires --xp1-snapshots (run collector first). "
                "See: python -m experiments.xp1_dynamic_audit.run_xp1 collect --help"
            )

    logger.info("All experiments complete. Results in %s", out)


if __name__ == "__main__":
    main()
