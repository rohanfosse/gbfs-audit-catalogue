"""XP1 — End-to-end runner for the dynamic audit experiment.

Usage
-----
# Phase 1: Collect station_status snapshots for 14 days
python -m experiments.xp1_dynamic_audit.run_xp1 collect \
    --feeds feeds.csv --output data/xp1_snapshots --days 14

# Phase 2: Detect zombie stations
python -m experiments.xp1_dynamic_audit.run_xp1 detect \
    --snapshots data/xp1_snapshots \
    --station-info catalogue/stations_gold_standard_final.parquet \
    --output results/xp1_liveness.parquet

# Phase 3: Generate figures and summary statistics
python -m experiments.xp1_dynamic_audit.run_xp1 report \
    --liveness results/xp1_liveness.parquet \
    --output results/xp1_figures/
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .collector import run_collection_loop
from .detector import classify_stations, compute_system_health, load_snapshots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_collect(args: argparse.Namespace) -> None:
    feeds = pd.read_csv(args.feeds)
    required = {"system_id", "station_status_url"}
    if not required.issubset(feeds.columns):
        sys.exit(f"Feed CSV must contain columns: {required}")
    asyncio.run(
        run_collection_loop(
            feeds,
            Path(args.output),
            interval_minutes=args.interval,
            duration_days=args.days,
            max_concurrent=args.concurrency,
        )
    )


def cmd_detect(args: argparse.Namespace) -> None:
    snapshots = load_snapshots(Path(args.snapshots))
    station_info = pd.read_parquet(args.station_info)
    classified = classify_stations(
        snapshots,
        station_info,
        min_snapshots=args.min_snapshots,
        staleness_threshold_hours=args.staleness_hours,
        entropy_epsilon=args.entropy_epsilon,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    classified.to_parquet(out, index=False)
    logger.info("Wrote %d classified stations to %s", len(classified), out)

    health = compute_system_health(classified)
    health_path = out.with_name(out.stem + "_system_health.csv")
    health.to_csv(health_path, index=False)
    logger.info("Wrote system health to %s", health_path)


def cmd_report(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.liveness)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Liveness distribution
    counts = df["liveness"].value_counts()
    colors = {"live": "#2ecc71", "zombie": "#e74c3c", "intermittent": "#f39c12", "decommissioned": "#95a5a6"}
    axes[0].bar(counts.index, counts.values, color=[colors.get(c, "#333") for c in counts.index])
    axes[0].set_title("Station liveness classification")
    axes[0].set_ylabel("Number of stations")

    # Panel B: Entropy distribution (log scale)
    live_H = df.loc[df["liveness"] == "live", "entropy"].dropna()
    zombie_H = df.loc[df["liveness"] == "zombie", "entropy"].dropna()
    if len(live_H) > 0:
        axes[1].hist(live_H, bins=50, alpha=0.7, label="live", color="#2ecc71")
    if len(zombie_H) > 0:
        axes[1].hist(zombie_H, bins=50, alpha=0.7, label="zombie", color="#e74c3c")
    axes[1].set_xlabel("Normalised Shannon entropy H(s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Entropy separation")
    axes[1].legend()

    # Panel C: Zombie rate per system (top 20)
    health = compute_system_health(df)
    top_zombie = health.nlargest(20, "zombie_rate")
    if len(top_zombie) > 0:
        axes[2].barh(top_zombie["system_id"], top_zombie["zombie_rate"], color="#e74c3c")
        axes[2].set_xlabel("Zombie rate")
        axes[2].set_title("Top-20 systems by zombie rate")

    plt.tight_layout()
    fig.savefig(out_dir / "xp1_liveness_overview.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    summary = {
        "total_stations": len(df),
        "n_live": int((df["liveness"] == "live").sum()),
        "n_zombie": int((df["liveness"] == "zombie").sum()),
        "n_intermittent": int((df["liveness"] == "intermittent").sum()),
        "n_decommissioned": int((df["liveness"] == "decommissioned").sum()),
        "zombie_rate_global": float((df["liveness"] == "zombie").mean()),
        "mean_entropy_live": float(df.loc[df["liveness"] == "live", "entropy"].mean()),
        "mean_entropy_zombie": float(df.loc[df["liveness"] == "zombie", "entropy"].mean()),
    }
    pd.Series(summary).to_json(out_dir / "xp1_summary.json", indent=2)
    logger.info("Report written to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="XP1 — Dynamic audit experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Collect station_status snapshots")
    p_collect.add_argument("--feeds", required=True, help="CSV with system_id and station_status_url")
    p_collect.add_argument("--output", required=True, help="Output directory for snapshots")
    p_collect.add_argument("--days", type=int, default=14)
    p_collect.add_argument("--interval", type=int, default=15, help="Polling interval in minutes")
    p_collect.add_argument("--concurrency", type=int, default=50)

    p_detect = sub.add_parser("detect", help="Classify stations by liveness")
    p_detect.add_argument("--snapshots", required=True, help="Directory of collected snapshots")
    p_detect.add_argument("--station-info", required=True, help="Parquet of station_information")
    p_detect.add_argument("--output", required=True, help="Output parquet path")
    p_detect.add_argument("--min-snapshots", type=int, default=100)
    p_detect.add_argument("--staleness-hours", type=float, default=72.0)
    p_detect.add_argument("--entropy-epsilon", type=float, default=0.01)

    p_report = sub.add_parser("report", help="Generate figures and summary")
    p_report.add_argument("--liveness", required=True, help="Parquet from detect phase")
    p_report.add_argument("--output", required=True, help="Output directory for figures")

    args = parser.parse_args()
    {"collect": cmd_collect, "detect": cmd_detect, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
