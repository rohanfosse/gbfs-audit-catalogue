"""XP3 — End-to-end runner for the LOOO cross-validation experiment.

Usage
-----
python -m experiments.xp3_looo_validation.run_xp3 \
    --catalogue catalogue/stations_gold_standard_final.parquet \
    --output results/xp3/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_pipeline.core import enrich
from .protocol import RULE_COLS, run_looo_full, summarise_looo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _plot_looo(summary, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Per-rule coefficient of variation
    rules = list(summary.per_rule_cv.keys())
    cvs = [summary.per_rule_cv[r] for r in rules]
    colors = ["#2ecc71" if cv < 0.20 else "#e74c3c" for cv in cvs]
    axes[0, 0].bar(rules, cvs, color=colors)
    axes[0, 0].axhline(0.20, color="#333", linestyle="--", linewidth=0.8, label="CV = 0.20 threshold")
    axes[0, 0].set_ylabel("Coefficient of variation")
    axes[0, 0].set_title("Per-rule flag-rate CV across LOOO folds")
    axes[0, 0].legend(fontsize=8)

    # Panel B: Bootstrap CI per rule
    for i, rule in enumerate(rules):
        ci = summary.bootstrap_ci[rule]
        axes[0, 1].errorbar(
            i, ci["mean"],
            yerr=[[ci["mean"] - ci["ci_lo"]], [ci["ci_hi"] - ci["mean"]]],
            fmt="o", color="#1A6FBF", capsize=4,
        )
    axes[0, 1].set_xticks(range(len(rules)))
    axes[0, 1].set_xticklabels(rules)
    axes[0, 1].set_ylabel("Flag rate (95% bootstrap CI)")
    axes[0, 1].set_title("Per-rule flag rate with confidence intervals")

    # Panel C: Heatmap of per-operator flag rates
    sdf = summary.summary_df
    rate_cols = [f"{r}_rate_test" for r in RULE_COLS]
    if len(sdf) > 0 and all(c in sdf.columns for c in rate_cols):
        matrix = sdf[rate_cols].to_numpy()
        im = axes[1, 0].imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        axes[1, 0].set_xticks(range(len(RULE_COLS)))
        axes[1, 0].set_xticklabels(RULE_COLS, fontsize=8)
        axes[1, 0].set_yticks(range(len(sdf)))
        axes[1, 0].set_yticklabels(sdf["operator"].values, fontsize=7)
        axes[1, 0].set_title("Flag rate per operator (test fold)")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel D: Rate ratio distribution (test/train) — box plot
    ratio_cols = [f"{r}_rate_ratio" for r in RULE_COLS]
    if len(sdf) > 0 and all(c in sdf.columns for c in ratio_cols):
        ratio_data = [sdf[c].replace([np.inf, -np.inf], np.nan).dropna().values for c in ratio_cols]
        axes[1, 1].boxplot(ratio_data, labels=RULE_COLS, vert=True)
        axes[1, 1].axhline(1.0, color="#2ecc71", linestyle="--", linewidth=0.8, label="ratio = 1 (no bias)")
        axes[1, 1].set_ylabel("Test/Train flag rate ratio")
        axes[1, 1].set_title("Operator-specific bias (ratio ≈ 1 = agnostic)")
        axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "xp3_looo_overview.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="XP3 — LOOO cross-validation")
    parser.add_argument("--catalogue", required=True, help="Parquet catalogue path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--min-stations", type=int, default=50, help="Min stations per operator")
    parser.add_argument("--min-systems", type=int, default=2, help="Min systems per operator")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.catalogue)
    logger.info("Loaded %d stations", len(df))

    if "operator_name" not in df.columns:
        logger.info("Running enrich() to add operator_name column")
        df = enrich(df)

    fold_results = run_looo_full(
        df,
        min_stations_per_operator=args.min_stations,
        min_systems_per_operator=args.min_systems,
    )

    if not fold_results:
        logger.error("No eligible operators for LOOO. Check min_stations/min_systems.")
        return

    summary = summarise_looo(
        fold_results,
        n_bootstrap=args.bootstrap_n,
        seed=args.bootstrap_seed,
    )

    summary.summary_df.to_csv(out_dir / "xp3_looo_per_operator.csv", index=False)

    report = {
        "n_folds": len(fold_results),
        "operators": [r.held_out_operator for r in fold_results],
        "per_rule_cv": summary.per_rule_cv,
        "bootstrap_ci": summary.bootstrap_ci,
        "verdict": {
            rule: "PASS" if cv < 0.20 else "FAIL"
            for rule, cv in summary.per_rule_cv.items()
        },
    }
    with open(out_dir / "xp3_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    _plot_looo(summary, out_dir)

    logger.info("=== XP3 LOOO Summary ===")
    for rule in RULE_COLS:
        cv = summary.per_rule_cv[rule]
        ci = summary.bootstrap_ci[rule]
        verdict = "PASS" if cv < 0.20 else "FAIL"
        logger.info(
            "  %s: CV=%.3f [%s], flag rate=%.4f [%.4f, %.4f]",
            rule, cv, verdict, ci["mean"], ci["ci_lo"], ci["ci_hi"],
        )


if __name__ == "__main__":
    main()
