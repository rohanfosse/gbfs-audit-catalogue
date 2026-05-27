"""Compute inter-rater reliability and precision/recall from annotation labels.

Usage:
    python -m experiments.annotation.compute_reliability \
        --labels1 experiments/annotation/labels_rohan.csv \
        --labels2 experiments/annotation/labels_gael.csv \
        --output experiments/annotation/reliability_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Cohen's kappa for two annotators on categorical labels."""
    classes = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return float("nan")

    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    class_idx = {c: i for i, c in enumerate(classes)}
    for a, b in zip(y1, y2):
        confusion[class_idx[a], class_idx[b]] += 1

    po = np.diag(confusion).sum() / n
    pe = sum(
        (confusion[i, :].sum() / n) * (confusion[:, i].sum() / n)
        for i in range(len(classes))
    )

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1 - pe))


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return float(p), float(max(0, center - half)), float(min(1, center + half))


def merge_annotations(labels1: pd.DataFrame, labels2: pd.DataFrame) -> pd.DataFrame:
    """Merge two annotator label files on (system_id, station_id)."""
    skip_mask_1 = labels1["Q5_verdict"] == "skipped"
    skip_mask_2 = labels2["Q5_verdict"] == "skipped"
    l1 = labels1[~skip_mask_1].copy()
    l2 = labels2[~skip_mask_2].copy()

    merged = l1.merge(
        l2,
        on=["system_id", "station_id", "stratum"],
        suffixes=("_a1", "_a2"),
    )
    return merged


def compute_kappas(merged: pd.DataFrame) -> dict:
    """Compute Cohen's kappa for each question."""
    questions = {
        "Q1_is_bikeshare": "Q1 (is bikeshare?)",
        "Q2_capacity_physical": "Q2 (capacity physical?)",
        "Q3_exists_at_coords": "Q3 (exists at coords?)",
        "Q4_within_perimeter": "Q4 (within perimeter?)",
        "Q5_verdict": "Q5 (overall verdict)",
    }

    kappas = {}
    for q_col, q_name in questions.items():
        col_a1 = f"{q_col}_a1"
        col_a2 = f"{q_col}_a2"
        if col_a1 in merged.columns and col_a2 in merged.columns:
            valid = merged[[col_a1, col_a2]].dropna()
            if len(valid) > 0:
                k = cohens_kappa(valid[col_a1].values, valid[col_a2].values)
                kappas[q_name] = {
                    "kappa": round(k, 3),
                    "n": len(valid),
                    "agreement": round(float((valid[col_a1] == valid[col_a2]).mean()), 3),
                }
    return kappas


def adjudicate(merged: pd.DataFrame) -> pd.DataFrame:
    """Simple majority adjudication: if both agree, use that; else mark for review."""
    gold = []
    for _, row in merged.iterrows():
        verdict_a1 = row.get("Q5_verdict_a1", "")
        verdict_a2 = row.get("Q5_verdict_a2", "")

        if verdict_a1 == verdict_a2:
            gold_verdict = verdict_a1
            needs_adjudication = False
        else:
            gold_verdict = "NEEDS_ADJUDICATION"
            needs_adjudication = True

        gold.append({
            "system_id": row["system_id"],
            "station_id": row["station_id"],
            "stratum": row["stratum"],
            "verdict_a1": verdict_a1,
            "verdict_a2": verdict_a2,
            "gold_verdict": gold_verdict,
            "needs_adjudication": needs_adjudication,
        })

    return pd.DataFrame(gold)


def compute_precision_recall(
    gold: pd.DataFrame,
    sample: pd.DataFrame,
) -> dict:
    """Compute per-rule and A4-detector precision/recall from gold labels."""
    results = {}

    merged = gold.merge(
        sample[["system_id", "station_id", "stratum"] +
               [f"flag_A{i}" for i in range(1, 8) if f"flag_A{i}" in sample.columns]],
        on=["system_id", "station_id", "stratum"],
        how="left",
    )

    adjudicated = merged[merged["gold_verdict"] != "NEEDS_ADJUDICATION"]
    is_anomaly = adjudicated["gold_verdict"] == "anomaly confirmed"
    is_clean = adjudicated["gold_verdict"] == "clean"
    is_fp = adjudicated["gold_verdict"] == "pipeline false positive"

    for i in range(1, 8):
        col = f"flag_A{i}"
        if col not in adjudicated.columns:
            continue
        flagged = adjudicated[col].astype(bool)
        tp = int((flagged & is_anomaly).sum())
        fp = int((flagged & (is_clean | is_fp)).sum())
        fn = int((~flagged & is_anomaly).sum())
        tn = int((~flagged & (is_clean | is_fp)).sum())

        prec_val, prec_lo, prec_hi = wilson_ci(tp, tp + fp)
        rec_val, rec_lo, rec_hi = wilson_ci(tp, tp + fn)
        f1 = 2 * prec_val * rec_val / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0

        results[f"A{i}"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec_val, 3),
            "precision_ci": [round(prec_lo, 3), round(prec_hi, 3)],
            "recall": round(rec_val, 3),
            "recall_ci": [round(rec_lo, 3), round(rec_hi, 3)],
            "f1": round(f1, 3),
        }

    # A4 detector-specific: DISCORDANT_LEGACY stratum
    disc_legacy = adjudicated[adjudicated["stratum"] == "A4_discordant_legacy"]
    if len(disc_legacy) > 0:
        n_true_fp = int((disc_legacy["gold_verdict"].isin(["clean", "pipeline false positive"])).sum())
        n_total = len(disc_legacy)
        rate, ci_lo, ci_hi = wilson_ci(n_true_fp, n_total)
        results["A4_discordant_legacy_fp_rate"] = {
            "n_confirmed_fp": n_true_fp,
            "n_total": n_total,
            "fp_rate": round(rate, 3),
            "wilson_ci": [round(ci_lo, 3), round(ci_hi, 3)],
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels1", required=True)
    parser.add_argument("--labels2", required=True)
    parser.add_argument("--sample", default=str(LABELS_DIR / "sample_200.csv"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    l1 = pd.read_csv(args.labels1)
    l2 = pd.read_csv(args.labels2)
    sample = pd.read_csv(args.sample)

    merged = merge_annotations(l1, l2)
    kappas = compute_kappas(merged)
    gold = adjudicate(merged)
    metrics = compute_precision_recall(gold, sample)

    report = {
        "annotator_1": args.labels1,
        "annotator_2": args.labels2,
        "n_merged": len(merged),
        "n_agreement": int((gold["gold_verdict"] != "NEEDS_ADJUDICATION").sum()),
        "n_needs_adjudication": int((gold["gold_verdict"] == "NEEDS_ADJUDICATION").sum()),
        "kappas": kappas,
        "precision_recall": metrics,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    gold.to_csv(out.with_name("gold_labels.csv"), index=False)

    print(f"Inter-rater reliability ({len(merged)} stations):")
    for q, info in kappas.items():
        print(f"  {q}: kappa={info['kappa']:.3f}, agreement={info['agreement']:.0%}")

    print(f"\nAdjudication: {report['n_agreement']} agree, {report['n_needs_adjudication']} need review")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
