"""Compute inter-rater reliability and precision/recall from annotation labels.

Methodological design (post-review 2026-05-28)
----------------------------------------------
The human annotation is **strictly pipeline-agnostic**: annotators judge the
real-world state of each station (Q1–Q4 factual observations + a holistic
"is this a real, correctly-described station?" verdict), never whether "the
pipeline is right".  Precision / recall / F1 for each rule A1–A7 are derived
*a posteriori* by joining the adjudicated factual answers with the pipeline
flags.  This removes the circularity of asking a blind annotator to label a
"pipeline false positive".

Per-rule ground-truth predicates (imagery-validatable rules)
------------------------------------------------------------
    A1 (out-of-domain)      anomaly real  <=>  Q1 (is bikeshare?)   == "non"
    A3 (free-floating)      anomaly real  <=>  Q3 (exists/physical) == "non"
    A4 (geospatial outlier) anomaly real  <=>  Q4 (within perimeter)== "non"
    A5 (out-of-perimeter)   anomaly real  <=>  Q4 (within perimeter)== "non"
    A6 (zero-capacity dock) anomaly real  <=>  Q3 (exists/physical) == "non"

A2 (placeholder capacity) and A7 (null capacity field) are **structural,
system-level** properties that cannot be validated from single-station
imagery, so they are reported separately (operator/system level) rather than
with per-station precision/recall.

Usage:
    python -m experiments.annotation.compute_reliability \
        --labels1 experiments/annotation/labels_rohan.csv \
        --labels2 experiments/annotation/labels_gael.csv \
        --output experiments/annotation/reliability_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_LABELS_DIR = Path(__file__).resolve().parent

# Questions adjudicated per station (legacy Q1–Q5 schema).
QUESTIONS = {
    "Q1_is_bikeshare": "Q1 (is bikeshare?)",
    "Q2_capacity_physical": "Q2 (capacity physical?)",
    "Q3_exists_at_coords": "Q3 (exists at coords?)",
    "Q4_within_perimeter": "Q4 (within perimeter?)",
    "Q5_verdict": "Q5 (holistic verdict)",
}

# Rules validatable from single-station imagery, with the factual question and
# the value that means "the flagged anomaly is real".
RULE_PREDICATES = {
    "A1": ("Q1_is_bikeshare", "non"),
    "A3": ("Q3_exists_at_coords", "non"),
    "A4": ("Q4_within_perimeter", "non"),
    "A5": ("Q4_within_perimeter", "non"),
    "A6": ("Q3_exists_at_coords", "non"),
}
# Structural, system-level rules — not single-station imagery-validatable.
SYSTEM_LEVEL_RULES = ["A2", "A7"]

# Values treated as "no usable label" for the binary per-rule derivation.
_UNAVAILABLE = {"indéterminé", "indetermine", "nan", "", "needs_adjudication"}


# =====================================================================
# Reliability metrics
# =====================================================================

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


def krippendorff_alpha_nominal(pairs: list[tuple[str, str]]) -> float:
    """Krippendorff's alpha (nominal metric) for two coders.

    ``pairs`` is a list of (rating_a1, rating_a2) for each co-rated unit.
    Naturally handles >2 categories; missing units are simply omitted by
    the caller.  Returns NaN if fewer than 2 usable units.
    """
    units = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(units) < 2:
        return float("nan")

    values = sorted({v for pair in units for v in pair})
    # Coincidence matrix (each unit has 2 ratings -> weight 1/(2-1) = 1).
    o: dict[tuple[str, str], float] = defaultdict(float)
    for a, b in units:
        o[(a, b)] += 1.0
        o[(b, a)] += 1.0
    n_c = {c: sum(o[(c, k)] for k in values) for c in values}
    n = sum(n_c.values())
    if n <= 1:
        return float("nan")

    do = sum(o[(c, k)] for c in values for k in values if c != k)
    de = sum(n_c[c] * n_c[k] for c in values for k in values if c != k) / (n - 1)
    if de == 0:
        return 1.0
    return float(1 - do / de)


def wilson_ci(
    successes: int, total: int, confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return float(p), float(max(0, center - half)), float(min(1, center + half))


# =====================================================================
# Merge + agreement
# =====================================================================

def merge_annotations(
    labels1: pd.DataFrame, labels2: pd.DataFrame,
) -> pd.DataFrame:
    """Merge two annotator label files on (system_id, station_id, stratum)."""
    l1 = labels1[labels1["Q5_verdict"] != "skipped"].copy()
    l2 = labels2[labels2["Q5_verdict"] != "skipped"].copy()
    return l1.merge(
        l2, on=["system_id", "station_id", "stratum"], suffixes=("_a1", "_a2"),
    )


def compute_agreement(merged: pd.DataFrame) -> dict:
    """Cohen's kappa + Krippendorff's alpha + raw agreement per question."""
    out = {}
    for q_col, q_name in QUESTIONS.items():
        c1, c2 = f"{q_col}_a1", f"{q_col}_a2"
        if c1 not in merged.columns or c2 not in merged.columns:
            continue
        valid = merged[[c1, c2]].dropna()
        if len(valid) == 0:
            continue
        y1 = valid[c1].astype(str).values
        y2 = valid[c2].astype(str).values
        out[q_name] = {
            "cohens_kappa": round(cohens_kappa(y1, y2), 3),
            "krippendorff_alpha": round(
                krippendorff_alpha_nominal(list(zip(y1, y2))), 3,
            ),
            "n": len(valid),
            "raw_agreement": round(float((valid[c1] == valid[c2]).mean()), 3),
        }
    return out


# =====================================================================
# Per-question adjudication
# =====================================================================

def adjudicate_questions(merged: pd.DataFrame) -> pd.DataFrame:
    """Consensus adjudication for each of Q1–Q5.

    For each question, the gold value is the consensus when both annotators
    agree, else ``None`` (disagreement → would go to a third adjudicator).
    """
    rows = []
    for _, r in merged.iterrows():
        rec = {
            "system_id": r["system_id"],
            "station_id": r["station_id"],
            "stratum": r["stratum"],
        }
        for q_col in QUESTIONS:
            a1 = str(r.get(f"{q_col}_a1", "")).strip()
            a2 = str(r.get(f"{q_col}_a2", "")).strip()
            rec[f"gold_{q_col}"] = a1 if a1 == a2 else None
            rec[f"disagree_{q_col}"] = a1 != a2
        rows.append(rec)
    return pd.DataFrame(rows)


def _usable(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() not in _UNAVAILABLE


# =====================================================================
# Per-rule precision / recall (derived analytically)
# =====================================================================

def compute_precision_recall(gold: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Per-rule precision/recall derived from factual gold answers.

    For each imagery-validatable rule, "anomaly is real" is read from the
    relevant adjudicated factual question — never from a pipeline-referencing
    verdict.  A2/A7 are reported at the system level instead.
    """
    flag_cols = [f"flag_A{i}" for i in range(1, 8) if f"flag_A{i}" in sample.columns]
    merged = gold.merge(
        sample[["system_id", "station_id", "stratum"] + flag_cols],
        on=["system_id", "station_id", "stratum"], how="left",
    )

    results: dict = {}

    for rule, (q_col, anomaly_value) in RULE_PREDICATES.items():
        flag_col = f"flag_{rule}"
        gold_col = f"gold_{q_col}"
        if flag_col not in merged.columns or gold_col not in merged.columns:
            continue

        # Only stations with a usable (consensus, non-indeterminate) answer.
        usable = merged[merged[gold_col].apply(_usable)].copy()
        n_excluded = len(merged) - len(usable)
        if len(usable) == 0:
            results[rule] = {"note": "no usable adjudicated answers", "n_excluded": n_excluded}
            continue

        flagged = usable[flag_col].astype(bool)
        anomaly_real = usable[gold_col].astype(str).str.strip() == anomaly_value

        tp = int((flagged & anomaly_real).sum())
        fp = int((flagged & ~anomaly_real).sum())
        fn = int((~flagged & anomaly_real).sum())
        tn = int((~flagged & ~anomaly_real).sum())

        prec, prec_lo, prec_hi = wilson_ci(tp, tp + fp)
        rec, rec_lo, rec_hi = wilson_ci(tp, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        results[rule] = {
            "validation": "per-station (imagery)",
            "ground_truth_question": q_col,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_excluded_unusable": int(n_excluded),
            "precision": round(prec, 3),
            "precision_ci": [round(prec_lo, 3), round(prec_hi, 3)],
            "recall": round(rec, 3),
            "recall_ci": [round(rec_lo, 3), round(rec_hi, 3)],
            "f1": round(f1, 3),
        }

    # A2 / A7 : structural, system-level — count flagged systems in the sample.
    for rule in SYSTEM_LEVEL_RULES:
        flag_col = f"flag_{rule}"
        if flag_col not in merged.columns:
            continue
        flagged_rows = merged[merged[flag_col].astype(bool)]
        results[rule] = {
            "validation": "system-level (not imagery-validatable)",
            "n_flagged_stations_in_sample": int(len(flagged_rows)),
            "n_flagged_systems": int(flagged_rows["system_id"].nunique()),
            "note": (
                "Structural data-property rule; precision is 1 by "
                "construction (the rule reads the data directly). Human "
                "imagery cannot refute it — report the count of affected "
                "systems and the semantic interpretation in the manuscript."
            ),
        }

    # A4 ablation: false-positive rate of the legacy detector on its
    # discordant stratum, derived from the geospatial ground truth (Q4).
    disc = merged[merged["stratum"] == "A4_discordant_legacy"]
    disc = disc[disc["gold_Q4_within_perimeter"].apply(_usable)]
    if len(disc) > 0:
        q4 = disc["gold_Q4_within_perimeter"].astype(str).str.strip()
        # Legacy flags these; a true FP is one the human says is in-perimeter.
        n_fp = int((q4 == "oui").sum())
        n = len(disc)
        rate, lo, hi = wilson_ci(n_fp, n)
        results["A4_discordant_legacy_fp_rate"] = {
            "n_confirmed_fp": n_fp,
            "n_usable": n,
            "fp_rate": round(rate, 3),
            "wilson_ci": [round(lo, 3), round(hi, 3)],
            "interpretation": (
                "Share of legacy-only flags the human judges in-perimeter "
                "(true false positives of the centroid detector)."
            ),
        }

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels1", required=True)
    parser.add_argument("--labels2", required=True)
    parser.add_argument("--sample", default=str(_LABELS_DIR / "sample.csv"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    l1 = pd.read_csv(args.labels1)
    l2 = pd.read_csv(args.labels2)
    sample = pd.read_csv(args.sample)

    merged = merge_annotations(l1, l2)
    agreement = compute_agreement(merged)
    gold = adjudicate_questions(merged)
    metrics = compute_precision_recall(gold, sample)

    # Holistic verdict adjudication summary + indeterminate rate.
    gold_v = gold["gold_Q5_verdict"]
    n_consensus = int(gold_v.notna().sum())
    n_disagree = int(gold_v.isna().sum())
    n_indet = int(
        gold_v.dropna().astype(str).str.strip().str.lower().isin(
            {"indéterminé", "indetermine"}
        ).sum()
    )
    indet_by_stratum = {}
    for stratum, grp in gold.groupby("stratum"):
        g = grp["gold_Q5_verdict"].dropna().astype(str).str.strip().str.lower()
        n_i = int(g.isin({"indéterminé", "indetermine"}).sum())
        if n_i > 0:
            indet_by_stratum[stratum] = {
                "n_indeterminate": n_i, "n_total": int(len(grp)),
                "rate": round(n_i / len(grp), 3),
            }

    report = {
        "annotator_1": args.labels1,
        "annotator_2": args.labels2,
        "n_merged": int(len(merged)),
        "verdict_consensus": n_consensus,
        "verdict_disagreement": n_disagree,
        "n_indeterminate_verdict": n_indet,
        "indeterminate_by_stratum": indet_by_stratum,
        "agreement": agreement,
        "precision_recall": metrics,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    gold.to_csv(out.with_name("gold_labels.csv"), index=False)

    print(f"Inter-rater reliability ({len(merged)} co-rated stations):")
    for q, info in agreement.items():
        print(
            f"  {q}: kappa={info['cohens_kappa']:.3f} "
            f"alpha={info['krippendorff_alpha']:.3f} "
            f"agree={info['raw_agreement']:.0%} (n={info['n']})"
        )
    print(f"\nVerdict: {n_consensus} consensus, {n_disagree} disagree, "
          f"{n_indet} indeterminate")
    print("\nPer-rule precision/recall:")
    for rule in [f"A{i}" for i in range(1, 8)]:
        info = metrics.get(rule)
        if not info:
            continue
        if "precision" in info:
            print(f"  {rule}: P={info['precision']:.2f} R={info['recall']:.2f} "
                  f"F1={info['f1']:.2f} (TP={info['tp']} FP={info['fp']} FN={info['fn']})")
        else:
            print(f"  {rule}: {info.get('validation', 'n/a')}")
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
