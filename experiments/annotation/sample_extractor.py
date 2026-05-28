"""Extract a stratified sample for human annotation.

Fixes applied after protocol audit (2026-05-27):
- Added A5 stratum (out-of-perimeter, flag_A5=True)
- Inter-stratum deduplication (a station appears in one stratum only)
- Output shuffled (eliminates order/anchoring effects)
- Empty strata logged with a warning instead of silently skipped

Usage:
    python -m experiments.annotation.sample_extractor \
        --catalogue catalogue/stations_gold_standard_final.parquet \
        --ablation results/xp2/xp2_ablation.parquet \
        --output experiments/annotation/sample_200.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd


# Sample sizes revised after methodological review (2026-05-28):
# A4 discordant strata bumped to n=50 (core contribution of the paper,
# tightest CI where it matters). Other physically-validatable strata
# increased for power. A2 and A7 are pool-capped (40 and 52 stations
# respectively, single operator each) and validated at the system level
# rather than per station, so they stay small by design.
STRATA = {
    "clean_docked":          {"n": 45, "desc": "No flag, high confidence, dock-based"},
    "A1_carsharing":         {"n": 30, "desc": "flag_A1 = True"},
    "A2_placeholder":        {"n": 20, "desc": "flag_A2 = True (system-level, pool-capped)"},
    "A3_freefloating":       {"n": 45, "desc": "flag_A3 = True, flag_A2 = False"},
    "A4_agree_flag":         {"n": 30, "desc": "Both detectors flag"},
    "A4_discordant_legacy":  {"n": 50, "desc": "Legacy centroid flags, composite does not"},
    "A4_discordant_composite": {"n": 50, "desc": "Composite flags, legacy does not"},
    "A5_out_of_perimeter":   {"n": 30, "desc": "flag_A5 = True"},
    "A6_zero_capacity":      {"n": 10, "desc": "flag_A6 = True (empty in v1.0 snapshot)"},
    "A7_null_capacity":      {"n": 25, "desc": "flag_A7 = True, flag_A3 = False (system-level, pool-capped)"},
    "A3_boundary":           {"n": 15, "desc": "Capacity ratio in [2, 5] (empty in v1.0 snapshot)"},
}

SEED = 42


def extract_sample(
    catalogue: pd.DataFrame,
    ablation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rng_state = SEED
    samples: list[pd.DataFrame] = []
    seen_keys: set[str] = set()
    empty_strata: list[str] = []

    def _key(row: pd.Series) -> str:
        return f"{row['system_id']}|{row['station_id']}"

    def _dedup(pool: pd.DataFrame) -> pd.DataFrame:
        pool = pool.copy()
        pool["_key"] = pool.apply(_key, axis=1)
        return pool[~pool["_key"].isin(seen_keys)]

    def _sample(pool: pd.DataFrame, n: int, stratum: str) -> pd.DataFrame:
        pool = _dedup(pool)
        if len(pool) == 0:
            empty_strata.append(stratum)
            warnings.warn(
                f"Stratum '{stratum}': 0 eligible stations after "
                f"deduplication (target n={n}). Skipped.",
                stacklevel=2,
            )
            return pd.DataFrame()
        k = min(n, len(pool))
        s = pool.sample(k, random_state=rng_state)
        s = s.copy()
        s["stratum"] = stratum
        seen_keys.update(s.apply(_key, axis=1))
        return s

    cat = catalogue.copy()

    # --- Clean dock-based (negative control) ---
    clean = cat[
        (cat["station_type"] == "docked_bike")
        & (cat["audit_confidence"] == "high")
    ]
    samples.append(_sample(clean, 45, "clean_docked"))

    # --- A1 ---
    a1 = cat[cat["flag_A1"] == True]  # noqa: E712
    samples.append(_sample(a1, 30, "A1_carsharing"))

    # --- A2 ---
    a2 = cat[cat["flag_A2"] == True]  # noqa: E712
    samples.append(_sample(a2, 20, "A2_placeholder"))

    # --- A3 ---
    a3 = cat[(cat["flag_A3"] == True) & (cat["flag_A2"] == False)]  # noqa: E712
    samples.append(_sample(a3, 45, "A3_freefloating"))

    # --- A4 strata (requires ablation data) ---
    if ablation is not None:
        agree_flag = ablation[ablation["discordance_class"] == "AGREE_FLAG"]
        disc_legacy = ablation[ablation["discordance_class"] == "FP_LEGACY"]
        disc_composite = ablation[ablation["discordance_class"] == "FN_COMPOSITE"]

        meta_cols = [
            "system_id", "station_id", "lat", "lon", "station_type",
            "capacity", "operator_name", "city", "audit_confidence",
        ]
        flag_cols = [f"flag_A{i}" for i in range(1, 8) if f"flag_A{i}" in cat.columns]
        join_cols = [c for c in meta_cols + flag_cols if c in cat.columns]

        for pool, n, name in [
            (agree_flag, 30, "A4_agree_flag"),
            (disc_legacy, 50, "A4_discordant_legacy"),
            (disc_composite, 50, "A4_discordant_composite"),
        ]:
            if len(pool) > 0:
                merged = pool.merge(
                    cat[join_cols], on=["system_id", "station_id"], how="left",
                )
                samples.append(_sample(merged, n, name))

    # --- A5 (out-of-perimeter) ---
    if "flag_A5" in cat.columns:
        a5 = cat[cat["flag_A5"] == True]  # noqa: E712
        samples.append(_sample(a5, 30, "A5_out_of_perimeter"))

    # --- A6 ---
    a6 = cat[cat["flag_A6"] == True]  # noqa: E712
    samples.append(_sample(a6, 10, "A6_zero_capacity"))

    # --- A7 ---
    a7 = cat[(cat["flag_A7"] == True) & (cat["flag_A3"] == False)]  # noqa: E712
    samples.append(_sample(a7, 25, "A7_null_capacity"))

    # --- A3 boundary (capacity ratio 2–5) ---
    docked_with_cap = cat[
        (cat["station_type"] == "docked_bike")
        & (cat["capacity"].notna())
        & (cat["capacity"] > 0)
    ]
    if len(docked_with_cap) > 0:
        sys_stats = docked_with_cap.groupby("system_id")["capacity"].agg(
            ["mean", "median"],
        )
        sys_stats["ratio"] = sys_stats["mean"] / sys_stats["median"].clip(lower=0.01)
        boundary_systems = sys_stats[
            (sys_stats["ratio"] >= 2) & (sys_stats["ratio"] <= 5)
        ].index
        boundary = cat[cat["system_id"].isin(boundary_systems)]
        samples.append(_sample(boundary, 15, "A3_boundary"))

    # --- Assemble, deduplicate key column, shuffle ---
    valid = [s for s in samples if len(s) > 0]
    if not valid:
        raise RuntimeError("No stations extracted — check catalogue and ablation inputs.")
    result = pd.concat(valid, ignore_index=True)
    result.drop(columns=["_key"], errors="ignore", inplace=True)

    keep_cols = [
        "stratum", "system_id", "station_id", "lat", "lon",
        "station_type", "capacity", "operator_name", "city",
        "flag_A1", "flag_A2", "flag_A3", "flag_A4", "flag_A5",
        "flag_A6", "flag_A7", "audit_confidence",
    ]
    keep_cols = [c for c in keep_cols if c in result.columns]

    result["Q1_is_bikeshare"] = ""
    result["Q2_capacity_physical"] = ""
    result["Q3_exists_at_coords"] = ""
    result["Q4_within_perimeter"] = ""
    result["Q5_verdict"] = ""
    result["annotator"] = ""
    result["notes"] = ""

    out = result[keep_cols + [
        "Q1_is_bikeshare", "Q2_capacity_physical", "Q3_exists_at_coords",
        "Q4_within_perimeter", "Q5_verdict", "annotator", "notes",
    ]]

    # Shuffle to eliminate order effects (anchoring, fatigue, learning)
    out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if empty_strata:
        print(f"WARNING: {len(empty_strata)} empty strata: {empty_strata}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cat = pd.read_parquet(args.catalogue)
    abl = pd.read_parquet(args.ablation) if args.ablation else None

    sample = extract_sample(cat, abl)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)

    print(f"Extracted {len(sample)} stations across {sample['stratum'].nunique()} strata:")
    print(sample["stratum"].value_counts().to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
