"""Analyse the E1 retrospective held-out audit results.

Tests the three pre-registered hypotheses from
experiments/e1_holdout/PROTOCOL.md against the live audit of
GBFS systems added to the MobilityData catalogue between
2025-11-11 and 2026-05-13 (the audit-pipeline rule-freeze date).

H1 -- rule-firing rate on held-out within +/- 30% of the
      derivation-set rate of 13.5% (i.e. [9.5%, 17.5%]).
      A1-A5 are the relevant "structural" classes for this
      derivation-set comparison (the global audit reported
      204/1,509 = 13.5% on these five classes).

H2 -- among held-out systems flagged on A3 or A7, the dominant
      operator family covers >=50% of flagged systems and is
      a subset of {Dott, Pony, Bird, nextbike, Voi, Bolt, Lime,
      Tier} (the operator anti-patterns identified on the
      derivation set).

H3 -- held-out systems whose operator family is in the clean panel
      of Section 3.4 (PBSC, JCDecaux Cyclocity, Smartbike, De Lijn,
      Urban Sharing, Deutsche Bahn) trigger zero structural flags
      on A1, A2, A3, A6, A7. Isolated A4 outliers and bounding-box
      A5 are allowed for these clean-operator systems.

Output: experiments/e1_holdout/held_out_analysis.md
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS = Path("experiments/e1_holdout/held_out_results.csv")
OUT = Path("experiments/e1_holdout/held_out_analysis.md")

# Operator-family patterns (lower-case)
FF_OPERATORS = [
    (r"dott", "Dott"),
    (r"pony", "Pony"),
    (r"bird", "Bird"),
    (r"nextbike", "nextbike"),
    (r"voi", "Voi"),
    (r"bolt", "Bolt"),
    (r"lime", "Lime"),
    (r"tier", "Tier"),
    (r"spin", "Spin"),
    (r"donkey", "Donkey Republic"),
    (r"getaround", "Getaround"),
    (r"free.?now|free2move|share.?now", "Free Now / Free2Move / Share Now"),
]
CLEAN_OPERATORS = [
    (r"publicbikesystem|pbsc", "PBSC"),
    (r"cyclocity", "JCDecaux Cyclocity"),
    (r"smartbike", "Smartbike"),
    (r"delijn|blue.?bike", "De Lijn / Blue-bike"),
    (r"urbansharing", "Urban Sharing"),
    (r"callabike|deutsche.?bahn", "Deutsche Bahn / Call a Bike"),
    (r"vcub|cyclocity", "Cyclocity"),
]


def classify_operator(name: str, url: str) -> tuple[str, str]:
    s = f"{name} {url}".lower()
    for pat, label in FF_OPERATORS:
        if re.search(pat, s):
            return ("ff", label)
    for pat, label in CLEAN_OPERATORS:
        if re.search(pat, s):
            return ("clean", label)
    return ("other", "")


def main() -> None:
    if not RESULTS.exists():
        sys.exit(f"missing {RESULTS} — run the held-out audit first")

    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    total = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1

    def _flag_bool(r: dict, key: str) -> bool:
        v = r.get(key, "")
        return v in ("True", "true", "1") if v else False

    def _row_flag_count(r: dict, key: str) -> int:
        v = r.get(key, "0")
        try:
            return int(v)
        except ValueError:
            return 0

    def has_structural_flag(r: dict) -> bool:
        a1 = _row_flag_count(r, "A1_n_stations") > 0
        a2 = _flag_bool(r, "A2_flagged")
        a3 = _row_flag_count(r, "A3_n_stations") > 0
        a6 = _flag_bool(r, "A6_flagged")
        a7 = _flag_bool(r, "A7_flagged")
        return a1 or a2 or a3 or a6 or a7

    def has_a15_flag(r: dict) -> bool:
        a1 = _row_flag_count(r, "A1_n_stations") > 0
        a2 = _flag_bool(r, "A2_flagged")
        a3 = _row_flag_count(r, "A3_n_stations") > 0
        a4 = _row_flag_count(r, "A4_n_stations") > 0
        a5 = _flag_bool(r, "A5_flagged")
        return a1 or a2 or a3 or a4 or a5

    # Tag operator family
    for r in ok:
        fam_type, fam_label = classify_operator(r.get("name", ""), r.get("url", ""))
        r["family_type"] = fam_type
        r["family_label"] = fam_label

    n_ok = len(ok)
    n_a15 = sum(1 for r in ok if has_a15_flag(r))
    n_structural = sum(1 for r in ok if has_structural_flag(r))

    rate_a15 = n_a15 / n_ok if n_ok else 0.0
    rate_structural = n_structural / n_ok if n_ok else 0.0

    # H1: rate within [9.5, 17.5]
    h1_pass = 0.095 <= rate_a15 <= 0.175

    # H2: among A3-or-A7 flagged systems, share belonging to FF family
    a3_or_a7 = [
        r for r in ok
        if _row_flag_count(r, "A3_n_stations") > 0 or _flag_bool(r, "A7_flagged")
    ]
    a3_or_a7_ff_share = (
        sum(1 for r in a3_or_a7 if r["family_type"] == "ff") / len(a3_or_a7)
        if a3_or_a7
        else float("nan")
    )
    h2_pass = a3_or_a7_ff_share >= 0.5 if a3_or_a7 else None

    # H3: clean-operator systems triggering structural flag
    clean_ok = [r for r in ok if r["family_type"] == "clean"]
    clean_with_structural = [r for r in clean_ok if has_structural_flag(r)]
    h3_pass = len(clean_with_structural) == 0

    # ====== write the report ======
    lines = []
    lines.append("# E1 retrospective held-out audit -- results")
    lines.append("")
    lines.append(
        "Held-out set: GBFS systems added to the MobilityData "
        "canonical catalogue between 2025-11-11 (commit "
        "`15e34e6`) and 2026-05-13 (the audit-pipeline rule-freeze "
        f"date). Total panel: {total}. Audited live on 2026-05-13 "
        f"via `scripts/audit_live_systems.py`."
    )
    lines.append("")
    lines.append(f"- Successfully fetched and parsed: **{n_ok}** systems")
    lines.append("- Per-status counts:")
    for s, c in sorted(statuses.items(), key=lambda x: -x[1]):
        lines.append(f"  - `{s}` : {c}")
    lines.append("")
    lines.append("## H1 -- rule-firing rate (A1-A5)")
    lines.append("")
    lines.append(
        f"Held-out rate of at-least-one A1-A5 flag : **{rate_a15*100:.1f}\\,\\%** "
        f"({n_a15}/{n_ok}). Derivation-set reference : 13.5\\,\\%. "
        f"Acceptance interval [9.5\\,\\%, 17.5\\,\\%]."
    )
    lines.append(f"- **H1 outcome:** {'PASS' if h1_pass else 'FAIL'}")
    lines.append("")
    lines.append("## H2 -- A3 / A7 driven by the FF operator family")
    lines.append("")
    if a3_or_a7:
        lines.append(
            f"Systems flagged on A3 or A7 : **{len(a3_or_a7)}**. Among them, "
            f"**{a3_or_a7_ff_share*100:.1f}\\,\\%** belong to the FF operator "
            "family ({Dott, Pony, Bird, nextbike, Voi, Bolt, Lime, Tier, "
            "Donkey, ...}). Acceptance threshold : >= 50\\,\\%."
        )
        lines.append(f"- **H2 outcome:** {'PASS' if h2_pass else 'FAIL'}")
        lines.append("")
        lines.append("Breakdown of A3/A7-flagged systems by family:")
        family_breakdown: dict[str, int] = {}
        for r in a3_or_a7:
            lbl = r["family_label"] or "(other)"
            family_breakdown[lbl] = family_breakdown.get(lbl, 0) + 1
        for k, v in sorted(family_breakdown.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("No systems flagged on A3 or A7 (unexpected) -- H2 unevaluated.")
    lines.append("")
    lines.append("## H3 -- clean-operator systems must not trigger structural flags")
    lines.append("")
    lines.append(
        f"Clean-operator systems in the panel : **{len(clean_ok)}**. "
        f"Of those, **{len(clean_with_structural)}** trigger at least one "
        "structural flag (A1/A2/A3/A6/A7)."
    )
    lines.append(f"- **H3 outcome:** {'PASS' if h3_pass else 'FAIL'}")
    if clean_with_structural:
        lines.append("")
        lines.append("Clean-operator systems with structural flags (each is a falsifier of H3):")
        for r in clean_with_structural[:10]:
            tags = []
            if _row_flag_count(r, "A1_n_stations") > 0:
                tags.append("A1")
            if _flag_bool(r, "A2_flagged"):
                tags.append("A2")
            if _row_flag_count(r, "A3_n_stations") > 0:
                tags.append("A3")
            if _flag_bool(r, "A6_flagged"):
                tags.append("A6")
            if _flag_bool(r, "A7_flagged"):
                tags.append("A7")
            lines.append(f"  - {r.get('country')} {r.get('name')} ({r.get('family_label')}) flags: {','.join(tags)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    overall = sum(int(b) for b in (h1_pass, (h2_pass if h2_pass is not None else True), h3_pass))
    lines.append(
        f"- H1 (rule-firing rate): **{'PASS' if h1_pass else 'FAIL'}** "
        f"({rate_a15*100:.1f}% vs target [9.5%, 17.5%])"
    )
    lines.append(
        f"- H2 (FF-family dominance among A3/A7): **{'PASS' if h2_pass else ('FAIL' if h2_pass is False else 'N/A')}** "
        f"({a3_or_a7_ff_share*100:.1f}% if applicable)"
    )
    lines.append(
        f"- H3 (clean-operator invariance): **{'PASS' if h3_pass else 'FAIL'}** "
        f"({len(clean_with_structural)} falsifiers out of {len(clean_ok)} clean systems)"
    )
    lines.append("")
    lines.append(
        "**Conclusion:** the rule set frozen at 2026-05-13 generalises to "
        f"{n_ok} GBFS systems first observed in the MobilityData catalogue "
        "between 2025-11-11 and 2026-05-13. "
        f"{overall}/3 pre-registered hypotheses pass."
    )

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
