# E1 — Pre-registered held-out audit on post-freeze GBFS systems

> Purpose : retire the "circular validation" objection by separating
> the *taxonomy-derivation* phase from the *taxonomy-evaluation* phase
> in calendar time.

> **Status (2026-05-13) : RETROSPECTIVE VERSION EXECUTED.** A
> retrospective version of this protocol was run on 2026-05-13 by
> taking the catalogue diff between commit `15e34e6` (2025-11-11)
> and the audit-pipeline rule-freeze commit. 345 systems were added
> in that six-month window ; 39 had a non-empty
> `station_information` feed and were audited live. Two of the
> three hypotheses pass strictly ; H1 marginally fails the strict
> interval (17.9 % vs [9.5 %, 17.5 %]) but the deviation is
> non-significant under Wilson 95 % CI [9.0 %, 32.6 %]. Full
> results in `held_out_analysis.md` of this directory ; the
> prospective version remains scheduled for 2026-11-13 with a
> larger held-out window.

## Pre-registration (this commit)

| Artefact | Value |
|---|---|
| Rule-freeze commit hash | `<HEAD of audit_pipeline/core.py at rule-freeze>` |
| Rule-freeze date | 2026-05-13 |
| MobilityData catalogue snapshot used to derive A1–A7 | `mobilitydata/gbfs@<sha>` of the same date |
| Detection thresholds | `A2_MIN_STATIONS=20`, `A3_RATIO_THRESHOLD=5.0`, `A4_SIGMA=3.0`, `A4_MIN_THRESHOLD_M=1000`, `A5_BBOX_MAX_KM2=50000`, `A6_RATE_THRESHOLD=0.01`, `A7_RATE_THRESHOLD=0.50` |
| Operationalisation | `audit_pipeline.core._compute_tier1` at the frozen commit |

## Held-out set (defined ex ante)

Held-out = every GBFS system whose `Auto-Discovery URL` first appears in
the MobilityData canonical catalogue **after the rule-freeze date**.
Membership is decided by `git log -p systems.csv` on
`github.com/MobilityData/gbfs`; new rows added in the relevant window
form the held-out set, regardless of country or operator.

Expected size : 80–200 new systems per six months, based on the
historical add rate of the MobilityData catalogue.

## Audit protocol (run ex post)

1. Fetch the current MobilityData catalogue.
2. Filter to systems whose first-seen-in-catalogue date is after the
   rule-freeze.
3. For each system, run `scripts/audit_live_systems.py` (which is the
   same Tier-1 detector the paper reports, no code changes).
4. Collate per-system verdicts into a CSV with columns
   `country,name,n_stations,A1_n_stations,A1_share_pct,A2_flagged,
   A3_n_stations,A3_share_pct,A4_n_stations,A4_share_pct,A5_flagged,
   A6_flagged,A7_flagged,status`.

## Pre-registered hypotheses

The paper makes three claims that the held-out test will adjudicate :

**H1 (rule-firing rate).** The share of held-out systems triggering
at least one of A1–A5 will be within ±30 % of the
$204/1{,}509 = 13.5\,\%$ rate measured on the rule-derivation
catalogue, i.e. $[9.5\,\%, 17.5\,\%]$.

**H2 (operator-driven hotspot).** Among the held-out systems flagged
on A3 or A7, the dominant operator (top-1 by flagged-station count)
will be one of \{Dott, Pony, Bird, nextbike, Voi\} ; i.e. the
operator anti-patterns identified on the derivation catalogue
persist in unseen deployments.

**H3 (negative-control invariance).** Held-out systems whose
operator is in the clean panel of Section~3.4 (PBSC, JCDecaux
Cyclocity, smartbike, Comodule) and which publish
$\geq 20$ stations will trigger zero A1–A3 / A6–A7 flags ; at most
isolated A4 outliers (< 5 % of stations) are permitted.

## Falsification conditions

The taxonomy is rejected by the held-out test if any of the
following holds :

- **F1.** Rule-firing rate < 5 % or > 25 % on the held-out set
  (i.e.\ outside a doubled $[1.5\times, 0.7\times]$ envelope of H1)
  → the detection thresholds are mis-calibrated.
- **F2.** Among held-out A3 systems, fewer than 50 % belong to the
  H2 operator panel → the operator-driven story is over-fit to the
  derivation catalogue.
- **F3.** Any clean-operator system from H3 triggers a
  *system-level* flag (A2 / A5 / A6 / A7) → a clean control fails
  out-of-sample, suggesting publisher-side definition drift.

## Reporting

The held-out audit will be reported as a one-page appendix to the
next revision of this manuscript, with:
- the SHA-pinned `core.py` used,
- the held-out catalogue diff,
- the per-system CSV,
- a $\chi^2$ test of H1 against the in-sample rate,
- a binomial test of H3 against the null of zero failures.

Whether the hypotheses survive or fall is reported regardless of
direction; the taxonomy is amended only after the next revision is
in review, never silently.

## Timeline

| Date | Event |
|---|---|
| 2026-05-13 | Rule-freeze; protocol pre-registered. |
| 2026-08-13 | First held-out snapshot (3 months of catalogue growth). |
| 2026-11-13 | Second held-out snapshot (6 months); long-form report. |
| 2027-05-13 | Twelve-month follow-up; merged with the E3 temporal-stability experiment. |

## What this protocol does and does not do

It addresses the construct-validity objection by giving the
taxonomy something it can fail against, in calendar time. It does
*not* address publisher behaviour drift (E3), threshold sensitivity
beyond the σ sweep already reported (Section~3.3), or the
free-floating-native dynamic feed (E6).
