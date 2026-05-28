# Human Annotation Protocol for Ground-Truth Validation

**Purpose**: Establish a human-annotated ground truth for a stratified
sample of stations to compute precision, recall and F1 of the A1–A7
audit rules and the A4v2 composite detector.

**Protocol version**: 3.0 (pipeline-agnostic revision, 2026-05-28)

## Sampling strategy

Stratified random sampling from the 46,307-station certified catalogue.
Each stratum targets one audit rule or detector comparison.  Stations
are **deduplicated** across strata (a station appears in one stratum
only) and the final sample is **shuffled** (SEED=42) to eliminate
order effects (anchoring, fatigue, learning).

Sample sizes were revised after methodological review: the A4 discordant
strata (the paper's core contribution) are raised to n=50 for publishable
confidence intervals; other imagery-validatable strata are enlarged for
power.  Total ≈ 325 stations.

| Stratum | Selection criterion | N | Rationale |
|---|---|---|---|
| Clean dock-based | No flag, confidence = high | 45 | False-negative check (negative control) |
| A1 (carsharing) | flag_A1 = True | 30 | Is the vehicle a bicycle? |
| A2 (placeholder) | flag_A2 = True | 20 | System-level (pool-capped: 40 stations, 1 operator) |
| A3 (free-floating) | flag_A3 = True, A2 = False | 45 | Physical dock or virtual anchor? |
| A4 AGREE_FLAG | Both detectors flag | 30 | Algorithmic consensus |
| A4 DISCORDANT_LEGACY | Legacy centroid only | 50 | Key stratum: are these true FP? |
| A4 DISCORDANT_COMPOSITE | Composite only | 50 | Does composite find real outliers? |
| **A5 (out-of-perimeter)** | flag_A5 = True | 30 | Are flagged coordinates truly aberrant? |
| A6 (zero-capacity) | flag_A6 = True | 10 | Does a 0-dock station exist? (empty in v1.0) |
| A7 (NaN capacity) | flag_A7 = True, A3 = False | 25 | System-level (pool-capped: 52 stations, 1 operator) |
| A3 boundary | capacity_ratio in [2, 5] | 15 | Threshold sensitivity zone (empty in v1.0) |

**Note on empty strata**: A6 and A3-boundary may yield 0 eligible
stations depending on the catalogue snapshot (A6 has 0 stations in the
v1.0 release; A3-boundary has 0 systems with ratio in [2, 5]).  These
are documented as untestable rather than silently skipped.

**Note on mono-operator strata**: A2 (VéloZef only) and A7 (Optymo
Belfort only) are limited to a single operator by the structure of the
catalogue.  Precision/recall on these strata reflects the operator's
data quality, not the rule's general validity.  This limitation must be
stated in the manuscript.

## Annotation procedure

### Phase 0 — Calibration (pilot)

Before independent annotation begins, both annotators jointly annotate
**5 calibration stations** (one per major stratum family: clean, A1, A3,
A4, A7).  They compare answers, discuss disagreements, and align their
interpretation of the criteria.  Calibration stations are **excluded**
from the final reliability computation.

### Phase 1 — Independent annotation (strictly pipeline-agnostic)

Each annotator evaluates every station with the **same universal rubric**
(no per-stratum hint, no stratum name shown — anti-bias).  They judge the
**real-world state** of the station, never whether "the pipeline is right":

1. **Observation** — Type of facility present; physical infrastructure
   visible.  (CyclOSM map, Street View, satellite, Overpass API overlay)
2. **Évaluation** — Is the declared capacity coherent?  Is the position
   consistent with the operator's network?
3. **Synthèse** — A single agnostic question: "Is this a real bikeshare
   station, physically present and correctly described?"
   (yes / no / indeterminate) + a confidence score (1–5).

The pipeline output (A1–A7 flags, audit confidence, audited type) is
**hidden behind an opt-in panel** to prevent anchoring.  Crucially, the
verdict options contain **no reference to the pipeline** — there is no
"pipeline false positive" option, which would require the (blind)
annotator to know the pipeline's decision.  Per-rule TP/FP/FN are derived
*a posteriori* (see Metrics).

### Phase 2 — Adjudication

Each factual question (Q1–Q4) and the holistic verdict are adjudicated
independently: consensus → gold; disagreement → third adjudicator.  The
per-rule metrics use the adjudicated **factual** answers, so the gold
standard never encodes a judgment about the pipeline.

## Annotators

- Minimum 2 independent annotators (blind to each other's labels)
- Annotator 1: domain expert (urban mobility researcher)
- Annotator 2: data engineer (familiar with GBFS but not with this audit)
- Disagreements resolved by a third adjudicator

## Inter-rater reliability

- **Cohen's kappa** AND **Krippendorff's alpha** (nominal) on each factual
  question and on the holistic verdict.  Alpha is preferred because it
  handles missing/indeterminate data and generalises to >2 coders.
- **Target**: κ ≥ 0.70 (substantial agreement per Landis & Koch, 1977)
- If κ < 0.60 on any question, revise the rubric and re-run calibration
- **Indeterminate rate** reported per stratum — indeterminate answers are
  excluded from precision/recall but their count is stated ("irreducible
  ambiguity rate" of human audit)

## Metrics to report

Per-rule precision/recall are **derived analytically** from the
adjudicated factual answers, not from a pipeline-referencing verdict:

| Rule | "Anomaly is real" iff | Validation |
|---|---|---|
| A1 (out-of-domain) | Q1 (is bikeshare?) = no | per-station |
| A3 (free-floating) | Q3 (physical infra?) = no | per-station |
| A4 (geospatial outlier) | Q4 (within perimeter?) = no | per-station |
| A5 (out-of-perimeter) | Q4 (within perimeter?) = no | per-station |
| A6 (zero-capacity dock) | Q3 (physical infra?) = no | per-station |
| A2 (placeholder capacity) | — | **system-level** |
| A7 (null capacity field) | — | **system-level** |

- **Precision** = TP / (TP+FP); **Recall** = TP / (TP+FN); **F1** = harmonic mean
- **Wilson 95% CI** on each (small-sample correction)
- **A2/A7** are structural data-property rules not validatable from
  single-station imagery → reported at the system level (count of affected
  systems + semantic interpretation), not as per-station precision/recall.

For the A4 ablation specifically:

- FP rate of the legacy centroid on the DISCORDANT_LEGACY stratum, where a
  true FP = a legacy-only flag the human judges in-perimeter (Q4 = yes).
- This directly answers "are the 8,005 discordant stations true FP?"

### Power analysis

| n per stratum | Width of 95% CI at p=0.80 | Interpretation |
|---|---|---|
| 10 | 0.50 | Too wide for standalone claims |
| 15 | 0.40 | Report with explicit caveats |
| 20 | 0.35 | Acceptable for directional evidence |
| 25 | 0.31 | Acceptable |
| 30 | 0.29 | Good |

Strata with n ≤ 15 should be discussed as "indicative" rather than
"definitive" in the manuscript.

## Quality assurance

- **Annotation time** is recorded per station; stations completed in
  < 10 seconds are flagged for review (likely accidental clicks)
- **Temporal drift**: agreement rate is computed on the first half vs
  the second half of each annotator's labels to detect fatigue effects
- **Confidence calibration**: stations with confidence = 1 are compared
  to the gold verdict to assess whether low-confidence annotations
  predict disagreement

## Practical timeline

| Phase | Duration | Output |
|---|---|---|
| Sample extraction + interface setup | 1 day | Stratified sample CSV |
| Calibration (5 joint stations) | 30 min | Aligned interpretation |
| Annotator 1 pass | 2 days | Labels CSV |
| Annotator 2 pass | 2 days | Labels CSV |
| Adjudication + kappa computation | 1 day | Gold labels + reliability report |
| Metrics computation + manuscript update | 1 day | Precision/recall table for paper |
| **Total** | **~7 days** | |

## Storage

- **Local**: SQLite database (`annotations.db`), zero-config
- **Online deployment**: set `ANNOTATION_DB_PATH` env var to a
  persistent volume, or `ANNOTATION_DB_URL` for PostgreSQL/Supabase
- **Legacy export**: CSV in Q1–Q5 format for `compute_reliability.py`

## Output files

- `experiments/annotation/sample.csv` — the stratified sample (325 stations)
- `experiments/annotation/annotations.db` — SQLite annotation store
- `experiments/annotation/labels_<annotator>.csv` — legacy CSV export
- `experiments/annotation/gold_labels.csv` — adjudicated consensus
- `experiments/annotation/reliability_report.json` — kappas, P/R, indeterminate rates
