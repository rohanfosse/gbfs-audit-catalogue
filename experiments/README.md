# Experiments

Validation experiments behind the GBFS Audit Catalogue. Each subpackage
is self-contained and reproducible; `run_all.py` orchestrates the two
core ablations (XP2 + XP3).

## Index

| Dir | What it validates | Entry point |
|---|---|---|
| [`xp1_dynamic_audit/`](xp1_dynamic_audit/) | Shannon-entropy zombie detector for `station_status` (collector + classifier) | `python -m experiments.xp1_dynamic_audit.run_xp1` |
| [`xp2_spatial_topology/`](xp2_spatial_topology/) | Topology-aware A4 detector (HDBSCAN + spectral) vs. legacy centroid, ablation on 46,307 stations | `python -m experiments.xp2_spatial_topology.run_xp2` |
| [`xp3_looo_validation/`](xp3_looo_validation/) | Leave-one-operator-out cross-validation with bootstrap CI | `python -m experiments.xp3_looo_validation.run_xp3` |
| [`annotation/`](annotation/) | Human ground-truth validation of A1–A7 (stratified sample, Streamlit annotator, inter-rater reliability) | see [`annotation/PROTOCOL.md`](annotation/PROTOCOL.md) |
| [`e1_holdout/`](e1_holdout/) | Retrospective hold-out on 12 months of MobilityData additions | data + `scripts/e1_analyze.py` |
| [`e2_threshold_sensitivity/`](e2_threshold_sensitivity/) | σ_max sweep, A3 KDE threshold, global panel | data + `scripts/` |
| [`e5_europe/`](e5_europe/) | Cross-country negative-control panel (13 systems, 6 stacks) | data |
| [`config/`](config/) | Hyperparameters (`defaults.yaml`) | — |

## Reproducing the core ablations

```bash
pip install -e ".[experiments]"

python -m experiments.run_all \
    --catalogue catalogue/stations_gold_standard_final.parquet \
    --output results/
```

XP1 needs a 14-day `station_status` collection first:

```bash
python -m experiments.xp1_dynamic_audit.run_xp1 collect \
    --feeds feeds.csv --output data/xp1/ --days 14
```

## Human annotation campaign

The `annotation/` subsystem establishes a human ground truth to compute
precision/recall of each rule. It is **pipeline-agnostic**: annotators
judge the real-world state of a station; per-rule TP/FP/FN are derived
afterwards. See [`annotation/PROTOCOL.md`](annotation/PROTOCOL.md) for the
design and [`annotation/HOSTING.md`](annotation/HOSTING.md) to deploy the
shared annotator (Supabase).

```bash
# 1. Extract the stratified sample
python -m experiments.annotation.sample_extractor \
    --catalogue catalogue/stations_gold_standard_final.parquet \
    --ablation results/xp2/xp2_ablation.parquet \
    --output experiments/annotation/sample.csv

# 2. Annotate (two independent annotators)
streamlit run experiments/annotation/annotator_app.py

# 3. Inter-rater reliability + per-rule metrics
python -m experiments.annotation.compute_reliability \
    --labels1 labels_rohan.csv --labels2 labels_gael.csv \
    --output reliability_report.json
```
