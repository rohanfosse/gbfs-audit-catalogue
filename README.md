# GBFS Audit Catalogue

> A certified, anomaly-flagged reference dataset for the 46,307 bike-sharing
> stations published under the French Mobility Orientation Law, together with
> the open-source audit pipeline that produced it.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20125460.svg)](https://doi.org/10.5281/zenodo.20125460)
[![tests](https://github.com/rohanfosse/gbfs-audit-catalogue/actions/workflows/tests.yml/badge.svg)](https://github.com/rohanfosse/gbfs-audit-catalogue/actions/workflows/tests.yml)
[![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/rohanfosse/gbfs-audit-catalogue)
[![Streamlit demo](https://img.shields.io/badge/Streamlit-Live%20demo-red)](https://gbfs-audit.streamlit.app)
[![Project page](https://img.shields.io/badge/GitHub%20Pages-Project%20page-lightgrey)](https://rohanfosse.github.io/gbfs-audit-catalogue)
[![Data licence: ODbL v1.0](https://img.shields.io/badge/data-ODbL%20v1.0-blue)](LICENSE-DATA)
[![Code licence: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)

![Visual abstract: audit of 1,509 GBFS systems, unified A1-A7 taxonomy, certified 46k-station catalogue.](paper/figures/fig00_visual_abstract.png)

## What this is

The General Bikeshare Feed Specification (GBFS) guarantees that bike-sharing
operators publish data in a consistent **syntax**. It does *not* guarantee
that the data is **semantically** consistent across operators — the
`station_information.capacity` integer alone carries six mutually
incompatible meanings in the French corpus (physical dock count,
free-floating fleet average, placeholder constant, NaN, etc.).

This repository releases:

- **Data** — a 46-column Parquet of 46,307 audited stations across 97 French
  cities, with per-row flags (`flag_A1` … `flag_A7`), audited
  capacities, network-geometry features and contextual enrichment from
  INSEE, IGN, ONISR and the national GTFS aggregator.
- **Code** — `audit_pipeline`, a small Python package that re-runs the
  Tier-1 audit (five structural errors A1–A5 + two semantic warnings
  A6–A7) and Tier-2 KD-tree network enrichment on any compatible raw
  GBFS dump.
- **Reproduction artefacts** — the manuscript LaTeX source, a Jupyter
  notebook with eight runnable recipes, a Docker image for bit-exact
  reproduction, a Streamlit dashboard with a Validation tab covering
  E5 (live cross-country panel) and E1 (retrospective hold-out), and
  a pytest suite that exercises every detector.

Intended for researchers in transport policy, urban mobility, data
quality and FAIR open data, as well as operators and regulators who
need a sanity-checked snapshot of the French bike-sharing landscape.

## Key figures

| Indicator | Value |
| --- | --- |
| GBFS systems audited worldwide | 1,509 (48 countries) |
| Structural errors flagged globally (A1–A5) | 204 systems |
| Semantic warnings flagged globally (A6, A7) | 14 (A6) + 215 (A7) |
| French GBFS systems audited | 123 |
| Certified stations released | 46,307 |
| Dock-based stations (all) | 5,442 |
| Dock-based stations (high audit confidence) | 4,721 |
| Cities covered | 97 |
| Raw stations removed from the catalogue | 30.9 % (further 61 % relabelled) |
| Columns in the released schema | 46 typed columns |
| Parquet file size | 6.6 MB |

## Quick start

### Hugging Face Datasets

```python
from datasets import load_dataset
gs = load_dataset("rohanfosse/gbfs-audit-catalogue", split="train").to_pandas()
```

### Direct from Zenodo (no authentication)

```python
import pandas as pd
gs = pd.read_parquet(
    "https://zenodo.org/records/20125460/files/"
    "stations_gold_standard_final.parquet"
)
```

### Local clone of this repository

```python
from audit_pipeline import load_catalogue, load_summary
gs = load_catalogue()        # the 46-column certified parquet
summary = load_summary()     # one row per audited system
```

### Inspect the audit at the row level

```python
# All dock-based stations: 5,442
docked = gs[gs.station_type == "docked_bike"]

# Same, restricted to systems the audit is confident about: 4,721
clean = docked[docked.audit_confidence == "high"]

# Per-operator flag rates
gs.groupby("operator_name").agg(
    n=("uid", "size"),
    A3_rate=("flag_A3", "mean"),
    A7_rate=("flag_A7", "mean"),
).sort_values("n", ascending=False).head(10)
```

A self-contained companion notebook with eight recipes is provided at
[`notebooks/catalogue_recipes.ipynb`](notebooks/catalogue_recipes.ipynb)
(loading, anomaly filtering, INSEE join, dock-density mapping,
Bordeaux before/after, soft-mobility ranking, capacity-semantics audit
and cross-country comparison).

## Repository layout

```text
gbfs-audit-catalogue/
├── catalogue/           Certified parquet and per-system summary CSV
├── audit_pipeline/      Standalone Python package (load + enrich API)
├── tests/               pytest suite (24 tests, 85% coverage)
├── notebooks/           Companion notebook with 8 reproducible recipes
├── paper/               Manuscript LaTeX, figures, BibTeX (CSI submission)
├── app/                 Focused Streamlit dashboard (4 tabs)
├── docs/                GitHub Pages project page sources
├── huggingface/         Dataset card and Hub publication instructions
├── .github/workflows/   Continuous integration (pytest on 3.10/3.11/3.12)
├── requirements.txt     Runtime dependencies
├── Dockerfile           Bit-exact reproduction environment
├── pyproject.toml       Python package metadata
├── LICENSE              MIT (code)
├── LICENSE-DATA         ODbL v1.0 (data)
└── CITATION.cff         Machine-readable citation
```

## The seven data-quality classes

Five **structural errors** (A1–A5) plus two **semantic warnings**
(A6–A7) for spec-compliant publication patterns whose
downstream-consumer interpretation is ambiguous. The unified
taxonomy is built jointly from the French corpus and the global
MobilityData catalogue.

| Class | Type | Name | Signature | FR | Global |
| --- | --- | --- | --- | --- | --- |
| A1 | structural | Out-of-domain inclusion | Car-sharing advertised as a bike-sharing system | 17 | 46 |
| A2 | structural | Placeholder capacity | Constant non-zero capacity across docked subset | 1 | 48 |
| A3 | structural | Structural over-capacity | Conditional averaging on free-floating fleet anchors | 41 | 33 |
| A4 | structural | Geospatial outlier | 3-sigma outlier on per-system nearest-neighbour distance | 78 (1.1 % stns) | 81 |
| A5 | structural | Out-of-perimeter coverage | System bounding box > 50,000 km² | 4 | 17 |
| A6 | warning | Zero-capacity dock | ≥ 1 % of docked stations declare capacity = 0 | 0 | 14 |
| A7 | warning | Null capacity field | ≥ 50 % of stations declare capacity = NaN | 32 (FF) | 215 |

## Schema (46 columns)

The full machine-readable schema (JSON Schema, DCAT-AP record,
Frictionless Data Package descriptor and Croissant JSON-LD manifest)
ships with the Zenodo deposit. The summary below groups the 46 columns
by purpose and source.

| Group | Columns | Source |
| --- | --- | --- |
| Identifiers (5) | `uid`, `station_id`, `system_id`, `system_name`, `source` | GBFS |
| Spatial (5) | `lat`, `lon`, `city`, `commune_name`, `code_commune`, `region_id` | GBFS + INSEE COG |
| Station description (4) | `station_name`, `address`, `capacity`, `n_stations_system` | GBFS |
| **Audit pipeline (11)** | `station_type`, `capacity_raw`, `capacity_audited`, `flag_A1`–`flag_A7`, `operator_name`, `audit_confidence`, `fetched_at` | This work |
| **Network geometry (5)** | `dist_to_nearest_station_m`, `n_stations_within_500m`, `n_stations_within_1km`, `nearest_system_dist_m`, `catchment_density_per_km2` | KNN on this work |
| Topography (2) | `elevation_m`, `topography_roughness_index` | IGN BD ALTI |
| Cycling infrastructure (2) | `infra_cyclable_km`, `infra_cyclable_pct` | IGN BD TOPO, 300 m buffer |
| Safety (1) | `baac_accidents_cyclistes` | ONISR BAAC, 500 m radius, 5 yr |
| Multimodal access (2) | `gtfs_heavy_stops_300m`, `gtfs_stops_within_300m_pct` | National GTFS aggregator |
| Socio-economy (5) | `revenu_median_uc`, `gini_revenu`, `revenu_d1`, `ecart_interquar`, `part_menages_voit0` | INSEE Filosofi |
| Modal share (1) | `part_velo_travail` | INSEE Recensement |

The 16 columns in bold are the audit-pipeline contribution at the row
level: every station carries its verdict per class, its operator
attribution, its audit confidence and its network-geometry context, so
downstream consumers can filter without rerunning the pipeline. Every
column has a stable name, a declared dtype, a source pointer and a
measured completeness rate (see the dataset card on Hugging Face for
the full table).

## Reproduction

### Local Python environment

```bash
git clone https://github.com/rohanfosse/gbfs-audit-catalogue.git
cd gbfs-audit-catalogue
pip install -r requirements.txt
python -c "from audit_pipeline import load_catalogue; print(load_catalogue().shape)"
```

### Containerised reproduction

```bash
docker build -t gbfs-audit:1.0 .
docker run --rm gbfs-audit:1.0 \
  python -c "from audit_pipeline import load_catalogue; print(load_catalogue().shape)"
```

### Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard is also deployed publicly at
[gbfs-audit.streamlit.app](https://gbfs-audit.streamlit.app).

### Development and tests

```bash
pip install -e ".[test]"
pytest                              # 24 tests, ~1 s
pytest --cov=audit_pipeline         # with coverage (currently 85 %)
```

The suite exercises each of the seven data-quality classes on dedicated
synthetic fixtures, plus the end-to-end `enrich()` pipeline and the
operator-normalisation lookup. It runs on every push to `main` and on
every pull request via the GitHub Actions workflow at
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) on
Python 3.10, 3.11 and 3.12.

## Manuscript and Overleaf

The Computer Standards & Interfaces manuscript LaTeX source lives
under [`paper/`](paper/). A dedicated `overleaf-paper` branch ships
the same files flat at the branch root, which is the layout Overleaf
expects. See [`paper/OVERLEAF.md`](paper/OVERLEAF.md) for the linking
instructions and [`paper/sync_overleaf.sh`](paper/sync_overleaf.sh)
for the helper script that refreshes the side branch after edits.

## Citation

Please cite both the manuscript and the Zenodo deposit when reusing
the catalogue.

```bibtex
@article{Fosse2026gbfs,
  author  = {Foss\'e, Rohan and Pallares, Ga\"el},
  title   = {Auditing GBFS bike-sharing feeds at country and global scale:
             A reproducible anomaly taxonomy for open mobility data},
  journal = {Computer Standards \& Interfaces},
  year    = {2026},
  note    = {Manuscript under peer review; preprint forthcoming}
}

@dataset{Fosse2026gbfsdata,
  author    = {Foss\'e, Rohan and Pallares, Ga\"el},
  title     = {{GBFS Audit Catalogue} v1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20125460}
}
```

A [`CITATION.cff`](CITATION.cff) file is also provided at the
repository root for GitHub's automatic citation tooling.

## Licences

Code (the `audit_pipeline`, `app`, `notebooks`, `tests` and `paper`
directories) is released under the MIT licence. Data (the contents of
`catalogue/`, the Zenodo deposit and the Hugging Face dataset mirror)
is released under the Open Data Commons Open Database License (ODbL)
v1.0. Upstream attributions for the data sources used in the
contextual enrichment are listed in [`LICENSE-DATA`](LICENSE-DATA).

## Contact

**Rohan Fossé** (lead contact, `rfosse@cesi.fr`) — CESI École d'Ingénieurs, Montpellier, France.

**Gaël Pallares** — CESI LINEACT (EA 7527), Montpellier, France.

Issues and contributions are welcome at
[github.com/rohanfosse/gbfs-audit-catalogue/issues](https://github.com/rohanfosse/gbfs-audit-catalogue/issues).

## See also

[`bikeshare-data-explorer`](https://github.com/rohanfosse/bikeshare-data-explorer)
hosts the wider research programme this paper belongs to, including
the Soft Mobility Index (IMD), the social-equity index (IES) and the
Bayesian urban-scale station-level monitor.
