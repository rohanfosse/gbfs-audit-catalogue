# GBFS Audit Catalogue

> A certified, anomaly-flagged reference dataset for 46,307 bike-sharing
> stations across 123 French operators, with the open-source pipeline that
> produced it.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20125460.svg)](https://doi.org/10.5281/zenodo.20125460)
[![tests](https://github.com/rohanfosse/gbfs-audit-catalogue/actions/workflows/tests.yml/badge.svg)](https://github.com/rohanfosse/gbfs-audit-catalogue/actions/workflows/tests.yml)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/rohanfosse/gbfs-audit-catalogue)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20demo-red)](https://gbfs-audit.streamlit.app)
[![Docs](https://img.shields.io/badge/Docs-Project%20page-lightgrey)](https://rohanfosse.github.io/gbfs-audit-catalogue)
[![Data: ODbL](https://img.shields.io/badge/data-ODbL%20v1.0-blue)](LICENSE-DATA)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)

## What this is

GBFS guarantees that bike-sharing feeds are syntactically consistent — not
that they are semantically comparable. An audit of 1,509 systems across
48 countries surfaces a unified taxonomy of seven data-quality classes
(five structural errors + two semantic warnings) that remove **30.9 %** of
the raw French stations and relabel a further **61 %**. The released
catalogue exposes the verdict per row, so downstream consumers can filter
without rerunning the pipeline.

## Quick start

```python
from datasets import load_dataset
gs = load_dataset("rohanfosse/gbfs-audit-catalogue", split="train").to_pandas()

# High-confidence dock-based stations (4,721)
clean = gs[(gs.station_type == "docked_bike") & (gs.audit_confidence == "high")]
```

The 46-column schema, the seven-class taxonomy and eight reproducible
recipes (anomaly filtering, INSEE join, Bordeaux before/after, etc.) are
documented on the [**project page**](https://rohanfosse.github.io/gbfs-audit-catalogue)
and the [**dataset card**](https://huggingface.co/datasets/rohanfosse/gbfs-audit-catalogue).

## Learn more

| Resource | Where |
| --- | --- |
| Project page (long-form) | [rohanfosse.github.io/gbfs-audit-catalogue](https://rohanfosse.github.io/gbfs-audit-catalogue) |
| Dataset card + full schema | [huggingface.co/datasets/rohanfosse/gbfs-audit-catalogue](https://huggingface.co/datasets/rohanfosse/gbfs-audit-catalogue) |
| Live dashboard | [gbfs-audit.streamlit.app](https://gbfs-audit.streamlit.app) |
| Manuscript LaTeX source | [`paper/`](paper/) |
| Reproducible recipes | [`notebooks/catalogue_recipes.ipynb`](notebooks/catalogue_recipes.ipynb) |
| Docker reproduction | `docker build -t gbfs-audit:1.0 . && docker run --rm gbfs-audit:1.0` |
| Tests | `pytest` (24 tests, 85 % coverage, Python 3.10–3.12) |

## Citation

```bibtex
@article{Fosse2026gbfs,
  author  = {Foss\'e, Rohan and Pallares, Ga\"el},
  title   = {Auditing GBFS bike-sharing feeds at country and global scale:
             A reproducible anomaly taxonomy for open mobility data},
  journal = {Computer Standards \& Interfaces},
  year    = {2026},
  note    = {Manuscript under peer review}
}

@dataset{Fosse2026gbfsdata,
  author    = {Foss\'e, Rohan and Pallares, Ga\"el},
  title     = {{GBFS Audit Catalogue} v1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20125460}
}
```

A [`CITATION.cff`](CITATION.cff) is also provided for GitHub's citation tooling.

## Licences

Code under [MIT](LICENSE); data under [ODbL v1.0](LICENSE-DATA). Upstream
attributions for the contextual-enrichment sources (INSEE, IGN, ONISR,
GTFS aggregator) are listed in `LICENSE-DATA`.

## Contact

**Rohan Fossé** ([rfosse@cesi.fr](mailto:rfosse@cesi.fr)) — CESI École d'Ingénieurs, Montpellier.
**Gaël Pallares** — CESI LINEACT (EA 7527), Montpellier.

Issues and contributions are welcome on the
[issue tracker](https://github.com/rohanfosse/gbfs-audit-catalogue/issues).
