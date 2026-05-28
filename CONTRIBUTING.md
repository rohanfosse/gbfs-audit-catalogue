# Contributing

Thanks for your interest in the GBFS Audit Catalogue. Issues, fixes and
new validation experiments are welcome.

## Reporting issues

Open an issue on the
[tracker](https://github.com/cycling-data-lab/gbfs-audit-catalogue/issues).
For data-quality reports, please include the `uid`, `system_id` and
`station_id` of the affected rows so the case is reproducible.

## Development setup

```bash
git clone https://github.com/cycling-data-lab/gbfs-audit-catalogue.git
cd gbfs-audit-catalogue
python -m pip install -e ".[test,experiments]"
pytest                      # 36 tests, Python 3.10–3.12
```

The Streamlit dashboard and the annotation tool:

```bash
streamlit run app/streamlit_app.py
streamlit run experiments/annotation/annotator_app.py
```

## Pull requests

1. Branch from `main`.
2. Keep changes focused; add or update tests under `tests/` when you touch
   `audit_pipeline/` or the experiment detectors.
3. Run `pytest` locally — CI runs the same on Python 3.10/3.11/3.12.
4. If you change the catalogue schema, regenerate the FAIR manifests:
   `python -m scripts.generate_metadata`.
5. Describe the *why* in the PR body, not just the *what*.

## Conventions

- Conventional-commit style messages (`feat(...)`, `fix(...)`, `test(...)`,
  `docs(...)`).
- Code is formatted to standard PEP 8; type hints are expected on public
  functions.
- The manuscript (`paper/`) is Overleaf-synced — edit it there, not via
  direct commits to `main` (see `paper/OVERLEAF.md`).

## Licensing of contributions

By contributing you agree that code is released under the
[MIT licence](LICENSE) and data/derived artefacts under
[ODbL v1.0](LICENSE-DATA).

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
