"""Root-level entry point for Streamlit Cloud.

Streamlit Cloud defaults to ``streamlit_app.py`` at the repository
root. The real app lives in ``app/streamlit_app.py`` to keep the
top-level layout clean ; this thin wrapper re-runs that script
under ``__name__ == '__main__'`` so the wrapped page renders
exactly as if launched directly.

Run locally :
    streamlit run streamlit_app.py        # via this wrapper
    streamlit run app/streamlit_app.py    # directly
"""
from pathlib import Path
from runpy import run_path

run_path(
    str(Path(__file__).parent / "app" / "streamlit_app.py"),
    run_name="__main__",
)
