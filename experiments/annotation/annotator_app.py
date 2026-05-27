"""Human annotation tool for GBFS Audit ground-truth validation.

Usage:
    streamlit run experiments/annotation/annotator_app.py

Each annotator enters their name, then works through the 175-station
sample one by one. Answers are saved to a per-annotator CSV that can
be merged offline for inter-rater reliability computation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_PATH = Path(__file__).resolve().parent / "sample_200.csv"
LABELS_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="GBFS Annotation Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: annotator identity ──────────────────────────────────────

with st.sidebar:
    st.markdown("### Annotator")
    annotator_name = st.text_input(
        "Your name",
        value="",
        placeholder="e.g. Rohan, Gaël",
        key="annotator_name",
    )
    if not annotator_name.strip():
        st.warning("Enter your name to start annotating.")
        st.stop()

    annotator_id = annotator_name.strip().lower().replace(" ", "_")
    labels_path = LABELS_DIR / f"labels_{annotator_id}.csv"

    st.markdown("---")
    st.markdown("### Protocol")
    st.markdown(
        "For each station:\n"
        "1. Check the **map** (is there a bike dock?)\n"
        "2. Read the **metadata** (capacity, type)\n"
        "3. Answer the **5 questions**\n"
        "4. Click **Save & Next**\n\n"
        "Your labels are auto-saved to:\n"
        f"`{labels_path.name}`"
    )

# ── Load data ────────────────────────────────────────────────────────

if not SAMPLE_PATH.exists():
    st.error(f"Sample file not found: {SAMPLE_PATH}")
    st.stop()

sample = pd.read_csv(SAMPLE_PATH)

if labels_path.exists():
    labels = pd.read_csv(labels_path)
    done_keys = set(
        labels.apply(lambda r: f"{r['system_id']}_{r['station_id']}", axis=1)
    )
else:
    labels = pd.DataFrame()
    done_keys = set()

sample["_key"] = sample.apply(
    lambda r: f"{r['system_id']}_{r['station_id']}", axis=1
)
remaining = sample[~sample["_key"].isin(done_keys)]

# ── Progress ─────────────────────────────────────────────────────────

n_total = len(sample)
n_done = n_total - len(remaining)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Progress")
    st.progress(n_done / n_total if n_total > 0 else 0)
    st.metric("Annotated", f"{n_done} / {n_total}")

    by_stratum = sample.copy()
    by_stratum["done"] = by_stratum["_key"].isin(done_keys)
    stratum_progress = (
        by_stratum.groupby("stratum")
        .agg(total=("_key", "count"), done=("done", "sum"))
        .reset_index()
    )
    stratum_progress["remaining"] = stratum_progress["total"] - stratum_progress["done"]
    st.dataframe(
        stratum_progress[["stratum", "done", "total"]],
        hide_index=True,
        height=300,
    )

if len(remaining) == 0:
    st.success("All 175 stations annotated. You're done!")
    st.balloons()
    st.stop()

# ── Current station ──────────────────────────────────────────────────

row = remaining.iloc[0]

st.title("GBFS Annotation Tool")
st.markdown(
    f"**Station {n_done + 1} / {n_total}** — "
    f"Stratum: `{row['stratum']}` — "
    f"System: `{row['system_id']}` — "
    f"Station: `{row['station_id']}`"
)

col_map, col_meta = st.columns([3, 2])

# ── Map ──────────────────────────────────────────────────────────────

with col_map:
    st.markdown("#### Location")
    lat = float(row["lat"]) if pd.notna(row["lat"]) else None
    lon = float(row["lon"]) if pd.notna(row["lon"]) else None

    if lat is not None and lon is not None:
        map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.map(map_df, zoom=14, size=20)

        st.markdown(
            f"[Open in Google Maps](https://www.google.com/maps/@{lat},{lon},18z) · "
            f"[Open in OpenStreetMap](https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=18)"
        )
    else:
        st.warning("No coordinates available for this station.")

# ── Metadata ─────────────────────────────────────────────────────────

with col_meta:
    st.markdown("#### Station metadata")

    meta_items = {
        "system_id": row.get("system_id"),
        "station_id": row.get("station_id"),
        "operator_name": row.get("operator_name"),
        "city": row.get("city"),
        "station_type": row.get("station_type"),
        "capacity": row.get("capacity"),
        "audit_confidence": row.get("audit_confidence"),
        "lat": lat,
        "lon": lon,
    }
    for k, v in meta_items.items():
        val = str(v) if pd.notna(v) else "*missing*"
        st.markdown(f"**{k}**: `{val}`")

    st.markdown("#### Audit flags")
    flags = []
    for i in range(1, 8):
        col = f"flag_A{i}"
        if col in row.index and row[col]:
            flags.append(f"A{i}")
    if flags:
        st.error(f"Flagged: **{', '.join(flags)}**")
    else:
        st.success("No flags triggered")

# ── Annotation form ──────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Annotation")

q1, q2 = st.columns(2)

with q1:
    a_q1 = st.radio(
        "**Q1.** Does this station represent a bike-sharing system?",
        ["yes", "no", "indeterminate"],
        index=None,
        key="q1",
        help="Check vehicle type: is it a bicycle/e-bike, or a car/scooter-only system?",
    )

    a_q3 = st.radio(
        "**Q3.** Does this station physically exist at these coordinates?",
        ["yes", "no", "indeterminate"],
        index=None,
        key="q3",
        help="Use satellite/Street View. Is there visible bike infrastructure?",
    )

with q2:
    a_q2 = st.radio(
        "**Q2.** Does the declared capacity reflect a physical dock count?",
        ["yes", "no", "NaN", "indeterminate"],
        index=None,
        key="q2",
        help="Is capacity a real number of physical docks, or a placeholder/estimator?",
    )

    a_q4 = st.radio(
        "**Q4.** Are these coordinates within the reasonable operating perimeter?",
        ["yes", "no"],
        index=None,
        key="q4",
        help="Is this station geographically consistent with the rest of the network?",
    )

a_q5 = st.radio(
    "**Q5.** Overall verdict:",
    ["clean", "anomaly confirmed", "pipeline false positive", "indeterminate"],
    index=None,
    key="q5",
    horizontal=True,
)

notes = st.text_area(
    "Notes (optional)",
    value="",
    placeholder="Any observation: Street View date, construction site, etc.",
    key="notes",
    height=80,
)

# ── Save ─────────────────────────────────────────────────────────────

all_answered = all([a_q1, a_q2, a_q3, a_q4, a_q5])

col_save, col_skip = st.columns([1, 1])

with col_save:
    if st.button(
        "Save & Next",
        type="primary",
        disabled=not all_answered,
        use_container_width=True,
    ):
        new_row = {
            "system_id": row["system_id"],
            "station_id": row["station_id"],
            "stratum": row["stratum"],
            "Q1_is_bikeshare": a_q1,
            "Q2_capacity_physical": a_q2,
            "Q3_exists_at_coords": a_q3,
            "Q4_within_perimeter": a_q4,
            "Q5_verdict": a_q5,
            "annotator": annotator_id,
            "notes": notes,
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }

        if labels_path.exists():
            existing = pd.read_csv(labels_path)
            updated = pd.concat(
                [existing, pd.DataFrame([new_row])], ignore_index=True
            )
        else:
            updated = pd.DataFrame([new_row])

        updated.to_csv(labels_path, index=False)
        st.rerun()

with col_skip:
    if st.button("Skip (come back later)", use_container_width=True):
        skip_row = {
            "system_id": row["system_id"],
            "station_id": row["station_id"],
            "stratum": row["stratum"],
            "Q1_is_bikeshare": "skipped",
            "Q2_capacity_physical": "skipped",
            "Q3_exists_at_coords": "skipped",
            "Q4_within_perimeter": "skipped",
            "Q5_verdict": "skipped",
            "annotator": annotator_id,
            "notes": "SKIPPED — to revisit",
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }

        if labels_path.exists():
            existing = pd.read_csv(labels_path)
            updated = pd.concat(
                [existing, pd.DataFrame([skip_row])], ignore_index=True
            )
        else:
            updated = pd.DataFrame([skip_row])

        updated.to_csv(labels_path, index=False)
        st.rerun()

if not all_answered:
    st.caption("Answer all 5 questions to enable the Save button.")
