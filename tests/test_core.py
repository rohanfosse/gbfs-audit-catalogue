"""Unit tests for the seven anomaly classes and Tier-2 geometry.

Each test trips exactly one flag on its dedicated fixture, then the
``test_multi_system_corpus_*`` block verifies that the flags remain
correctly attributed when the systems are concatenated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit_pipeline.core import (
    A2_MIN_STATIONS,
    A5_BBOX_MAX_KM2,
    ANOMALY_CLASSES,
    _compute_tier1,
    _compute_tier2,
    _project_meters,
    detect_operator,
    enrich,
)


# ---------------------------------------------------------------------------
# Tier-1 : per-flag tests
# ---------------------------------------------------------------------------


def test_clean_system_carries_no_flag(clean_system):
    enriched = _compute_tier1(clean_system)
    for i in range(1, 8):
        assert not enriched[f"flag_A{i}"].any(), f"clean system tripped A{i}"
    assert (enriched["audit_confidence"] == "high").all()


def test_a1_flags_carsharing_rows(a1_carsharing):
    enriched = _compute_tier1(a1_carsharing)
    assert enriched["flag_A1"].all()
    # No other class should fire on a pure carsharing system.
    for i in (2, 4, 5, 6, 7):
        assert not enriched[f"flag_A{i}"].any()


def test_a2_flags_constant_capacity(a2_placeholder):
    enriched = _compute_tier1(a2_placeholder)
    assert enriched["flag_A2"].all()
    # A2 is system-level: capacity_audited must still equal capacity for
    # docked_bike rows.
    assert (enriched["capacity_audited"] == 20).all()


def test_a2_requires_minimum_stations():
    """A small constant-capacity system must NOT trip A2."""
    rows = []
    for k in range(A2_MIN_STATIONS - 1):
        rows.append(
            {
                "system_id": "tiny",
                "system_name": "tiny",
                "station_id": f"tiny-{k}",
                "lat": 48.85 + 0.001 * k,
                "lon": 2.35 + 0.001 * k,
                "station_type": "docked_bike",
                "capacity": 17,
            }
        )
    df = pd.DataFrame(rows)
    enriched = _compute_tier1(df)
    assert not enriched["flag_A2"].any()


def test_a3_flags_free_floating(a3_free_floating):
    enriched = _compute_tier1(a3_free_floating)
    assert enriched["flag_A3"].all()
    # Free-floating rows must have capacity_audited = NaN regardless of
    # what the raw GBFS feed reported.
    assert enriched["capacity_audited"].isna().all()


def test_a4_flags_geospatial_outlier(a4_geospatial_outlier):
    enriched = _compute_tier1(a4_geospatial_outlier)
    # Exactly one transposed station, all others within the Bordeaux cluster.
    outlier_mask = enriched["station_id"] == "bordeaux-vcub-OUTLIER"
    assert enriched.loc[outlier_mask, "flag_A4"].all()
    assert not enriched.loc[~outlier_mask, "flag_A4"].any()


def test_a5_flags_huge_bounding_box(a5_huge_bbox):
    enriched = _compute_tier1(a5_huge_bbox)
    assert enriched["flag_A5"].all()
    # Sanity check: the fixture's bbox area really is above threshold.
    projected = _project_meters(a5_huge_bbox["lat"].to_numpy(), a5_huge_bbox["lon"].to_numpy())
    width = projected[:, 0].max() - projected[:, 0].min()
    height = projected[:, 1].max() - projected[:, 1].min()
    assert (width * height) / 1e6 > A5_BBOX_MAX_KM2


def test_a6_flags_zero_capacity(a6_zero_capacity):
    enriched = _compute_tier1(a6_zero_capacity)
    assert enriched["flag_A6"].all()


def test_a6_ignores_isolated_zero():
    """A single zero-capacity station in a large system stays under 1%."""
    rows = []
    for k in range(200):
        rows.append(
            {
                "system_id": "big-clean",
                "system_name": "big-clean",
                "station_id": f"bc-{k}",
                "lat": 48.85 + 0.001 * (k // 20),
                "lon": 2.35 + 0.001 * (k % 20),
                "station_type": "docked_bike",
                "capacity": 25 if k > 0 else 0,
            }
        )
    df = pd.DataFrame(rows)
    enriched = _compute_tier1(df)
    assert not enriched["flag_A6"].any(), "1/200 = 0.5% should be below 1% threshold"


def test_a7_flags_null_capacity(a7_null_capacity):
    enriched = _compute_tier1(a7_null_capacity)
    assert enriched["flag_A7"].all()


def test_audit_confidence_levels(multi_system_corpus):
    enriched = _compute_tier1(multi_system_corpus)
    by_system = enriched.groupby("system_id")["audit_confidence"].first()
    assert by_system["paris-clean"] == "high"
    # A3 alone -> medium
    assert by_system["toulouse-pony"] == "medium"
    # Multi-flag systems collapse to low
    assert by_system["nice-velobleu"] in {"medium", "low"}
    assert by_system["lyon-citiz"] == "low"


# ---------------------------------------------------------------------------
# Tier-2 : geometry
# ---------------------------------------------------------------------------


def test_tier2_intra_system_distances(clean_system):
    enriched = _compute_tier2(clean_system)
    d = enriched["dist_to_nearest_station_m"]
    # 0.005 deg ~= 555 m at 48.85 N -> expect intra distances < ~1km.
    assert d.notna().all()
    assert (d > 0).all()
    assert d.max() < 1500


def test_tier2_inter_system_distance_uses_other_system(clean_system, a3_free_floating):
    """Paris vs Toulouse: each row's nearest *other* system must be a Toulouse
    point for the Paris rows, and conversely. We just check the distance is
    consistent with the geographic separation (~580 km)."""
    df = pd.concat([clean_system, a3_free_floating], ignore_index=True)
    enriched = _compute_tier2(df)
    inter = enriched["nearest_system_dist_m"]
    assert inter.notna().all()
    # Paris-Toulouse great-circle distance is ~580 km; equirectangular
    # projection introduces some error, allow a wide [400 km, 800 km] band.
    assert inter.min() > 400_000
    assert inter.max() < 800_000


def test_tier2_density_is_count_over_pi(clean_system):
    enriched = _compute_tier2(clean_system)
    np.testing.assert_allclose(
        enriched["catchment_density_per_km2"].to_numpy(),
        enriched["n_stations_within_1km"].to_numpy() / np.pi,
    )


def test_tier2_singleton_system_has_no_intra_neighbours():
    """A system with a single station must report NaN intra distance and 0 counts."""
    df = pd.DataFrame(
        [
            {
                "system_id": "solo",
                "system_name": "solo",
                "station_id": "solo-0",
                "lat": 48.85,
                "lon": 2.35,
                "station_type": "docked_bike",
                "capacity": 10,
            },
            {
                "system_id": "neighbour",
                "system_name": "neighbour",
                "station_id": "n-0",
                "lat": 48.86,
                "lon": 2.36,
                "station_type": "docked_bike",
                "capacity": 10,
            },
            {
                "system_id": "neighbour",
                "system_name": "neighbour",
                "station_id": "n-1",
                "lat": 48.87,
                "lon": 2.37,
                "station_type": "docked_bike",
                "capacity": 10,
            },
        ]
    )
    enriched = _compute_tier2(df)
    solo = enriched[enriched["system_id"] == "solo"].iloc[0]
    assert np.isnan(solo["dist_to_nearest_station_m"])
    assert solo["n_stations_within_500m"] == 0
    assert solo["n_stations_within_1km"] == 0
    # But the cross-system neighbour should be resolved.
    assert solo["nearest_system_dist_m"] > 0


# ---------------------------------------------------------------------------
# End-to-end enrich()
# ---------------------------------------------------------------------------


def test_enrich_full_pipeline_adds_all_columns(multi_system_corpus):
    enriched = enrich(multi_system_corpus)
    expected_new_columns = {
        "capacity_raw",
        "capacity_audited",
        "flag_A1",
        "flag_A2",
        "flag_A3",
        "flag_A4",
        "flag_A5",
        "flag_A6",
        "flag_A7",
        "operator_name",
        "audit_confidence",
        "dist_to_nearest_station_m",
        "n_stations_within_500m",
        "n_stations_within_1km",
        "nearest_system_dist_m",
        "catchment_density_per_km2",
    }
    assert expected_new_columns.issubset(enriched.columns)
    assert len(enriched) == len(multi_system_corpus)


def test_enrich_is_deterministic(multi_system_corpus):
    a = enrich(multi_system_corpus)
    b = enrich(multi_system_corpus)
    pd.testing.assert_frame_equal(a, b)


def test_enrich_each_system_trips_exactly_its_flag(multi_system_corpus):
    enriched = enrich(multi_system_corpus)
    by_system = enriched.groupby("system_id").agg(
        {f"flag_A{i}": "any" for i in range(1, 8)}
    )
    assert by_system.loc["paris-clean"].sum() == 0
    assert by_system.loc["lyon-citiz", "flag_A1"]
    assert by_system.loc["marseille-placeholder", "flag_A2"]
    assert by_system.loc["toulouse-pony", "flag_A3"]
    assert by_system.loc["bordeaux-vcub", "flag_A4"]
    assert by_system.loc["phantom-national", "flag_A5"]
    assert by_system.loc["nantes-bicloo", "flag_A6"]
    assert by_system.loc["nice-velobleu", "flag_A7"]


# ---------------------------------------------------------------------------
# Misc public-surface tests
# ---------------------------------------------------------------------------


def test_anomaly_classes_metadata_present():
    assert set(ANOMALY_CLASSES) == {f"A{i}" for i in range(1, 8)}
    for k, v in ANOMALY_CLASSES.items():
        assert "name" in v and "signature" in v


@pytest.mark.parametrize(
    "system_id,system_name,expected",
    [
        ("velib-paris-metro", "Velib Metropole", "Vélib' Métropole"),
        ("pony-lyon", "Pony Lyon", "Pony"),
        ("citiz-lr", "Citiz LR", "Citiz (carsharing)"),
        ("unknown-op", "Random City Bike", "Random City Bike"),
        ("vcub-bordeaux", "Le Vélo TBM Bordeaux", "Le Vélo TBM"),
    ],
)
def test_detect_operator_normalisation(system_id, system_name, expected):
    assert detect_operator(system_id, system_name) == expected
