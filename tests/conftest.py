"""Shared synthetic fixtures for the GBFS Audit Catalogue test suite.

Each fixture below is a deterministic in-memory ``DataFrame`` engineered
to trigger exactly one anomaly class. Combining them via ``pd.concat``
yields a multi-system corpus that exercises every branch of
``audit_pipeline.core``.

Coordinates are chosen inside metropolitan France (lat 43..50, lon
-1..7) so the equirectangular projection in ``_project_meters`` stays
locally accurate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _grid(
    system_id: str,
    *,
    n: int,
    centre_lat: float,
    centre_lon: float,
    spacing_deg: float = 0.005,
    station_type: str = "docked_bike",
    capacity: float | int | None = 25,
    system_name: str | None = None,
) -> pd.DataFrame:
    """Build a square-ish grid of stations around a city centre."""
    side = int(np.ceil(np.sqrt(n)))
    rows = []
    for k in range(n):
        i, j = divmod(k, side)
        rows.append(
            {
                "system_id": system_id,
                "system_name": system_name or system_id,
                "station_id": f"{system_id}-{k:04d}",
                "lat": centre_lat + i * spacing_deg,
                "lon": centre_lon + j * spacing_deg,
                "station_type": station_type,
                "capacity": capacity,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-anomaly fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_system() -> pd.DataFrame:
    """25 healthy docked stations. Must end up with no flag and high confidence."""
    df = _grid("paris-clean", n=25, centre_lat=48.8566, centre_lon=2.3522)
    df["capacity"] = np.random.default_rng(0).integers(10, 40, size=len(df))
    return df


@pytest.fixture
def a1_carsharing() -> pd.DataFrame:
    """All stations declare station_type = 'carsharing' -> A1 row-level.

    Capacity is varied across the system so A2 (constant capacity) does
    not co-fire on a purely A1 fixture.
    """
    df = _grid(
        "lyon-citiz",
        n=25,
        centre_lat=45.7640,
        centre_lon=4.8357,
        station_type="carsharing",
    )
    df["capacity"] = np.arange(3, 3 + len(df))
    return df


@pytest.fixture
def a2_placeholder() -> pd.DataFrame:
    """Constant non-zero capacity across the whole system -> A2 system-level."""
    return _grid(
        "marseille-placeholder",
        n=25,
        centre_lat=43.2965,
        centre_lon=5.3698,
        capacity=20,
    )


@pytest.fixture
def a3_free_floating() -> pd.DataFrame:
    """All stations declare station_type = 'free_floating' -> A3 row-level."""
    return _grid(
        "toulouse-pony",
        n=25,
        centre_lat=43.6047,
        centre_lon=1.4442,
        station_type="free_floating",
        capacity=1,
    )


@pytest.fixture
def a4_geospatial_outlier() -> pd.DataFrame:
    """24 stations clustered in Bordeaux + 1 transposed (lat/lon swap) -> A4."""
    df = _grid("bordeaux-vcub", n=24, centre_lat=44.8378, centre_lon=-0.5792)
    df.loc[len(df)] = {
        "system_id": "bordeaux-vcub",
        "system_name": "bordeaux-vcub",
        "station_id": "bordeaux-vcub-OUTLIER",
        "lat": -0.5792,  # swapped
        "lon": 44.8378,  # swapped
        "station_type": "docked_bike",
        "capacity": 18,
    }
    return df


@pytest.fixture
def a5_huge_bbox() -> pd.DataFrame:
    """Bounding box > 50,000 km^2 -> A5 system-level for every station."""
    # 6 degrees of latitude ~ 667 km, 6 degrees of longitude ~ 450 km at
    # 45 N -> ~300,000 km^2 bbox.
    coords = [
        (43.0, -1.0),
        (49.0, 5.0),
        (43.5, 4.5),
        (48.5, -0.5),
    ]
    rows = []
    for k, (lat, lon) in enumerate(coords * 8):  # 32 stations
        rows.append(
            {
                "system_id": "phantom-national",
                "system_name": "phantom-national",
                "station_id": f"phantom-national-{k:04d}",
                "lat": lat + 0.01 * k,
                "lon": lon + 0.01 * k,
                "station_type": "docked_bike",
                "capacity": 15 + k,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def a6_zero_capacity() -> pd.DataFrame:
    """25 docked stations, 3 with capacity = 0 (12%) -> A6 system-level."""
    df = _grid(
        "nantes-bicloo",
        n=25,
        centre_lat=47.2184,
        centre_lon=-1.5536,
    )
    df["capacity"] = [25] * len(df)
    df.loc[:2, "capacity"] = 0  # 3 / 25 = 12% >= 1%
    return df


@pytest.fixture
def a7_null_capacity() -> pd.DataFrame:
    """25 docked stations, 60% with capacity = NaN -> A7 system-level."""
    df = _grid(
        "nice-velobleu",
        n=25,
        centre_lat=43.7102,
        centre_lon=7.2620,
    )
    df["capacity"] = df["capacity"].astype("float64")
    df.loc[:14, "capacity"] = np.nan  # 15/25 = 60% >= 50%
    return df


@pytest.fixture
def multi_system_corpus(
    clean_system,
    a1_carsharing,
    a2_placeholder,
    a3_free_floating,
    a4_geospatial_outlier,
    a5_huge_bbox,
    a6_zero_capacity,
    a7_null_capacity,
) -> pd.DataFrame:
    return pd.concat(
        [
            clean_system,
            a1_carsharing,
            a2_placeholder,
            a3_free_floating,
            a4_geospatial_outlier,
            a5_huge_bbox,
            a6_zero_capacity,
            a7_null_capacity,
        ],
        ignore_index=True,
    )
