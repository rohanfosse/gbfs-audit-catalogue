"""XP1 — GBFS station_status time-series collector.

Polls station_status endpoints at regular intervals and stores snapshots
as partitioned Parquet files for offline analysis.  Designed to run
unattended for 7–14 days on a modest VM (< 2 GB RAM).

Mathematical rationale
----------------------
A station s is characterised at time t by a state vector:

    v_s(t) = (num_bikes_available, num_docks_available, is_renting,
              is_returning, last_reported)

The collector stores v_s(t) for every station of every monitored system
at each polling epoch.  Downstream (see ``detector.py``), the temporal
variance σ²(v_s) over the observation window W discriminates live
stations from zombies.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATUS_FIELDS = [
    "station_id",
    "num_bikes_available",
    "num_docks_available",
    "is_renting",
    "is_returning",
    "last_reported",
]


async def _fetch_status(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch and parse a single station_status endpoint."""
    resp = await client.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    stations = payload.get("data", {}).get("stations", [])
    return stations


async def collect_snapshot(
    feeds: pd.DataFrame,
    output_dir: Path,
    *,
    timeout_s: float = 30.0,
    max_concurrent: int = 50,
) -> pd.DataFrame:
    """Collect a single epoch snapshot from all feeds.

    Parameters
    ----------
    feeds : DataFrame
        Must contain columns ``system_id`` and ``station_status_url``.
    output_dir : Path
        Root directory; snapshot is written to a date-partitioned subfolder.
    timeout_s : float
        Per-request timeout in seconds.
    max_concurrent : int
        Semaphore limit for concurrent HTTP requests.

    Returns
    -------
    DataFrame
        Concatenated snapshot with columns:
        system_id, station_id, num_bikes_available, num_docks_available,
        is_renting, is_returning, last_reported, collected_at.
    """
    epoch = datetime.now(timezone.utc)
    semaphore = asyncio.Semaphore(max_concurrent)
    rows: list[pd.DataFrame] = []

    async def _guarded_fetch(
        client: httpx.AsyncClient,
        system_id: str,
        url: str,
    ) -> pd.DataFrame | None:
        async with semaphore:
            try:
                stations = await _fetch_status(client, url, timeout=timeout_s)
                if not stations:
                    return None
                df = pd.DataFrame(stations)
                for col in STATUS_FIELDS:
                    if col not in df.columns:
                        df[col] = np.nan
                df = df[STATUS_FIELDS].copy()
                df["system_id"] = system_id
                df["collected_at"] = epoch.isoformat()
                return df
            except Exception as exc:
                logger.warning("Failed %s (%s): %s", system_id, url, exc)
                return None

    async with httpx.AsyncClient(
        follow_redirects=True,
        http2=True,
    ) as client:
        tasks = [
            _guarded_fetch(client, row.system_id, row.station_status_url)
            for row in feeds.itertuples(index=False)
        ]
        results = await asyncio.gather(*tasks)

    rows = [r for r in results if r is not None]
    if not rows:
        logger.error("No feeds returned data at epoch %s", epoch.isoformat())
        return pd.DataFrame()

    snapshot = pd.concat(rows, ignore_index=True)

    partition = epoch.strftime("%Y-%m-%d")
    part_dir = output_dir / f"date={partition}"
    part_dir.mkdir(parents=True, exist_ok=True)
    ts = epoch.strftime("%Y%m%dT%H%M%SZ")
    snapshot.to_parquet(part_dir / f"snapshot_{ts}.parquet", index=False)

    logger.info(
        "Epoch %s: %d stations across %d systems",
        epoch.isoformat(),
        len(snapshot),
        snapshot["system_id"].nunique(),
    )
    return snapshot


async def run_collection_loop(
    feeds: pd.DataFrame,
    output_dir: Path,
    *,
    interval_minutes: int = 15,
    duration_days: int = 14,
    timeout_s: float = 30.0,
    max_concurrent: int = 50,
) -> None:
    """Run the collector for ``duration_days`` at ``interval_minutes`` cadence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = int(duration_days * 24 * 60 / interval_minutes)
    logger.info(
        "Starting collection: %d epochs over %d days (every %d min)",
        total_epochs,
        duration_days,
        interval_minutes,
    )

    for epoch_idx in range(total_epochs):
        t0 = time.monotonic()
        await collect_snapshot(
            feeds,
            output_dir,
            timeout_s=timeout_s,
            max_concurrent=max_concurrent,
        )
        elapsed = time.monotonic() - t0
        sleep_s = max(0, interval_minutes * 60 - elapsed)
        if epoch_idx < total_epochs - 1:
            await asyncio.sleep(sleep_s)

    logger.info("Collection complete: %d epochs written to %s", total_epochs, output_dir)
