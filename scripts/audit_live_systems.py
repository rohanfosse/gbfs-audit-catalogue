"""Live audit of a panel of foreign GBFS systems for the E5
generalisation experiment of the paper.

Given a CSV of (country, name, auto_discovery_url), the script:
1. fetches each system's auto-discovery feed
2. resolves the station_information.json URL
3. fetches station_information.json
4. extracts (station_id, lat, lon, capacity)
5. applies the seven Tier-1 detectors of audit_pipeline.core
6. writes a per-system summary CSV with the A1-A7 verdicts

Designed for read-only validation: no parquet is mutated, no global
audit state is touched. Failures (timeouts, schema mismatches, paywalls)
are surfaced explicitly rather than swallowed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from audit_pipeline.core import _compute_tier1  # noqa: E402

TIMEOUT = 20.0
USER_AGENT = (
    "GBFS-Audit-Catalogue/1.0.1 (Fossé & Pallares 2026 CSI E5 panel)"
)


def _fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def _resolve_station_information_url(disco_url: str) -> str:
    disco = _fetch(disco_url)
    feeds_root = disco.get("data", {})
    # GBFS v1/v2 nests by language; GBFS v3 surfaces feeds directly
    for key, val in feeds_root.items():
        feeds = val.get("feeds") if isinstance(val, dict) else None
        if feeds is None and isinstance(val, list):
            feeds = val
        if not feeds:
            continue
        for f in feeds:
            if f.get("name") == "station_information":
                return f["url"]
    raise LookupError(f"no station_information feed in {disco_url}")


def _audit_one(name: str, country: str, disco_url: str) -> dict:
    t0 = time.monotonic()
    try:
        si_url = _resolve_station_information_url(disco_url)
        si = _fetch(si_url)
    except Exception as exc:  # noqa: BLE001 — we surface every failure
        return {
            "name": name,
            "country": country,
            "status": f"fetch_error: {type(exc).__name__}",
            "url": disco_url,
            "elapsed_s": round(time.monotonic() - t0, 1),
        }

    stations = si.get("data", {}).get("stations", [])
    if len(stations) < 20:
        return {
            "name": name,
            "country": country,
            "status": "too_small_below_Nmin",
            "n_stations": len(stations),
            "url": disco_url,
            "elapsed_s": round(time.monotonic() - t0, 1),
        }

    rows = []
    sid = f"e5__{country.lower()}__{name.lower().replace(' ', '_')}"
    for s in stations:
        cap = s.get("capacity")
        # Treat None / missing as NaN for the audit; the detectors rely on it.
        if cap is None:
            cap = np.nan
        rows.append(
            {
                "uid": f"{sid}::{s.get('station_id', '?')}",
                "system_id": sid,
                "system_name": name,
                "station_id": str(s.get("station_id", "")),
                "station_type": "docked_bike",  # E5 panel = known dock-based
                "lat": float(s.get("lat", float("nan"))),
                "lon": float(s.get("lon", float("nan"))),
                "capacity": float(cap),
            }
        )
    df = pd.DataFrame(rows)

    enriched = _compute_tier1(df)

    # Distinguish system-level flags (A2, A5, A6, A7) from row-level
    # (A1, A3, A4): for row-level, report the count rather than the
    # boolean to expose the difference between "one isolated outlier"
    # and "structural over-capacity".
    flags = {}
    for i in (1, 3, 4):  # row-level
        n_flagged = int(enriched[f"flag_A{i}"].sum())
        flags[f"A{i}_n_stations"] = n_flagged
        flags[f"A{i}_share_pct"] = round(100.0 * n_flagged / len(enriched), 2)
    for i in (2, 5, 6, 7):  # system-level
        flags[f"A{i}_flagged"] = bool(enriched[f"flag_A{i}"].any())
    return {
        "name": name,
        "country": country,
        "status": "ok",
        "n_stations": len(stations),
        "n_capacity_nan": int(df["capacity"].isna().sum()),
        "n_capacity_zero": int((df["capacity"] == 0).sum()),
        "capacity_median": float(df["capacity"].median()) if df["capacity"].notna().any() else float("nan"),
        "capacity_unique_values": int(df["capacity"].nunique(dropna=True)),
        **flags,
        "url": disco_url,
        "elapsed_s": round(time.monotonic() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("experiments/e5_europe/panel.csv"),
        help="CSV with columns: country, name, auto_discovery_url",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/e5_europe/results.csv"),
        help="Where to write per-system audit results",
    )
    args = parser.parse_args()

    panel_rows = []
    with args.panel.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("auto_discovery_url"):
                continue
            panel_rows.append(row)

    results = []
    for r in panel_rows:
        safe_name = r["name"].encode("ascii", errors="replace").decode("ascii")
        print(f"[*] {r['country']} {safe_name}  ...", end="", flush=True)
        out = _audit_one(r["name"], r["country"], r["auto_discovery_url"])
        results.append(out)
        print(f" {out['status']} ({out.get('elapsed_s', '?')}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        print("\nE5 panel summary:")
        for r in ok:
            row_flags = []
            for i in (1, 3, 4):
                n = r.get(f"A{i}_n_stations", 0)
                if n:
                    row_flags.append(f"A{i}={n} ({r[f'A{i}_share_pct']}%)")
            sys_flags = [
                f"A{i}" for i in (2, 5, 6, 7) if r.get(f"A{i}_flagged")
            ]
            tag = ", ".join(row_flags + sys_flags) or "none"
            safe_name = r["name"].encode("ascii", errors="replace").decode("ascii")
            print(f"  {r['country']} {safe_name:30s} n={r['n_stations']:5d}  {tag}")


if __name__ == "__main__":
    main()
