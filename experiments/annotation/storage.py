# -*- coding: utf-8 -*-
"""Persistent annotation storage — SQLite backend.

Zero-config for local use (annotations.db next to this file).
For online deployment, three options documented below.

Deployment options
------------------
1. **Local / Streamlit Cloud** : SQLite (default). Set the env var
   ``ANNOTATION_DB_PATH`` to override the database file location
   (e.g. a mounted persistent volume on Railway / Render).
2. **PostgreSQL / Supabase** : set ``ANNOTATION_DB_URL`` to a DSN
   such as ``postgresql://user:pass@host:5432/dbname``.  Requires
   ``psycopg2-binary`` — swap ``_connect()`` and the schema DDL.
3. **Google Sheets** : use ``gspread`` with a service-account JSON.
   Useful for collaborative annotation where both annotators can
   see progress in real time.

The public interface (save / get / export) is backend-agnostic.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_DEFAULT_DB = Path(__file__).resolve().parent / "annotations.db"

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS annotations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT    NOT NULL,
    annotator               TEXT    NOT NULL,
    system_id               TEXT    NOT NULL,
    station_id              TEXT    NOT NULL,
    stratum                 TEXT,
    lat                     REAL,
    lon                     REAL,

    -- Phase 1 : observation terrain
    ground_reality          TEXT,
    infrastructure_elements TEXT,
    streetview_date         TEXT,

    -- Phase 2 : évaluation technique
    capacity_assessment     TEXT,
    location_assessment     TEXT,

    -- Phase 3 : verdict
    verdict                 TEXT    NOT NULL,
    confidence              INTEGER NOT NULL DEFAULT 3,
    notes                   TEXT,

    -- Chrono
    duration_s              REAL,
    created_at              TEXT    NOT NULL,

    -- Pipeline flags (snapshot at annotation time)
    flag_A1 INTEGER DEFAULT 0,
    flag_A2 INTEGER DEFAULT 0,
    flag_A3 INTEGER DEFAULT 0,
    flag_A4 INTEGER DEFAULT 0,
    flag_A5 INTEGER DEFAULT 0,
    flag_A6 INTEGER DEFAULT 0,
    flag_A7 INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_ann_annotator
    ON annotations(annotator);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ann_unique
    ON annotations(annotator, system_id, station_id);
"""


class AnnotationStore:
    """SQLite annotation store with CSV import / export."""

    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path) if db_path else Path(
            os.environ.get("ANNOTATION_DB_PATH", str(_DEFAULT_DB))
        )
        self._path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, row: dict[str, Any]) -> int:
        if isinstance(row.get("infrastructure_elements"), (list, tuple)):
            row = {
                **row,
                "infrastructure_elements": json.dumps(
                    row["infrastructure_elements"], ensure_ascii=False,
                ),
            }
        cols = [
            "session_id", "annotator", "system_id", "station_id", "stratum",
            "lat", "lon",
            "ground_reality", "infrastructure_elements", "streetview_date",
            "capacity_assessment", "location_assessment",
            "verdict", "confidence", "notes",
            "duration_s", "created_at",
            "flag_A1", "flag_A2", "flag_A3", "flag_A4",
            "flag_A5", "flag_A6", "flag_A7",
        ]
        present = {k: row[k] for k in cols if k in row}
        names = ", ".join(present.keys())
        placeholders = ", ".join(["?"] * len(present))
        cur = self._conn.execute(
            f"INSERT OR REPLACE INTO annotations ({names}) VALUES ({placeholders})",
            list(present.values()),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_done_keys(self, annotator: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT system_id, station_id FROM annotations WHERE annotator = ?",
            (annotator,),
        ).fetchall()
        return {f"{r['system_id']}|{r['station_id']}" for r in rows}

    def get_annotation(
        self, annotator: str, system_id: str, station_id: str,
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM annotations "
            "WHERE annotator = ? AND system_id = ? AND station_id = ?",
            (annotator, system_id, station_id),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("infrastructure_elements"):
            try:
                d["infrastructure_elements"] = json.loads(
                    d["infrastructure_elements"],
                )
            except (json.JSONDecodeError, TypeError):
                d["infrastructure_elements"] = []
        return d

    def get_all(self, annotator: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT * FROM annotations WHERE annotator = ? ORDER BY created_at",
            self._conn,
            params=(annotator,),
        )
        if "infrastructure_elements" in df.columns:
            df["infrastructure_elements"] = df["infrastructure_elements"].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x else [],
            )
        return df

    def count(self, annotator: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM annotations WHERE annotator = ?",
            (annotator,),
        ).fetchone()
        return row["n"] if row else 0

    def count_non_skipped(self, annotator: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM annotations "
            "WHERE annotator = ? AND verdict != 'skipped'",
            (annotator,),
        ).fetchone()
        return row["n"] if row else 0

    def median_duration(self, annotator: str) -> float | None:
        rows = self._conn.execute(
            "SELECT duration_s FROM annotations "
            "WHERE annotator = ? AND verdict != 'skipped' AND duration_s > 0 "
            "ORDER BY duration_s",
            (annotator,),
        ).fetchall()
        if not rows:
            return None
        vals = [r["duration_s"] for r in rows]
        n = len(vals)
        if n % 2:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, annotator: str, path: Path) -> None:
        df = self.get_all(annotator)
        if "infrastructure_elements" in df.columns:
            df["infrastructure_elements"] = df["infrastructure_elements"].apply(
                lambda x: "|".join(x) if isinstance(x, list) else str(x),
            )
        df.to_csv(path, index=False)

    def export_legacy_csv(self, annotator: str, path: Path) -> None:
        """Q1–Q5 format compatible with ``compute_reliability.py``."""
        df = self.get_all(annotator)
        if df.empty:
            pd.DataFrame().to_csv(path, index=False)
            return

        out = pd.DataFrame()
        out["system_id"] = df["system_id"]
        out["station_id"] = df["station_id"]
        out["stratum"] = df["stratum"]
        out["lat"] = df["lat"]
        out["lon"] = df["lon"]

        q1 = {
            "station_vls": "oui",
            "trottinettes": "non",
            "autopartage": "non",
            "aucune_infrastructure": "non",
            "indetermine": "indéterminé",
        }
        out["Q1_is_bikeshare"] = df["ground_reality"].map(q1).fillna("indéterminé")

        q2 = {
            "coherente": "oui",
            "placeholder": "non",
            "champ_vide": "NaN (champ vide)",
            "zero_suspect": "non",
            "impossible": "indéterminé",
        }
        out["Q2_capacity_physical"] = (
            df["capacity_assessment"].map(q2).fillna("indéterminé")
        )

        def _q3(elems: Any) -> str:
            if isinstance(elems, list):
                if not elems or elems == ["rien_visible"]:
                    return "non"
                return "oui"
            return "indéterminé"

        out["Q3_exists_at_coords"] = df["infrastructure_elements"].apply(_q3)

        q4 = {
            "integree_reseau": "oui",
            "isolee_legitime": "oui",
            "isolee_suspecte": "non",
            "coordonnees_erronees": "non",
        }
        out["Q4_within_perimeter"] = (
            df["location_assessment"].map(q4).fillna("oui")
        )

        # Pipeline-agnostic ground-truth label (no reference to the pipeline).
        q5 = {
            "legitime": "vraie station (légitime)",
            "problematique": "station problématique",
            "indetermine": "indéterminé",
            "skipped": "skipped",
        }
        out["Q5_verdict"] = df["verdict"].map(q5).fillna(df["verdict"])

        out["annotator"] = df["annotator"]
        out["notes"] = df["notes"]
        out["duration_s"] = df["duration_s"]
        out["annotated_at"] = df["created_at"]
        out.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Import legacy
    # ------------------------------------------------------------------

    def import_legacy_csv(self, path: Path, session_id: str = "imported") -> int:
        """Import from a Q1–Q5 CSV.  Returns count of new rows."""
        df = pd.read_csv(path)
        n = 0
        for _, r in df.iterrows():
            ann = str(r.get("annotator", "unknown"))
            sid, stid = str(r["system_id"]), str(r["station_id"])
            if self.get_annotation(ann, sid, stid):
                continue

            q1 = str(r.get("Q1_is_bikeshare", ""))
            ground = (
                "station_vls" if q1 == "oui"
                else "aucune_infrastructure" if q1 == "non"
                else "indetermine"
            )
            q2 = str(r.get("Q2_capacity_physical", ""))
            cap = (
                "coherente" if q2 == "oui"
                else "champ_vide" if "NaN" in q2
                else "placeholder" if q2 == "non"
                else "impossible"
            )
            q4 = str(r.get("Q4_within_perimeter", ""))
            loc = "integree_reseau" if q4 == "oui" else "isolee_suspecte"

            # Map legacy verdicts to the agnostic scheme. The old 4-class
            # labels referenced the pipeline; we translate to the factual
            # state: "faux positif" meant the station was actually fine
            # (legitime), "anomalie confirmée" meant it was problematic.
            v = str(r.get("Q5_verdict", ""))
            verdict = (
                "skipped" if "skipped" in v
                else "legitime" if ("propre" in v or "vraie station" in v
                                    or "faux positif" in v)
                else "problematique" if ("problématique" in v or "anomalie" in v)
                else "indetermine"
            )

            self.save({
                "session_id": session_id,
                "annotator": ann,
                "system_id": sid,
                "station_id": stid,
                "stratum": r.get("stratum"),
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "ground_reality": ground,
                "infrastructure_elements": [],
                "capacity_assessment": cap,
                "location_assessment": loc,
                "verdict": verdict,
                "confidence": 3,
                "notes": str(r.get("notes", "") or ""),
                "duration_s": float(r.get("duration_s", 0) or 0),
                "created_at": str(
                    r.get(
                        "annotated_at",
                        datetime.now(timezone.utc).isoformat(),
                    )
                ),
            })
            n += 1
        return n

    def close(self) -> None:
        self._conn.close()
