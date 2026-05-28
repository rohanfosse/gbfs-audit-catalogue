"""Generate machine-readable metadata manifests for the catalogue.

Reads the certified parquet and emits, into ``catalogue/metadata/``:

- ``schema.json``      — JSON Schema (Draft 2020-12) for one station record
- ``datapackage.json`` — Frictionless Tabular Data Package descriptor
- ``dcat-ap.jsonld``   — DCAT-AP dataset record (JSON-LD)
- ``croissant.jsonld`` — ML Commons Croissant manifest (JSON-LD)

These are the FAIR descriptors referenced in the manuscript. Regenerate
after any schema change:

    python -m scripts.generate_metadata
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "catalogue" / "stations_gold_standard_final.parquet"
OUT_DIR = REPO / "catalogue" / "metadata"

# --- Dataset-level metadata (mirrors CITATION.cff) -------------------
DATASET = {
    "name": "gbfs-audit-catalogue",
    "title": "GBFS Audit Catalogue",
    "description": (
        "A certified, anomaly-flagged reference dataset of 46,307 "
        "bike-sharing stations across 123 French operators, with per-row "
        "audit verdicts (seven data-quality classes A1–A7) and contextual "
        "enrichment from INSEE, BAAC, BD TOPO, BD ALTI and the national "
        "GTFS aggregator."
    ),
    "version": "1.0.1",
    "issued": "2026-05-13",
    "doi": "10.5281/zenodo.20125460",
    "homepage": "https://gbfs-audit.streamlit.app",
    "repository": "https://github.com/cycling-data-lab/gbfs-audit-catalogue",
    "zenodo_file": (
        "https://zenodo.org/records/20125460/files/"
        "stations_gold_standard_final.parquet"
    ),
    "license_id": "ODbL-1.0",
    "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
    "license_title": "Open Database License (ODbL) v1.0",
    "keywords": [
        "GBFS", "bike-sharing", "open mobility data", "data quality",
        "FAIR", "standards compliance", "reproducibility", "smart city",
    ],
    "creators": [
        {"name": "Fossé, Rohan", "affiliation": "CESI École d'Ingénieurs, Montpellier, France"},
        {"name": "Pallares, Gaël", "affiliation": "CESI LINEACT (EA 7527), Montpellier, France"},
    ],
    "cite_as": (
        "Fossé, R. & Pallares, G. (2026). GBFS Audit Catalogue v1.0. "
        "Zenodo. https://doi.org/10.5281/zenodo.20125460"
    ),
}

# --- Per-column human descriptions (one per parquet column) ----------
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "uid": "Audit Catalogue primary key (stable per certified station).",
    "system_id": "Operator-system identifier.",
    "city": "Normalised city name.",
    "system_name": "Operator-system label as published in the feed.",
    "source": "Feed URL or canonical-catalogue source pointer.",
    "station_id": "GBFS native station identifier.",
    "station_name": "GBFS station name.",
    "lat": "Geofiltered WGS84 latitude (decimal degrees).",
    "lon": "Geofiltered WGS84 longitude (decimal degrees).",
    "capacity": "Raw declared dock capacity (may be a placeholder or null).",
    "address": "GBFS-declared street address, when present.",
    "region_id": "Administrative region identifier, when present.",
    "n_stations_system": "Total number of stations in the parent system.",
    "fetched_at": "UTC timestamp of the audited feed snapshot.",
    "elevation_m": "Ground elevation in metres (IGN BD ALTI).",
    "topography_roughness_index": "Local relief amplitude around the station.",
    "infra_cyclable_km": "Cycle-lane linear within a 300 m buffer (IGN BD TOPO), km.",
    "infra_cyclable_pct": "Share of dedicated cycle right-of-way in the buffer.",
    "baac_accidents_cyclistes": "Severe cyclist-crash count within 500 m over 5 years (ONISR BAAC).",
    "gtfs_heavy_stops_300m": "Heavy-transit stops within 300 m (national GTFS aggregator).",
    "gtfs_stops_within_300m_pct": "Share of accessible heavy-transit stops in the buffer.",
    "code_commune": "INSEE commune code.",
    "commune_name": "INSEE commune label.",
    "revenu_median_uc": "INSEE Filosofi median disposable income per consumption unit.",
    "gini_revenu": "Local Gini index of income.",
    "revenu_d1": "First-decile income (INSEE Filosofi).",
    "ecart_interquar": "Interquartile income spread (INSEE Filosofi).",
    "part_menages_voit0": "Share of car-less households (INSEE).",
    "part_velo_travail": "Share of commuting by bicycle (INSEE).",
    "station_type": "Audited station type: docked_bike, free_floating or carsharing.",
    "capacity_raw": "Raw GBFS capacity preserved verbatim (NaN/placeholders kept).",
    "capacity_audited": "Post-audit capacity; NaN for non-dock types.",
    "flag_A1": "Structural error A1: out-of-domain inclusion (carsharing under a bike schema).",
    "flag_A2": "Structural error A2: placeholder capacity constant across a system.",
    "flag_A3": "Structural error A3: structural over-capacity (free-floating anchor).",
    "flag_A4": "Structural error A4: topology-aware geospatial outlier (HDBSCAN + spectral).",
    "flag_A5": "Structural error A5: out-of-perimeter coverage (bbox > 50,000 km²).",
    "flag_A6": "Semantic warning A6: zero-capacity dock.",
    "flag_A7": "Semantic warning A7: null-capacity field.",
    "operator_name": "Normalised operator label.",
    "audit_confidence": "Per-row audit confidence: high, medium or low.",
    "dist_to_nearest_station_m": "Intra-system distance to nearest station (metres).",
    "n_stations_within_500m": "Intra-system station count within 500 m.",
    "n_stations_within_1km": "Intra-system station count within 1 km.",
    "nearest_system_dist_m": "Distance to the nearest station of a different system (metres).",
    "catchment_density_per_km2": "Stations per km² within a 1 km buffer.",
}


def _types(dtype: str) -> tuple[str, str, str, bool]:
    """Return (json_schema_type, frictionless_type, croissant_dataType, is_datetime)."""
    d = dtype.lower()
    if "datetime" in d:
        return "string", "datetime", "sc:Date", True
    if "bool" in d:
        return "boolean", "boolean", "sc:Boolean", False
    if "int" in d:
        return "integer", "integer", "sc:Integer", False
    if "float" in d:
        return "number", "number", "sc:Float", False
    return "string", "string", "sc:Text", False


def _columns() -> list[dict]:
    df = pd.read_parquet(PARQUET)
    cols = []
    for name in df.columns:
        dtype = str(df[name].dtype)
        js, fr, cr, is_dt = _types(dtype)
        completeness = float(df[name].notna().mean())
        cols.append({
            "name": name,
            "dtype": dtype,
            "js_type": js,
            "fr_type": fr,
            "cr_type": cr,
            "is_datetime": is_dt,
            "required": completeness >= 0.999999,
            "completeness": round(completeness, 4),
            "description": COLUMN_DESCRIPTIONS.get(name, name),
        })
    return cols


def _file_stats() -> tuple[int, str]:
    data = PARQUET.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def build_json_schema(cols: list[dict]) -> dict:
    props = {}
    for c in cols:
        if c["is_datetime"]:
            spec: dict = {"type": "string", "format": "date-time"}
        else:
            spec = {"type": c["js_type"]}
        if not c["required"]:
            spec["type"] = [spec["type"], "null"]
        spec["description"] = c["description"]
        props[c["name"]] = spec
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{DATASET['repository']}/blob/main/catalogue/metadata/schema.json",
        "title": "GBFS Audit Catalogue — station record",
        "description": DATASET["description"],
        "type": "object",
        "properties": props,
        "required": [c["name"] for c in cols if c["required"]],
        "additionalProperties": False,
    }


def build_datapackage(cols: list[dict], n_bytes: int, sha256: str) -> dict:
    fields = [{
        "name": c["name"],
        "type": c["fr_type"],
        "description": c["description"],
        "constraints": {"required": c["required"]},
    } for c in cols]
    return {
        "profile": "tabular-data-package",
        "name": DATASET["name"],
        "id": f"https://doi.org/{DATASET['doi']}",
        "title": DATASET["title"],
        "description": DATASET["description"],
        "version": DATASET["version"],
        "created": DATASET["issued"],
        "homepage": DATASET["homepage"],
        "keywords": DATASET["keywords"],
        "licenses": [{
            "name": DATASET["license_id"],
            "path": DATASET["license_url"],
            "title": DATASET["license_title"],
        }],
        "contributors": [
            {"title": c["name"], "organization": c["affiliation"], "role": "author"}
            for c in DATASET["creators"]
        ],
        "resources": [{
            "name": "stations_gold_standard_final",
            "path": "../stations_gold_standard_final.parquet",
            "format": "parquet",
            "mediatype": "application/vnd.apache.parquet",
            "profile": "data-resource",
            "bytes": n_bytes,
            "hash": f"sha256:{sha256}",
            "schema": {
                "fields": fields,
                "primaryKey": "uid",
            },
        }],
    }


def build_dcat_ap(n_bytes: int, sha256: str) -> dict:
    today = dt.date.today().isoformat()
    return {
        "@context": {
            "dcat": "http://www.w3.org/ns/dcat#",
            "dct": "http://purl.org/dc/terms/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "spdx": "http://spdx.org/rdf/terms#",
        },
        "@type": "dcat:Dataset",
        "dct:title": DATASET["title"],
        "dct:description": DATASET["description"],
        "dct:identifier": f"https://doi.org/{DATASET['doi']}",
        "dct:issued": DATASET["issued"],
        "dct:modified": today,
        "dct:publisher": {"@type": "foaf:Organization", "foaf:name": "CESI LINEACT (EA 7527)"},
        "dct:creator": [{"@type": "foaf:Person", "foaf:name": c["name"]} for c in DATASET["creators"]],
        "dct:license": {"@id": DATASET["license_url"]},
        "dcat:keyword": DATASET["keywords"],
        "dcat:landingPage": {"@id": DATASET["homepage"]},
        "dcat:distribution": [{
            "@type": "dcat:Distribution",
            "dct:title": "Certified catalogue (Apache Parquet)",
            "dcat:downloadURL": {"@id": DATASET["zenodo_file"]},
            "dcat:mediaType": "application/vnd.apache.parquet",
            "dcat:byteSize": n_bytes,
            "spdx:checksum": {"spdx:algorithm": "spdx:checksumAlgorithm_sha256",
                              "spdx:checksumValue": sha256},
            "dct:license": {"@id": DATASET["license_url"]},
        }],
    }


def build_croissant(cols: list[dict], n_bytes: int, sha256: str) -> dict:
    file_id = "catalogue-parquet"
    rs_id = "stations"
    fields = []
    for c in cols:
        fields.append({
            "@type": "cr:Field",
            "@id": f"{rs_id}/{c['name']}",
            "name": c["name"],
            "description": c["description"],
            "dataType": c["cr_type"],
            "source": {
                "fileObject": {"@id": file_id},
                "extract": {"column": c["name"]},
            },
        })
    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileProperty": "cr:fileProperty",
            "format": "cr:format",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "repeated": "cr:repeated",
            "sc": "https://schema.org/",
            "source": "cr:source",
            "subField": "cr:subField",
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": DATASET["name"],
        "description": DATASET["description"],
        "version": DATASET["version"],
        "datePublished": DATASET["issued"],
        "url": DATASET["homepage"],
        "license": DATASET["license_url"],
        "citeAs": DATASET["cite_as"],
        "keywords": DATASET["keywords"],
        "creator": [{"@type": "sc:Person", "name": c["name"]} for c in DATASET["creators"]],
        "distribution": [{
            "@type": "cr:FileObject",
            "@id": file_id,
            "name": "stations_gold_standard_final.parquet",
            "description": "Certified catalogue in Apache Parquet.",
            "contentUrl": DATASET["zenodo_file"],
            "encodingFormat": "application/vnd.apache.parquet",
            "contentSize": f"{n_bytes} B",
            "sha256": sha256,
        }],
        "recordSet": [{
            "@type": "cr:RecordSet",
            "@id": rs_id,
            "name": rs_id,
            "description": "One record per certified station.",
            "field": fields,
        }],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = _columns()
    n_bytes, sha256 = _file_stats()

    artefacts = {
        "schema.json": build_json_schema(cols),
        "datapackage.json": build_datapackage(cols, n_bytes, sha256),
        "dcat-ap.jsonld": build_dcat_ap(n_bytes, sha256),
        "croissant.jsonld": build_croissant(cols, n_bytes, sha256),
    }
    for fname, obj in artefacts.items():
        path = OUT_DIR / fname
        path.write_text(
            json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path.relative_to(REPO)}")

    print(f"\n{len(cols)} columns described · parquet sha256 {sha256[:16]}… "
          f"({n_bytes:,} bytes)")


if __name__ == "__main__":
    main()
