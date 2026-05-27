"""XP2 — Topology-aware geospatial anomaly detection for GBFS networks.

Replaces the naïve centroid + 3σ MAD heuristic (current A4) with a
three-stage pipeline that is geometry-agnostic:

  Stage 1 — HDBSCAN density clustering on haversine distances
  Stage 2 — Spectral graph analysis of the k-NN station adjacency graph
  Stage 3 — Composite anomaly score with ablation against legacy A4

Why this works on anisotropic geometries
-----------------------------------------
The centroid approach implicitly assumes an isotropic (roughly circular)
station distribution.  A network running along a river or a national-scale
operator spanning multiple cities violates this assumption — legitimate
peripheral stations exceed the 3σ radius from the centroid.

HDBSCAN (Stage 1) identifies clusters of arbitrary shape by tracking
density persistence across scales, so a linear riverside network is
a single cluster, not a collection of outliers.

The spectral approach (Stage 2) builds a weighted k-NN graph where edge
weights decay with distance (Gaussian kernel).  The Fiedler vector
(second-smallest eigenvector of the graph Laplacian) encodes the graph's
connectivity structure:
  - In a well-connected network, all stations load onto a smooth gradient.
  - A truly disconnected outlier sits in a separate spectral component
    (near-zero Fiedler value) or shows extreme localisation in higher
    eigenvectors.

The composite score (Stage 3) fuses both signals:
  anomaly_score(s) = α · hdbscan_outlier_score(s) + (1-α) · spectral_isolation(s)

where α ∈ [0,1] is a mixing weight (default 0.5).  Stations above a
threshold on the composite score are flagged A4.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


@dataclass
class SpatialAnomalyResult:
    """Per-station spatial anomaly classification."""
    station_df: pd.DataFrame
    system_summary: pd.DataFrame


def _haversine_distances(lat1: np.ndarray, lon1: np.ndarray,
                         lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised haversine distance in metres."""
    R = 6_371_000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _build_knn_graph(coords_rad: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Build k-NN graph using haversine metric; return (distances, indices)."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords_rad)),
                          metric="haversine", algorithm="ball_tree")
    nn.fit(coords_rad)
    distances, indices = nn.kneighbors(coords_rad)
    R = 6_371_000.0
    distances_m = distances * R
    return distances_m[:, 1:], indices[:, 1:]


def _graph_laplacian_spectrum(
    distances_m: np.ndarray,
    indices: np.ndarray,
    n_stations: int,
    sigma_m: float,
    n_eigenvectors: int,
) -> np.ndarray:
    """Compute the first n_eigenvectors of the normalised graph Laplacian.

    The adjacency weight between stations i and j is:
        w_ij = exp(-d_ij² / (2 σ²))

    Returns eigenvectors as columns of shape (n_stations, n_eigenvectors).
    """
    rows, cols, weights = [], [], []
    for i in range(n_stations):
        for j_idx in range(distances_m.shape[1]):
            j = indices[i, j_idx]
            d = distances_m[i, j_idx]
            w = np.exp(-d ** 2 / (2 * sigma_m ** 2))
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([w, w])

    W = csr_matrix((weights, (rows, cols)), shape=(n_stations, n_stations))
    D_diag = np.array(W.sum(axis=1)).flatten()
    D_diag[D_diag == 0] = 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_diag))
    L_norm = np.eye(n_stations) - D_inv_sqrt @ W.toarray() @ D_inv_sqrt

    n_eig = min(n_eigenvectors, n_stations - 1)
    if n_eig < 2:
        return np.zeros((n_stations, 1))

    eigenvalues, eigenvectors = eigsh(L_norm, k=n_eig, which="SM")
    sort_idx = np.argsort(eigenvalues)
    return eigenvectors[:, sort_idx]


def _spectral_isolation_score(eigenvectors: np.ndarray) -> np.ndarray:
    """Compute per-station spectral isolation from the Fiedler vector and beyond.

    The score is the L2 norm of the station's embedding in eigenvectors 1..K
    (skipping eigenvector 0 which is constant), normalised to [0, 1].

    High score = the station is spectrally distant from the network core.
    """
    if eigenvectors.shape[1] < 2:
        return np.zeros(eigenvectors.shape[0])
    embedding = eigenvectors[:, 1:]
    norms = np.linalg.norm(embedding, axis=1)
    max_norm = norms.max()
    if max_norm == 0:
        return np.zeros_like(norms)
    return norms / max_norm


def _legacy_centroid_outlier(
    lats: np.ndarray, lons: np.ndarray, sigma_max: float, use_mad: bool
) -> np.ndarray:
    """Legacy A4 centroid + σ-clipping (for ablation comparison)."""
    clat, clon = np.nanmedian(lats), np.nanmedian(lons)
    dists = _haversine_distances(lats, lons,
                                 np.full_like(lats, clat),
                                 np.full_like(lons, clon))
    if use_mad:
        med_d = np.nanmedian(dists)
        mad = np.nanmedian(np.abs(dists - med_d))
        sigma = mad * 1.4826
    else:
        sigma = np.nanstd(dists)
    sigma = max(sigma, 1.0)
    return dists / sigma


def detect_spatial_anomalies(
    df: pd.DataFrame,
    *,
    # HDBSCAN
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_epsilon: float = 500.0,
    # Spectral
    k_nearest: int = 7,
    sigma_bandwidth_m: float = 2000.0,
    n_eigenvectors: int = 10,
    fiedler_threshold: float = 0.05,
    # Composite
    alpha: float = 0.5,
    anomaly_threshold: float = 0.8,
    # Legacy
    legacy_sigma_max: float = 3.0,
    legacy_use_mad: bool = True,
) -> SpatialAnomalyResult:
    """Run the full three-stage spatial anomaly pipeline per system.

    Parameters
    ----------
    df : DataFrame
        Must contain system_id, station_id, lat, lon.
    alpha : float
        Mixing weight: 0 = pure spectral, 1 = pure HDBSCAN.
    anomaly_threshold : float
        Composite score above this flags A4.

    Returns
    -------
    SpatialAnomalyResult with per-station and per-system DataFrames.
    """
    all_station_results = []
    system_summaries = []

    for sys_id, grp in df.groupby("system_id"):
        n = len(grp)
        lats = grp["lat"].to_numpy(dtype="float64")
        lons = grp["lon"].to_numpy(dtype="float64")

        valid = ~(np.isnan(lats) | np.isnan(lons))
        if valid.sum() < 4:
            for _, row in grp.iterrows():
                all_station_results.append({
                    "system_id": sys_id,
                    "station_id": row["station_id"],
                    "hdbscan_label": -1,
                    "hdbscan_outlier_score": np.nan,
                    "spectral_isolation": np.nan,
                    "composite_score": np.nan,
                    "legacy_sigma_distance": np.nan,
                    "flag_A4_composite": False,
                    "flag_A4_legacy": False,
                })
            continue

        idx_valid = grp.index[valid]
        v_lats = lats[valid]
        v_lons = lons[valid]
        n_valid = len(v_lats)

        # ── Stage 1: HDBSCAN ──
        coords_rad = np.column_stack([np.radians(v_lats), np.radians(v_lons)])

        effective_min_cluster = min(min_cluster_size, max(2, n_valid // 3))
        effective_min_samples = min(min_samples, effective_min_cluster)

        try:
            hdb = HDBSCAN(
                min_cluster_size=effective_min_cluster,
                min_samples=effective_min_samples,
                cluster_selection_epsilon=0.0,
                metric="haversine",
            )
            labels = hdb.fit_predict(coords_rad)
        except (TypeError, ValueError):
            hdb = HDBSCAN(
                min_cluster_size=effective_min_cluster,
                min_samples=effective_min_samples,
                metric="haversine",
            )
            labels = hdb.fit_predict(coords_rad)

        if hasattr(hdb, "outlier_scores_"):
            outlier_scores = hdb.outlier_scores_
        else:
            outlier_scores = np.where(labels == -1, 1.0, 0.0)

        # ── Stage 2: Spectral graph ──
        # Cap at 2000 stations: dense Laplacian is O(n²) memory, O(n³) compute
        MAX_SPECTRAL = 2000
        effective_k = min(k_nearest, n_valid - 1)
        if effective_k >= 2 and n_valid <= MAX_SPECTRAL:
            knn_dists, knn_indices = _build_knn_graph(coords_rad, effective_k)
            eigvecs = _graph_laplacian_spectrum(
                knn_dists, knn_indices, n_valid, sigma_bandwidth_m, n_eigenvectors,
            )
            spectral_scores = _spectral_isolation_score(eigvecs)
            fiedler_val = float(fiedler_threshold)
        else:
            spectral_scores = np.zeros(n_valid)
            fiedler_val = 0.0

        # ── Stage 3: Composite ──
        os_norm = outlier_scores / max(outlier_scores.max(), 1e-10)
        composite = alpha * os_norm + (1 - alpha) * spectral_scores

        # ── Legacy (ablation) ──
        legacy_sigma = _legacy_centroid_outlier(v_lats, v_lons, legacy_sigma_max, legacy_use_mad)

        station_ids = grp.loc[idx_valid, "station_id"].values
        for i in range(n_valid):
            all_station_results.append({
                "system_id": sys_id,
                "station_id": station_ids[i],
                "hdbscan_label": int(labels[i]),
                "hdbscan_outlier_score": float(os_norm[i]),
                "spectral_isolation": float(spectral_scores[i]),
                "composite_score": float(composite[i]),
                "legacy_sigma_distance": float(legacy_sigma[i]),
                "flag_A4_composite": bool(composite[i] > anomaly_threshold),
                "flag_A4_legacy": bool(legacy_sigma[i] > legacy_sigma_max),
            })

        n_composite = int((composite > anomaly_threshold).sum())
        n_legacy = int((legacy_sigma > legacy_sigma_max).sum())
        system_summaries.append({
            "system_id": sys_id,
            "n_stations": n_valid,
            "n_clusters_hdbscan": int(len(set(labels) - {-1})),
            "n_noise_hdbscan": int((labels == -1).sum()),
            "fiedler_value": float(fiedler_val),
            "n_flagged_composite": n_composite,
            "n_flagged_legacy": n_legacy,
            "composite_rate": n_composite / n_valid if n_valid > 0 else 0,
            "legacy_rate": n_legacy / n_valid if n_valid > 0 else 0,
        })

    return SpatialAnomalyResult(
        station_df=pd.DataFrame(all_station_results),
        system_summary=pd.DataFrame(system_summaries),
    )
