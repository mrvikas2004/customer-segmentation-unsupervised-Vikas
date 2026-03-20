# src/clustering/dbscan.py
"""DBSCAN clustering with k-distance eps estimation."""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score


def estimate_eps(X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Compute sorted k-distances to help visually pick eps.

    Returns:
        k_distances array (sorted descending) for plotting.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, k - 1])[::-1]
    return k_distances


def train_dbscan(X: np.ndarray,
                 eps: float = 0.8,
                 min_samples: int = 5) -> tuple:
    """
    Fit DBSCAN and report cluster and noise statistics.

    Returns:
        (labels, model)
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    noise_pct  = n_noise / len(labels) * 100

    print(f"  DBSCAN (eps={eps}): "
          f"clusters={n_clusters}  noise={n_noise} ({noise_pct:.1f}%)")

    # Only score if valid
    mask = labels != -1
    if len(set(labels[mask])) >= 2:
        sil = silhouette_score(X[mask], labels[mask],
                               sample_size=3000, random_state=42)
        dbi = davies_bouldin_score(X[mask], labels[mask])
        print(f"  Silhouette={sil:.4f}  DBI={dbi:.4f}")
    return labels, model