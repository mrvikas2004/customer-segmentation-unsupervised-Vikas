# src/clustering/kmeans.py
"""K-Means clustering with elbow, silhouette, and DBI evaluation."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def find_optimal_k(X: np.ndarray,
                   k_range: range = range(2, 12),
                   random_state: int = 42) -> dict:
    """
    Compute inertia, silhouette, and DBI for each K.

    Returns dict with keys: inertias, silhouettes, dbis, best_k
    """
    inertias, silhouettes, dbis = [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                    max_iter=300, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, km.labels_,
                               sample_size=3000, random_state=random_state)
        dbi = davies_bouldin_score(X, km.labels_)
        silhouettes.append(sil)
        dbis.append(dbi)
        print(f"  K={k:2d}  Inertia={km.inertia_:>12,.0f}  "
              f"Silhouette={sil:.4f}  DBI={dbi:.4f}")

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"\n  Best K (max silhouette) = {best_k}")

    return {
        'inertias'    : inertias,
        'silhouettes' : silhouettes,
        'dbis'        : dbis,
        'best_k'      : best_k,
        'k_range'     : list(k_range),
    }


def train_kmeans(X: np.ndarray,
                 n_clusters: int,
                 random_state: int = 42) -> tuple:
    """
    Train final K-Means model.

    Returns:
        (labels, kmeans_model)
    """
    km = KMeans(n_clusters=n_clusters, init='k-means++',
                n_init=20, max_iter=500, random_state=random_state)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=3000,
                           random_state=random_state)
    dbi = davies_bouldin_score(X, labels)
    print(f"  K-Means (K={n_clusters}): "
          f"Silhouette={sil:.4f}  DBI={dbi:.4f}")
    return labels, km