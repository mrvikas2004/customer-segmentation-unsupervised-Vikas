# src/clustering/hierarchical.py
"""Agglomerative (hierarchical) clustering with Ward linkage."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score


def train_hierarchical(X: np.ndarray,
                       n_clusters: int) -> tuple:
    """
    Fit AgglomerativeClustering with Ward linkage.

    Returns:
        (labels, model)
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=3000, random_state=42)
    dbi = davies_bouldin_score(X, labels)
    print(f"  Hierarchical (K={n_clusters}): "
          f"Silhouette={sil:.4f}  DBI={dbi:.4f}")
    return labels, model