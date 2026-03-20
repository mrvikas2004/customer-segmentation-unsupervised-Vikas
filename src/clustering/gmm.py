# src/clustering/gmm.py
"""Gaussian Mixture Model clustering with BIC/AIC component selection."""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


def find_optimal_components(X: np.ndarray,
                             n_range: range = range(2, 12),
                             random_state: int = 42) -> dict:
    """
    Fit GMM for each n_components and return BIC/AIC scores.

    Returns dict with keys: bics, aics, best_n
    """
    bics, aics = [], []

    for n in n_range:
        gmm = GaussianMixture(n_components=n, covariance_type='full',
                              random_state=random_state, n_init=3)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
        print(f"  n={n:2d}  BIC={gmm.bic(X):>12,.1f}  AIC={gmm.aic(X):>12,.1f}")

    best_n = list(n_range)[int(np.argmin(bics))]
    print(f"\n  Best n_components (min BIC) = {best_n}")

    return {'bics': bics, 'aics': aics,
            'best_n': best_n, 'n_range': list(n_range)}


def train_gmm(X: np.ndarray,
              n_components: int,
              random_state: int = 42) -> tuple:
    """
    Train final GMM model.

    Returns:
        (labels, gmm_model, probabilities)
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                          random_state=random_state, n_init=5, max_iter=200)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs  = gmm.predict_proba(X).max(axis=1)

    sil = silhouette_score(X, labels, sample_size=3000,
                           random_state=random_state)
    dbi = davies_bouldin_score(X, labels)
    print(f"  GMM (n={n_components}): Silhouette={sil:.4f}  "
          f"DBI={dbi:.4f}  AvgConf={probs.mean():.3f}")
    return labels, gmm, probs