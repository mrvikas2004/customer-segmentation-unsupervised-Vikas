# src/evaluation.py
"""Model evaluation utilities — scores, comparison table, winner selection."""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score


def score_model(X: np.ndarray,
                labels: np.ndarray,
                model_name: str) -> dict:
    """
    Compute silhouette, DBI, noise%, and cluster count for one model.

    Returns:
        Dict of metrics, or None if fewer than 2 clusters found.
    """
    mask         = labels != -1
    X_clean      = X[mask]
    labels_clean = labels[mask]
    n_clusters   = len(set(labels_clean))

    if n_clusters < 2:
        print(f"  {model_name}: only {n_clusters} cluster — skipping.")
        return None

    sil       = silhouette_score(X_clean, labels_clean,
                                 sample_size=3000, random_state=42)
    dbi       = davies_bouldin_score(X_clean, labels_clean)
    noise_pct = (~mask).sum() / len(labels) * 100

    return {
        'model'          : model_name,
        'n_clusters'     : n_clusters,
        'silhouette'     : round(sil, 4),
        'davies_bouldin' : round(dbi, 4),
        'noise_pct'      : round(noise_pct, 2),
    }


def build_comparison_table(results: list) -> pd.DataFrame:
    """
    Build and rank a comparison DataFrame from a list of score dicts.

    Returns:
        DataFrame sorted by weighted final score (descending).
    """
    df = pd.DataFrame([r for r in results if r is not None])

    # Normalise 0–1
    df['sil_norm']   = (df['silhouette'] - df['silhouette'].min()) / \
                        (df['silhouette'].max() - df['silhouette'].min() + 1e-9)
    df['dbi_norm']   = 1 - (df['davies_bouldin'] - df['davies_bouldin'].min()) / \
                            (df['davies_bouldin'].max() - df['davies_bouldin'].min() + 1e-9)
    df['noise_norm'] = 1 - df['noise_pct'] / 100

    # Weighted score
    df['final_score'] = (0.50 * df['sil_norm'] +
                         0.30 * df['dbi_norm'] +
                         0.20 * df['noise_norm'])

    return df.sort_values('final_score', ascending=False).reset_index(drop=True)


def print_comparison_table(df: pd.DataFrame) -> None:
    """Pretty-print the model comparison table."""
    medals = ['🥇', '🥈', '🥉']
    print("\n" + "=" * 65)
    print("  ALGORITHM COMPARISON")
    print("=" * 65)
    for i, row in df.iterrows():
        m = medals[i] if i < 3 else "  "
        print(f"  {m}  {row['model']:<32} "
              f"Score={row['final_score']:.4f}  "
              f"Sil={row['silhouette']:.4f}  "
              f"DBI={row['davies_bouldin']:.4f}")
    print("=" * 65)
    print(f"\n  🏆 Winner: {df.iloc[0]['model']}")