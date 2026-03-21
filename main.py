# main.py
"""
Customer Segmentation Pipeline — main entry point.
Runs the full pipeline: load → preprocess → engineer → cluster → evaluate.

Usage:
    python main.py
"""

import os
import pandas as pd
import numpy as np

from src.utils              import ensure_dirs, load_raw_data, save_dataframe, print_section
from src.data_preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.clustering.kmeans       import find_optimal_k, train_kmeans
from src.clustering.hierarchical import train_hierarchical
from src.clustering.dbscan       import train_dbscan
from src.clustering.gmm          import find_optimal_components, train_gmm
from src.evaluation              import score_model, build_comparison_table, print_comparison_table

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH = os.path.join('data', 'raw', 'online_retail_II.xlsx')
PROCESSED_DIR = os.path.join('data', 'processed')
METRICS_DIR   = os.path.join('results', 'metrics')


def main():
    ensure_dirs(PROCESSED_DIR, METRICS_DIR,
                os.path.join('results', 'cluster_plots'),
                os.path.join('results', 'pca_outputs'))

    # ── 1. Load ────────────────────────────────────────────────────────────
    print_section("STEP 1 — Load Raw Data")
    raw_df = load_raw_data(RAW_DATA_PATH)

    # ── 2. Preprocess ──────────────────────────────────────────────────────
    print_section("STEP 2 — Preprocessing")
    customer_df = run_preprocessing(raw_df)
    save_dataframe(customer_df,
                   os.path.join(PROCESSED_DIR, 'customer_features.csv'),
                   label='Preprocessing')

    # ── 3. Feature Engineering ─────────────────────────────────────────────
    print_section("STEP 3 — Feature Engineering")
    fe = run_feature_engineering(customer_df)

    X_scaled  = fe['X_scaled']
    X_pca_95  = fe['X_pca_95']
    X_pca_2d  = fe['X_pca_2d']
    ids       = fe['customer_ids']

    # Save engineered features
    save_dataframe(
        pd.DataFrame(X_pca_95).assign(**{'Customer ID': ids.values}),
        os.path.join(PROCESSED_DIR, 'features_pca_95.csv'), 'FE')

    # ── 4. Clustering ──────────────────────────────────────────────────────
    print_section("STEP 4 — Clustering")
    results = []

    # K-Means
    print("\n→ K-Means")
    km_search = find_optimal_k(X_pca_95, min_business_k=3)  # ← add min_business_k=3
    OPTIMAL_K = km_search['best_k']
    km_labels, _ = train_kmeans(X_pca_95, OPTIMAL_K)
    results.append(score_model(X_pca_95, km_labels, f'K-Means (K={OPTIMAL_K})'))

    # Hierarchical
    print("\n→ Hierarchical")
    h_labels, _ = train_hierarchical(X_pca_95, OPTIMAL_K)
    results.append(score_model(X_pca_95, h_labels, f'Hierarchical (K={OPTIMAL_K})'))

    # DBSCAN
    print("\n→ DBSCAN")
    db_labels, _ = train_dbscan(X_pca_95, eps=0.8, min_samples=5)
    results.append(score_model(X_pca_95, db_labels, 'DBSCAN'))

    # GMM
    print("\n→ GMM")
    gmm_search = find_optimal_components(X_pca_95)
    gmm_labels, _, _ = train_gmm(X_pca_95, gmm_search['best_n'])
    results.append(score_model(X_pca_95, gmm_labels, f'GMM (n={gmm_search["best_n"]})'))

    # ── 5. Evaluate & Compare ──────────────────────────────────────────────
    print_section("STEP 5 — Model Comparison")
    comparison_df = build_comparison_table(results)
    print_comparison_table(comparison_df)
    comparison_df.to_csv(
        os.path.join(METRICS_DIR, 'model_comparison.csv'), index=False)

    # ── 6. Save Final Clusters ─────────────────────────────────────────────
    print_section("STEP 6 — Save Final Output")
    winner_name = comparison_df.iloc[0]['model']
    label_map   = {
        f'K-Means (K={OPTIMAL_K})'         : km_labels,
        f'Hierarchical (K={OPTIMAL_K})'    : h_labels,
        'DBSCAN'                            : db_labels,
        f'GMM (n={gmm_search["best_n"]})'  : gmm_labels,
    }
    final_labels = label_map.get(winner_name, km_labels)

    final_df = customer_df.copy()
    final_df['Final_Cluster'] = final_labels
    final_df = final_df[final_df['Final_Cluster'] != -1].reset_index(drop=True)

    save_dataframe(final_df,
                   os.path.join(PROCESSED_DIR, 'final_clusters.csv'),
                   label='Final')

    # ── 7. Print Summary ───────────────────────────────────────────────────
    print_section("PIPELINE COMPLETE")
    best = comparison_df.iloc[0]
    print(f"""
  Best algorithm   : {best['model']}
  Clusters found   : {int(best['n_clusters'])}
  Silhouette score : {best['silhouette']:.4f}
  Davies-Bouldin   : {best['davies_bouldin']:.4f}
  Final customers  : {len(final_df):,}
  Output saved to  : {PROCESSED_DIR}/final_clusters.csv
    """)


if __name__ == '__main__':
    main()