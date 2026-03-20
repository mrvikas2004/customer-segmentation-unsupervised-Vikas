# src/feature_engineering.py
"""
Feature engineering: log transformation, scaling, and PCA.
All transformers are returned so they can be reused for new data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


FEATURES = [
    'Recency', 'Frequency', 'Monetary',
    'AvgOrderValue', 'TotalItems', 'AvgItemsPerOrder',
    'UniqueProducts', 'CustomerAge', 'AvgDaysBetweenOrders',
    'SpendPerItem', 'cancellation_rate'
]


def apply_log_transform(df: pd.DataFrame,
                         features: list = None,
                         skew_threshold: float = 1.0) -> pd.DataFrame:
    """
    Apply log1p to features with absolute skewness above threshold.

    Args:
        df:              DataFrame with features.
        features:        List of columns to consider.
        skew_threshold:  Skewness cutoff (default 1.0).

    Returns:
        Transformed DataFrame copy.
    """
    if features is None:
        features = FEATURES
    df = df.copy()
    transformed = []
    for col in features:
        if col in df.columns and abs(df[col].skew()) > skew_threshold:
            df[col] = np.log1p(df[col])
            transformed.append(col)
    print(f"  log1p applied to: {transformed}")
    return df


def scale_features(df: pd.DataFrame,
                   features: list = None) -> tuple:
    """
    Scale features with RobustScaler.

    Returns:
        (scaled_df, scaler) — scaler kept for inverse transforms.
    """
    if features is None:
        features = FEATURES
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(X_scaled, columns=features)
    print(f"  RobustScaler applied to {len(features)} features")
    return scaled_df, scaler


def apply_pca(X: np.ndarray,
              n_components,
              random_state: int = 42) -> tuple:
    """
    Fit and apply PCA.

    Args:
        X:            Scaled feature array.
        n_components: Number of components or variance fraction (e.g. 0.95).
        random_state: Seed for reproducibility.

    Returns:
        (X_pca, pca_object)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA({n_components}) — {X_pca.shape[1]} components, "
          f"{var:.1f}% variance explained")
    return X_pca, pca


def run_feature_engineering(customer_df: pd.DataFrame) -> dict:
    """
    Full feature engineering pipeline.

    Args:
        customer_df: Output of run_preprocessing().

    Returns:
        Dictionary with keys:
          'X_scaled'     — scaled DataFrame
          'X_pca_2d'     — 2-component PCA array
          'X_pca_3d'     — 3-component PCA array
          'X_pca_95'     — 95% variance PCA array
          'scaler'       — fitted RobustScaler
          'pca_95'       — fitted PCA (95% variance)
          'customer_ids' — Series of Customer IDs
    """
    print("\n[Feature Engineering] Starting ...")

    customer_ids = customer_df['Customer ID']
    X = customer_df[FEATURES].copy()

    # Log transform
    X = apply_log_transform(X)

    # Scale
    X_scaled_df, scaler = scale_features(X)
    X_scaled = X_scaled_df.values

    # PCA variants
    X_pca_2d,  _       = apply_pca(X_scaled, n_components=2)
    X_pca_3d,  _       = apply_pca(X_scaled, n_components=3)
    X_pca_95,  pca_95  = apply_pca(X_scaled, n_components=0.95)

    n95 = X_pca_95.shape[1]
    print(f"\n[Feature Engineering] Complete — "
          f"scaled={X_scaled.shape}, pca_95={X_pca_95.shape}")

    return {
        'X_scaled'     : X_scaled,
        'X_scaled_df'  : X_scaled_df,
        'X_pca_2d'     : X_pca_2d,
        'X_pca_3d'     : X_pca_3d,
        'X_pca_95'     : X_pca_95,
        'scaler'       : scaler,
        'pca_95'       : pca_95,
        'customer_ids' : customer_ids,
        'n_pca_95'     : n95,
    }