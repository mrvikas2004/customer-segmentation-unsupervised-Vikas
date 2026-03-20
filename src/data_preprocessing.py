# src/data_preprocessing.py
"""
Data cleaning and preprocessing for the customer segmentation pipeline.
Handles missing values, invalid records, cancellations, and outlier treatment.
"""

import pandas as pd
import numpy as np


def remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that cannot be used for customer segmentation:
      - Missing CustomerID
      - Missing Description
      - Negative or zero Quantity
      - Negative or zero Price

    Args:
        df: Raw transaction DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    before = len(df)
    df = df.dropna(subset=['Customer ID', 'Description'])
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    print(f"  Removed {before - len(df):,} invalid rows  →  {len(df):,} remaining")
    return df.reset_index(drop=True)


def extract_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cancellation rate per customer before removing cancelled orders.

    Args:
        df: Raw DataFrame (before filtering cancellations).

    Returns:
        customer_cancel_df: DataFrame with [Customer ID, cancellation_rate].
    """
    df = df.copy()
    df['is_cancelled'] = df['Invoice'].astype(str).str.startswith('C')
    cancel_df = (
        df.groupby('Customer ID')['is_cancelled']
        .mean()
        .reset_index()
        .rename(columns={'is_cancelled': 'cancellation_rate'})
    )
    return cancel_df


def remove_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all cancelled transactions (Invoice starting with 'C')."""
    before = len(df)
    df = df[~df['Invoice'].astype(str).str.startswith('C')].copy()
    print(f"  Removed {before - len(df):,} cancellations  →  {len(df):,} remaining")
    return df.reset_index(drop=True)


def cap_outliers(df: pd.DataFrame, columns: list, percentile: float = 0.99) -> pd.DataFrame:
    """
    Winsorise specified columns at the given upper percentile.

    Args:
        df:         DataFrame to process.
        columns:    List of column names to cap.
        percentile: Upper cap percentile (default 0.99).

    Returns:
        DataFrame with outliers capped.
    """
    df = df.copy()
    for col in columns:
        upper = df[col].quantile(percentile)
        df[col] = df[col].clip(upper=upper)
        print(f"  {col:<30} capped at {upper:.2f}")
    return df


def run_preprocessing(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline — single entry point.

    Args:
        raw_df: Raw combined transaction DataFrame.

    Returns:
        customer_df: Clean customer-level DataFrame ready for feature engineering.
    """
    print("\n[Preprocessing] Starting ...")

    # Step 1 — extract cancellation rates before removing them
    cancel_df = extract_cancellations(raw_df)

    # Step 2 — remove invalid records
    df = remove_invalid_records(raw_df)

    # Step 3 — remove cancellations
    df = remove_cancellations(df)

    # Step 4 — add TotalSpend column
    df['TotalSpend'] = df['Quantity'] * df['Price']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Step 5 — aggregate to customer level
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"  Reference date : {reference_date.date()}")

    customer_df = df.groupby('Customer ID').agg(
        Recency            = ('InvoiceDate',  lambda x: (reference_date - x.max()).days),
        Frequency          = ('Invoice',      'nunique'),
        Monetary           = ('TotalSpend',   'sum'),
        AvgOrderValue      = ('TotalSpend',   'mean'),
        TotalItems         = ('Quantity',     'sum'),
        AvgItemsPerOrder   = ('Quantity',     'mean'),
        UniqueProducts     = ('StockCode',    'nunique'),
        NumCountries       = ('Country',      'nunique'),
        FirstPurchase      = ('InvoiceDate',  'min'),
        LastPurchase       = ('InvoiceDate',  'max'),
    ).reset_index()

    # Step 6 — derived features
    customer_df['CustomerAge'] = (
        reference_date - customer_df['FirstPurchase']
    ).dt.days
    customer_df['AvgDaysBetweenOrders'] = (
        customer_df['CustomerAge'] / customer_df['Frequency']
    )
    customer_df['SpendPerItem'] = (
        customer_df['Monetary'] / customer_df['TotalItems']
    )

    # Step 7 — merge cancellation rate
    customer_df = customer_df.merge(cancel_df, on='Customer ID', how='left')
    customer_df['cancellation_rate'] = customer_df['cancellation_rate'].fillna(0)

    # Step 8 — drop date columns
    customer_df.drop(columns=['FirstPurchase', 'LastPurchase'], inplace=True)

    # Step 9 — cap outliers
    cap_cols = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'TotalItems']
    customer_df = cap_outliers(customer_df, cap_cols)

    print(f"\n[Preprocessing] Complete — {len(customer_df):,} customers, "
          f"{customer_df.shape[1]} features")
    return customer_df