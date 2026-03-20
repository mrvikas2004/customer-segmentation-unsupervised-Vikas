# src/utils.py
"""Utility functions used across the pipeline."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dirs(*paths):
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load and combine both sheets from the UCI Online Retail II Excel file.

    Args:
        filepath: Path to online_retail_II.xlsx

    Returns:
        Combined DataFrame from both year sheets.
    """
    print(f"Loading data from: {filepath}")
    df_2009 = pd.read_excel(filepath, sheet_name='Year 2009-2010')
    df_2010 = pd.read_excel(filepath, sheet_name='Year 2010-2011')
    df = pd.concat([df_2009, df_2010], ignore_index=True)
    print(f"  Loaded {len(df):,} total transactions")
    return df


def save_dataframe(df: pd.DataFrame, path: str, label: str = "") -> None:
    """Save a DataFrame to CSV with a confirmation message."""
    df.to_csv(path, index=False)
    msg = f"[{label}] " if label else ""
    print(f"  {msg}Saved → {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")


def print_section(title: str) -> None:
    """Print a formatted section header."""
    line = "=" * 55
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def plot_and_save(fig: plt.Figure, path: str, label: str = "") -> None:
    """Save a matplotlib figure and print confirmation."""
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    msg = f"[{label}] " if label else ""
    print(f"  {msg}Plot saved → {path}")