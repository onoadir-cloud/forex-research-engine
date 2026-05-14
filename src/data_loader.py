from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


REQUIRED_OHLC = ["open", "high", "low", "close"]


def _parse_datetime(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="raise", utc=False)
    elif "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="raise", utc=False)
    else:
        raise ValueError("CSV must include either 'datetime' or both 'date' and 'time' columns")
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt


def load_ohlc_csv(path: str) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in REQUIRED_OHLC:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["datetime"] = _parse_datetime(df)

    for col in REQUIRED_OHLC:
        df[col] = pd.to_numeric(df[col], errors="raise")

    invalid_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    invalid_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        raise ValueError("Invalid OHLC relationships detected")

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    df = df[["datetime", "open", "high", "low", "close"]].sort_values("datetime").reset_index(drop=True)

    if len(df) < 1000:
        warnings.append("Fewer than 1000 candles; evidence is likely weak and requires further validation.")

    if len(df) > 1:
        diffs = df["datetime"].diff().dropna()
        base_gap = diffs.mode().iloc[0] if not diffs.empty else None
        if base_gap is not None:
            large = diffs > base_gap * 5
            if large.any():
                warnings.append(f"Detected {int(large.sum())} large timestamp gaps.")

    return df, warnings
