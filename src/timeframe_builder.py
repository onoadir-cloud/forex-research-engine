from __future__ import annotations

from typing import Dict

import pandas as pd

TIMEFRAME_TO_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
BUILD_MAP = {
    "M5": ["M15", "M30", "H1", "H4", "D1"],
    "M15": ["M30", "H1", "H4", "D1"],
    "M30": ["H1", "H4", "D1"],
}


def _pandas_rule(tf: str) -> str:
    return {"M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1d"}[tf]


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = _pandas_rule(timeframe)
    out = (
        df.set_index("datetime")[["open", "high", "low", "close"]]
        .resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    return out


def build_timeframes(df: pd.DataFrame, base_timeframe: str) -> Dict[str, pd.DataFrame]:
    if base_timeframe not in BUILD_MAP:
        raise ValueError("Unsupported base timeframe")
    frames = {}
    for tf in BUILD_MAP[base_timeframe]:
        frames[tf] = resample_ohlc(df, tf)
    return frames


def get_completed_htf_row(htf_df: pd.DataFrame, base_timestamp, htf_minutes: int):
    cutoff = pd.Timestamp(base_timestamp) - pd.Timedelta(minutes=htf_minutes)
    eligible = htf_df[htf_df["datetime"] <= cutoff]
    if eligible.empty:
        return None
    return eligible.iloc[-1]
