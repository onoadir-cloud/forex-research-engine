from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    h = out["high"]
    l = out["low"]
    for p in [20, 50, 200]:
        out[f"sma{p}"] = c.rolling(p).mean()
    out["sma20_slope"] = out["sma20"].diff()
    out["sma50_slope"] = out["sma50"].diff()

    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atr_percentile_200"] = out["atr14"].rolling(200).rank(pct=True)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    for n in [1, 3, 5, 10]:
        out[f"return_{n}"] = c.pct_change(n)

    std20 = c.rolling(20).std()
    out["z_score_sma20"] = (c - out["sma20"]) / std20
    out["distance_sma50"] = (c - out["sma50"]) / out["sma50"]
    out["range_compression_20"] = (h.rolling(20).max() - l.rolling(20).min()) / c
    out["recent_20_high"] = h.rolling(20).max().shift(1)
    out["recent_20_low"] = l.rolling(20).min().shift(1)
    out["breakout_20_up"] = c > out["recent_20_high"]
    out["breakout_20_down"] = c < out["recent_20_low"]
    return out
