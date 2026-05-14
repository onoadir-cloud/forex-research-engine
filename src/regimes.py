from __future__ import annotations

import pandas as pd


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["trend_state"] = "Unknown"
    has_trend = out[["sma200", "sma50_slope"]].notna().all(axis=1)
    out.loc[has_trend & (out["close"] > out["sma200"]) & (out["sma50_slope"] > 0), "trend_state"] = "Uptrend"
    out.loc[has_trend & (out["close"] < out["sma200"]) & (out["sma50_slope"] < 0), "trend_state"] = "Downtrend"
    out.loc[has_trend & (out["trend_state"] == "Unknown"), "trend_state"] = "Range"

    out["volatility_state"] = "Unknown"
    ap = out["atr_percentile_200"]
    out.loc[ap < 0.33, "volatility_state"] = "Low"
    out.loc[(ap >= 0.33) & (ap <= 0.66), "volatility_state"] = "Normal"
    out.loc[ap > 0.66, "volatility_state"] = "High"

    out["setup_state"] = "Neutral"
    out.loc[out["trend_state"] == "Unknown", "setup_state"] = "Unknown"
    out.loc[(out["trend_state"] == "Uptrend") & (out["distance_sma50"] < 0), "setup_state"] = "PullbackUptrend"
    out.loc[(out["trend_state"] == "Downtrend") & (out["distance_sma50"] > 0), "setup_state"] = "PullbackDowntrend"
    out.loc[out["range_compression_20"] < out["range_compression_20"].rolling(100).quantile(0.2), "setup_state"] = "Compression"
    out.loc[out["return_3"] > 0.002, "setup_state"] = "MomentumUp"
    out.loc[out["return_3"] < -0.002, "setup_state"] = "MomentumDown"

    out["trigger_state"] = "Neutral"
    out.loc[out["trend_state"] == "Unknown", "trigger_state"] = "Unknown"
    out.loc[out["rsi14"] < 30, "trigger_state"] = "RsiRecoveryLow"
    out.loc[out["rsi14"] > 70, "trigger_state"] = "RsiRejectionHigh"
    out.loc[out["return_1"] > 0, "trigger_state"] = "MomentumPositive"
    out.loc[out["return_1"] < 0, "trigger_state"] = "MomentumNegative"
    out.loc[out["z_score_sma20"] < -2, "trigger_state"] = "ZScoreStretchedDown"
    out.loc[out["z_score_sma20"] > 2, "trigger_state"] = "ZScoreStretchedUp"
    out.loc[out["range_compression_20"] < out["range_compression_20"].rolling(100).quantile(0.2), "trigger_state"] = "Compression"
    out.loc[out["breakout_20_up"], "trigger_state"] = "BreakoutUp20"
    out.loc[out["breakout_20_down"], "trigger_state"] = "BreakdownDown20"

    return out
