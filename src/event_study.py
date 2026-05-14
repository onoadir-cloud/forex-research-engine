from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .timeframe_builder import TIMEFRAME_TO_MINUTES, get_completed_htf_row

HORIZONS = [5, 10, 20]
EXPECTED_EVENT_STUDY_COLUMNS = [
    "pattern_id",
    "pattern_name",
    "context",
    "setup",
    "trigger",
    "session",
    "horizon",
    "sample_size",
    "upward_win_rate",
    "downward_win_rate",
    "avg_forward_return",
    "median_forward_return",
    "avg_favorable_excursion",
    "avg_adverse_excursion",
    "expected_move_atr",
    "estimated_cost_return",
    "ev_after_costs",
    "is_sample_size",
    "oos_sample_size",
    "is_ev_after_costs",
    "oos_ev_after_costs",
    "is_direction",
    "oos_direction",
    "oos_agrees_with_is",
]


def _state(value, fallback: str) -> str:
    """Normalize missing/Unknown states into broad research buckets.

    Early rows and compact synthetic datasets often lack enough history for
    SMA200/ATR-percentile based states, especially on H4. Returning broad
    fallback buckets keeps the event study usable while still preserving the
    completed-candle/no-lookahead higher timeframe lookup.
    """
    if value is None or pd.isna(value):
        return fallback
    text = str(value)
    if text == "Unknown":
        return fallback
    return text


def _direction(series: pd.Series) -> str:
    if series.empty or pd.isna(series.mean()):
        return "Unknown"
    return "Up" if series.mean() >= 0 else "Down"


def _pattern_id(keys: tuple) -> str:
    raw = "|".join(map(str, keys)).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_EVENT_STUDY_COLUMNS)


def run_event_study(base_df, timeframes, spread_pips=0.0, slippage_pips=0.0) -> pd.DataFrame:
    rows = []
    base = base_df.reset_index(drop=True).copy()
    h1 = timeframes.get("H1", pd.DataFrame())
    h4 = timeframes.get("H4", pd.DataFrame())
    split_cutoff = int(len(base) * 0.7)

    for i, row in base.iterrows():
        if "datetime" not in row or pd.isna(row["datetime"]):
            continue

        h1r = get_completed_htf_row(h1, row["datetime"], TIMEFRAME_TO_MINUTES["H1"]) if not h1.empty else None
        h4r = get_completed_htf_row(h4, row["datetime"], TIMEFRAME_TO_MINUTES["H4"]) if not h4.empty else None
        if h1r is None or h4r is None:
            continue

        context = "|".join([
            _state(h4r.get("trend_state", "Unknown"), "AnyTrend"),
            _state(h4r.get("volatility_state", "Unknown"), "AnyVolatility"),
        ])
        setup = _state(h1r.get("setup_state", "Unknown"), "Neutral")
        trigger = _state(row.get("trigger_state", "Unknown"), "Neutral")
        session = _state(row.get("session", "Other"), "Other")

        for horizon in HORIZONS:
            if i + horizon >= len(base):
                continue
            future = base.iloc[i + horizon]
            window = base.iloc[i + 1 : i + horizon + 1]
            if window.empty:
                continue

            fwd = (future["close"] / row["close"]) - 1
            mfe = (window["high"].max() / row["close"]) - 1
            mae = (window["low"].min() / row["close"]) - 1
            cost = (spread_pips + slippage_pips) * 0.0001 / row["close"]
            split = "IS" if i < split_cutoff else "OOS"
            rows.append({
                "context": context,
                "setup": setup,
                "trigger": trigger,
                "session": session,
                "horizon": horizon,
                "forward_return": fwd,
                "favorable": mfe,
                "adverse": mae,
                "cost": cost,
                "split": split,
                "atr14": row.get("atr14", np.nan),
            })

    ev = pd.DataFrame(rows)
    if ev.empty:
        return _empty_result()

    grouped = []
    for keys, g in ev.groupby(["context", "setup", "trigger", "session", "horizon"], dropna=False):
        if len(g) < 20:
            continue

        is_df = g[g["split"] == "IS"]
        oos_df = g[g["split"] == "OOS"]
        avg = g["forward_return"].mean()
        is_direction = _direction(is_df["forward_return"])
        oos_direction = _direction(oos_df["forward_return"])
        atr_mean = g["atr14"].mean()
        expected_move_atr = avg / atr_mean if pd.notna(atr_mean) and atr_mean != 0 else np.nan

        grouped.append({
            "pattern_id": _pattern_id(keys),
            "pattern_name": " | ".join(map(str, keys)),
            "context": keys[0],
            "setup": keys[1],
            "trigger": keys[2],
            "session": keys[3],
            "horizon": keys[4],
            "sample_size": len(g),
            "upward_win_rate": (g["forward_return"] > 0).mean(),
            "downward_win_rate": (g["forward_return"] < 0).mean(),
            "avg_forward_return": avg,
            "median_forward_return": g["forward_return"].median(),
            "avg_favorable_excursion": g["favorable"].mean(),
            "avg_adverse_excursion": g["adverse"].mean(),
            "expected_move_atr": expected_move_atr,
            "estimated_cost_return": g["cost"].mean(),
            "ev_after_costs": avg - g["cost"].mean(),
            "is_sample_size": len(is_df),
            "oos_sample_size": len(oos_df),
            "is_ev_after_costs": is_df["forward_return"].mean() - is_df["cost"].mean() if len(is_df) else np.nan,
            "oos_ev_after_costs": oos_df["forward_return"].mean() - oos_df["cost"].mean() if len(oos_df) else np.nan,
            "is_direction": is_direction,
            "oos_direction": oos_direction,
            "oos_agrees_with_is": is_direction != "Unknown" and is_direction == oos_direction,
        })

    if not grouped:
        return _empty_result()
    return pd.DataFrame(grouped, columns=EXPECTED_EVENT_STUDY_COLUMNS)
