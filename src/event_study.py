from __future__ import annotations

import numpy as np
import pandas as pd

from .timeframe_builder import TIMEFRAME_TO_MINUTES, get_completed_htf_row


def run_event_study(base_df, timeframes, spread_pips=0.0, slippage_pips=0.0) -> pd.DataFrame:
    rows = []
    h1 = timeframes.get("H1", pd.DataFrame())
    h4 = timeframes.get("H4", pd.DataFrame())
    for i, row in base_df.iterrows():
        for horizon in [5, 10, 20]:
            if i + horizon >= len(base_df):
                continue
            h1r = get_completed_htf_row(h1, row["datetime"], TIMEFRAME_TO_MINUTES["H1"]) if not h1.empty else None
            h4r = get_completed_htf_row(h4, row["datetime"], TIMEFRAME_TO_MINUTES["H4"]) if not h4.empty else None
            if h1r is None or h4r is None:
                continue
            if "Unknown" in (h1r.get("setup_state", "Unknown"), h4r.get("trend_state", "Unknown"), h4r.get("volatility_state", "Unknown"), row.get("trigger_state", "Unknown")):
                continue
            future = base_df.iloc[i + horizon]
            window = base_df.iloc[i + 1 : i + horizon + 1]
            fwd = (future["close"] / row["close"]) - 1
            mfe = (window["high"].max() / row["close"]) - 1
            mae = (window["low"].min() / row["close"]) - 1
            cost = (spread_pips + slippage_pips) * 0.0001 / row["close"]
            split = "IS" if i < int(len(base_df) * 0.7) else "OOS"
            rows.append({
                "context": f"{h4r['trend_state']}|{h4r['volatility_state']}",
                "setup": h1r["setup_state"],
                "trigger": row["trigger_state"],
                "session": row["session"],
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
        return ev
    grouped = []
    for keys, g in ev.groupby(["context", "setup", "trigger", "session", "horizon"]):
        if len(g) < 20:
            continue
        is_df = g[g["split"] == "IS"]
        oos_df = g[g["split"] == "OOS"]
        avg = g["forward_return"].mean()
        direction = "Up" if avg >= 0 else "Down"
        grouped.append({
            "pattern_id": abs(hash(keys)) % 10**10,
            "pattern_name": " | ".join(map(str, keys)),
            "context": keys[0], "setup": keys[1], "trigger": keys[2], "session": keys[3], "horizon": keys[4],
            "sample_size": len(g),
            "upward_win_rate": (g["forward_return"] > 0).mean(),
            "downward_win_rate": (g["forward_return"] < 0).mean(),
            "avg_forward_return": avg,
            "median_forward_return": g["forward_return"].median(),
            "avg_favorable_excursion": g["favorable"].mean(),
            "avg_adverse_excursion": g["adverse"].mean(),
            "expected_move_atr": avg / g["atr14"].mean() if g["atr14"].mean() and not np.isnan(g["atr14"].mean()) else np.nan,
            "estimated_cost_return": g["cost"].mean(),
            "ev_after_costs": avg - g["cost"].mean(),
            "is_sample_size": len(is_df),
            "oos_sample_size": len(oos_df),
            "is_ev_after_costs": is_df["forward_return"].mean() - is_df["cost"].mean() if len(is_df) else np.nan,
            "oos_ev_after_costs": oos_df["forward_return"].mean() - oos_df["cost"].mean() if len(oos_df) else np.nan,
            "is_direction": "Up" if is_df["forward_return"].mean() >= 0 else "Down",
            "oos_direction": "Up" if oos_df["forward_return"].mean() >= 0 else "Down",
            "oos_agrees_with_is": ("Up" if is_df["forward_return"].mean() >= 0 else "Down") == ("Up" if oos_df["forward_return"].mean() >= 0 else "Down"),
        })
    return pd.DataFrame(grouped)
