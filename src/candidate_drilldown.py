from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .event_study import HORIZONS, _pattern_id, _state
from .timeframe_builder import TIMEFRAME_TO_MINUTES, get_completed_htf_row


EVENT_COLUMNS = [
    "datetime", "context", "setup", "trigger", "session", "horizon", "entry_price", "future_close",
    "forward_return", "forward_return_after_costs", "mfe", "mae", "atr14", "split", "wf_window", "hour", "year", "month",
]

TP_VALUES_ATR = [0.5, 1.0, 1.5, 2.0]
TP_VALUES_PIPS = [3, 5, 8, 10, 15]
MAX_HOLD_BARS = [10, 20, 40]


def _wf_window(total: int, idx: int) -> int:
    edges = np.linspace(0, total, 4).astype(int)
    for w in range(3):
        if edges[w] <= idx < edges[w + 1]:
            return w + 1
    return 3


def _first_touch(event, base, direction, tp_distance, sl_distance, max_hold_bars):
    start = int(event["base_index"]) + 1
    end = min(len(base), start + max_hold_bars)
    entry = float(event["entry_price"])
    if start >= end:
        return "TIMEOUT", 0.0
    for i in range(start, end):
        bar = base.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        if direction == "Up":
            tp_hit = high >= entry + tp_distance
            sl_hit = low <= entry - sl_distance
        else:
            tp_hit = low <= entry - tp_distance
            sl_hit = high >= entry + sl_distance
        if tp_hit and sl_hit:
            return "SL", -1.0
        if tp_hit:
            return "TP", 1.0
        if sl_hit:
            return "SL", -1.0
    return "TIMEOUT", 0.0




def _build_forward_test_strategy_spec(pattern_id: str, direction: str, events: pd.DataFrame, best: pd.DataFrame, concentration: dict, final: str) -> dict:
    top = best.iloc[0].to_dict() if len(best) else {}
    return {
        "pattern_id": pattern_id,
        "direction": direction,
        "forward_test_readiness": final,
        "sample": {
            "event_count": int(len(events)),
            "is_count": int((events["split"] == "IS").sum()),
            "oos_count": int((events["split"] == "OOS").sum()),
            "date_start": str(events["datetime"].min()),
            "date_end": str(events["datetime"].max()),
        },
        "recommended_tp_sl": {
            "tp_type": top.get("tp_type"),
            "tp_value": top.get("tp_value"),
            "sl_value": top.get("sl_value"),
            "max_hold_bars": top.get("max_hold_bars"),
            "avg_r_after_costs": top.get("avg_r_after_costs"),
            "wf_positive_windows": top.get("wf_positive_windows"),
            "is_oos_agree": top.get("is_oos_agree"),
            "verdict": top.get("verdict"),
        },
        "risk_flags": {
            "profit_concentration_top_10pct": concentration.get("top_10_percent_events_contribution", 0.0),
            "profit_concentration_top_5pct": concentration.get("top_5_percent_events_contribution", 0.0),
            "largest_single_event_contribution": concentration.get("largest_single_event_contribution", 0.0),
        },
        "execution_guardrails": [
            "Paper/forward test only; no live deployment.",
            "Keep costs at or below modeled spread+slippage assumptions.",
            "Re-check direction and TP/SL profile after each forward-test block.",
            "Stop forward testing if OOS expectancy turns negative for 2 consecutive review windows.",
        ],
    }

def run_candidate_drilldown(base_df, timeframes, pattern_id: str, spread_pips: float = 0.0, slippage_pips: float = 0.0, output_dir: str = "drilldown_reports") -> dict:
    base = base_df.reset_index(drop=True).copy()
    h1 = timeframes.get("H1", pd.DataFrame())
    h4 = timeframes.get("H4", pd.DataFrame())
    split_cutoff = int(len(base) * 0.7)

    rows = []
    for i, row in base.iterrows():
        if pd.isna(row.get("datetime")):
            continue
        h1r = get_completed_htf_row(h1, row["datetime"], TIMEFRAME_TO_MINUTES["H1"]) if not h1.empty else None
        h4r = get_completed_htf_row(h4, row["datetime"], TIMEFRAME_TO_MINUTES["H4"]) if not h4.empty else None
        if h1r is None or h4r is None:
            continue

        context = "|".join([_state(h4r.get("trend_state", "Unknown"), "AnyTrend"), _state(h4r.get("volatility_state", "Unknown"), "AnyVolatility")])
        setup = _state(h1r.get("setup_state", "Unknown"), "Neutral")
        trigger = _state(row.get("trigger_state", "Unknown"), "Neutral")
        session = _state(row.get("session", "Other"), "Other")

        for horizon in HORIZONS:
            if i + horizon >= len(base):
                continue
            keys = (context, setup, trigger, session, horizon)
            if _pattern_id(keys) != pattern_id:
                continue
            future = base.iloc[i + horizon]
            window = base.iloc[i + 1:i + horizon + 1]
            if window.empty:
                continue
            fwd = (future["close"] / row["close"]) - 1
            cost = (spread_pips + slippage_pips) * 0.0001 / row["close"]
            rows.append({
                "datetime": row["datetime"], "context": context, "setup": setup, "trigger": trigger, "session": session,
                "horizon": horizon, "entry_price": row["close"], "future_close": future["close"], "forward_return": fwd,
                "forward_return_after_costs": fwd - cost, "mfe": (window["high"].max() / row["close"]) - 1,
                "mae": (window["low"].min() / row["close"]) - 1, "atr14": row.get("atr14", np.nan),
                "split": "IS" if i < split_cutoff else "OOS", "wf_window": _wf_window(len(base), i),
                "hour": pd.Timestamp(row["datetime"]).hour, "year": pd.Timestamp(row["datetime"]).year,
                "month": pd.Timestamp(row["datetime"]).month, "base_index": i,
            })

    events = pd.DataFrame(rows)
    if events.empty:
        raise ValueError(f"No events found for pattern_id={pattern_id}")

    monthly = events.groupby(["year", "month"], as_index=False).agg(
        event_count=("forward_return_after_costs", "size"),
        avg_forward_return_after_costs=("forward_return_after_costs", "mean"),
        median_forward_return_after_costs=("forward_return_after_costs", "median"),
        upward_win_rate=("forward_return", lambda s: float((s > 0).mean())),
        avg_mfe=("mfe", "mean"),
        avg_mae=("mae", "mean"),
    )

    yearly = events.groupby("year", as_index=False).agg(events=("forward_return_after_costs", "size"), ev=("forward_return_after_costs", "mean"), win_rate=("forward_return", lambda s: float((s > 0).mean())))
    contrib = events["forward_return_after_costs"].clip(lower=0).sort_values(ascending=False)
    total_pos = float(contrib.sum())
    def _share(p):
        n = max(1, int(np.ceil(len(contrib) * p)))
        return float(contrib.head(n).sum() / total_pos) if total_pos > 0 else 0.0
    concentration = {
        "top_5_percent_events_contribution": _share(0.05),
        "top_10_percent_events_contribution": _share(0.10),
        "largest_single_event_contribution": float(contrib.iloc[0] / total_pos) if total_pos > 0 and len(contrib) else 0.0,
    }

    direction = "Up" if events["forward_return"].mean() >= 0 else "Down"
    grid_rows = []
    cost_pips = spread_pips + slippage_pips
    for tp_type, vals in (("ATR", TP_VALUES_ATR), ("PIPS", TP_VALUES_PIPS)):
        for tpv in vals:
            for slv in vals:
                for hold in MAX_HOLD_BARS:
                    outcomes = []
                    for _, e in events.iterrows():
                        if tp_type == "ATR":
                            atr = float(e["atr14"]) if pd.notna(e["atr14"]) and e["atr14"] > 0 else np.nan
                            if pd.isna(atr):
                                continue
                            tp_d, sl_d = tpv * atr, slv * atr
                        else:
                            tp_d = tpv * 0.0001
                            sl_d = slv * 0.0001
                        hit, r = _first_touch(e, base, direction, tp_d, sl_d, hold)
                        est_cost_r = (cost_pips * 0.0001) / sl_d if sl_d > 0 else 0.0
                        outcomes.append({"hit": hit, "r": r, "r_after": r - est_cost_r, "split": e["split"], "wf_window": int(e["wf_window"])})
                    odf = pd.DataFrame(outcomes)
                    if odf.empty:
                        continue
                    wf_pos = int(sum((odf[odf["wf_window"] == w]["r_after"].mean() > 0) for w in [1, 2, 3]))
                    oos = odf[odf["split"] == "OOS"]
                    is_ = odf[odf["split"] == "IS"]
                    avg_r = float(odf["r"].mean())
                    avg_r_after = float(odf["r_after"].mean())
                    is_avg = float(is_["r_after"].mean()) if len(is_) else np.nan
                    oos_avg = float(oos["r_after"].mean()) if len(oos) else np.nan
                    stress_2x = float((odf["r"] - (2 * ((cost_pips * 0.0001) / (sl_d if tp_type == 'PIPS' else max(1e-9, slv * events['atr14'].mean()))))).mean())
                    verdict = "Reject"
                    if avg_r_after > 0:
                        verdict = "Weak"
                    if avg_r_after > 0 and wf_pos >= 2 and bool((oos_avg > 0) and (is_avg > 0)):
                        verdict = "Interesting Drilldown"
                    if len(odf) >= 150 and len(oos) >= 75 and is_avg > 0 and oos_avg > 0 and wf_pos >= 2 and stress_2x > 0 and concentration["top_10_percent_events_contribution"] <= 0.60:
                        verdict = "Candidate For Forward Test"
                    grid_rows.append({
                        "tp_type": tp_type, "tp_value": tpv, "sl_value": slv, "max_hold_bars": hold, "trades": len(odf),
                        "tp_first_rate": float((odf["hit"] == "TP").mean()), "sl_first_rate": float((odf["hit"] == "SL").mean()), "timeout_rate": float((odf["hit"] == "TIMEOUT").mean()),
                        "avg_r_multiple": avg_r, "median_r_multiple": float(odf["r"].median()), "estimated_cost_r": float((odf["r"] - odf["r_after"]).mean()),
                        "avg_r_after_costs": avg_r_after, "is_avg_r_after_costs": is_avg, "oos_avg_r_after_costs": oos_avg,
                        "is_oos_agree": bool(pd.notna(is_avg) and pd.notna(oos_avg) and is_avg > 0 and oos_avg > 0),
                        "wf_positive_windows": wf_pos, "verdict": verdict,
                    })

    grid = pd.DataFrame(grid_rows)
    best = grid.sort_values(["avg_r_after_costs", "trades"], ascending=[False, False]).head(10)
    final = "Reject"
    if (grid["verdict"] == "Candidate For Forward Test").any():
        final = "Candidate for forward test"
    elif (grid["verdict"] == "Interesting Drilldown").any():
        final = "Interesting drilldown result"
    elif (grid["verdict"] == "Weak").any():
        final = "Weak research lead"

    symbol = str(base_df.get("symbol", pd.Series(["EURUSD"])).iloc[0]) if "symbol" in base_df.columns else "EURUSD"
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol}_{pattern_id}"
    events_path = out / f"{stem}_events.csv"; monthly_path = out / f"{stem}_monthly.csv"; grid_path = out / f"{stem}_tp_sl_grid.csv"; summary_path = out / f"{stem}_summary.json"; md_path = out / f"{stem}_drilldown.md"; spec_path = out / f"{stem}_forward_test_strategy_spec.json"
    events[EVENT_COLUMNS].to_csv(events_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    grid.to_csv(grid_path, index=False)
    summary = {"pattern_id": pattern_id, "direction": direction, "event_count": len(events), "is_count": int((events['split']=='IS').sum()), "oos_count": int((events['split']=='OOS').sum()), "concentration": concentration, "final_conclusion": final}
    summary_path.write_text(json.dumps(summary, indent=2))

    spec = _build_forward_test_strategy_spec(pattern_id, direction, events, best, concentration, final)
    spec_path.write_text(json.dumps(spec, indent=2))
    warn = []
    if concentration["top_10_percent_events_contribution"] > 0.60:
        warn.append("Profit concentration risk detected.")
    if len(yearly) and yearly.sort_values("ev", ascending=False).iloc[0]["ev"] > yearly["ev"].sum() * 0.7:
        warn.append("Yearly concentration warning: most EV comes from one year.")
    md_path.write_text("\n".join([
        "# Candidate Drilldown Report", "", "## Selected pattern", f"- Pattern ID: {pattern_id}",
        f"- Context: {events.iloc[0]['context']} | {events.iloc[0]['setup']} | {events.iloc[0]['trigger']} | {events.iloc[0]['session']} | {events.iloc[0]['horizon']}",
        "## Data range", f"- {events['datetime'].min()} to {events['datetime'].max()}", "## Event count", f"- {len(events)}",
        "## IS/OOS summary", f"- IS: {(events['split']=='IS').sum()} OOS: {(events['split']=='OOS').sum()}",
        "## Walk-forward summary", f"- Positive windows: {int(sum(events.groupby('wf_window')['forward_return_after_costs'].mean()>0))}/3",
        "## Monthly stability", "- See monthly CSV for full details.", "## Yearly stability", yearly.to_string(index=False),
        "## Profit concentration", json.dumps(concentration, indent=2), "## TP/SL grid best rows", best.to_string(index=False),
        "## Failure warnings", *(warn or ["- None."]),
        "## Final conservative conclusion", f"- {final}",
        "- This drilldown is historical analysis only and is not proof of future performance.",
        "- TP/SL first-touch is bar-based and not tick-accurate.",
        "- Passing drilldown means only candidate for paper/forward test, not live trading.",
    ]))

    return {"markdown": str(md_path), "events_csv": str(events_path), "monthly_csv": str(monthly_path), "tp_sl_grid_csv": str(grid_path), "summary_json": str(summary_path), "forward_test_strategy_spec_json": str(spec_path), "summary": summary, "forward_test_strategy_spec": spec}
