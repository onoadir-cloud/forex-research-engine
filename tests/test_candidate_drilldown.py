import json

import numpy as np
import pandas as pd

from src.candidate_drilldown import _first_touch, run_candidate_drilldown
from src.event_study import run_event_study
from src.features import add_features
from src.regimes import add_regime_features
from src.sessions import add_session_features
from src.timeframe_builder import build_timeframes

FORBIDDEN = ["profitable strategy", "guaranteed", "ready for live trading", "works in all markets"]


def _prep(n=2400):
    dt = pd.date_range("2023-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(7)
    close = 1.08 + np.cumsum(rng.normal(0.00001, 0.00018, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0.00035, 0.00008, size=n))
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    df = pd.DataFrame({"datetime": dt, "open": open_, "high": high, "low": low, "close": close, "symbol": "EURUSD"})
    df = add_regime_features(add_session_features(add_features(df)))
    tfs = build_timeframes(df[["datetime", "open", "high", "low", "close"]], "M15")
    for k in tfs:
        tfs[k] = add_regime_features(add_session_features(add_features(tfs[k])))
    return df, tfs


def test_first_touch_cases():
    base = pd.DataFrame({"high": [1.01, 1.02], "low": [0.99, 0.98]})
    event = {"base_index": -1, "entry_price": 1.0}
    assert _first_touch(event, base, "Up", 0.005, 0.005, 2)[0] == "SL"  # same bar conservative

    base2 = pd.DataFrame({"high": [1.006, 1.004], "low": [0.999, 0.998]})
    assert _first_touch(event, base2, "Up", 0.005, 0.005, 2)[0] == "TP"

    base3 = pd.DataFrame({"high": [1.001, 1.002], "low": [0.994, 0.996]})
    assert _first_touch(event, base3, "Up", 0.005, 0.005, 2)[0] == "SL"

    base4 = pd.DataFrame({"high": [1.001, 1.002], "low": [0.999, 0.998]})
    assert _first_touch(event, base4, "Up", 0.005, 0.005, 2)[0] == "TIMEOUT"


def test_candidate_drilldown_outputs_and_schema(tmp_path):
    df, tfs = _prep()
    patterns = run_event_study(df, tfs, 1.2, 0.2)
    assert not patterns.empty
    pid = patterns.iloc[0]["pattern_id"]
    out = run_candidate_drilldown(df, tfs, pid, 1.2, 0.2, str(tmp_path))
    events = pd.read_csv(out["events_csv"])
    monthly = pd.read_csv(out["monthly_csv"])
    grid = pd.read_csv(out["tp_sl_grid_csv"])

    assert len(events) > 0
    for c in ["datetime", "context", "setup", "trigger", "session", "horizon", "entry_price", "future_close", "forward_return", "forward_return_after_costs", "mfe", "mae", "atr14", "split", "wf_window", "hour", "year", "month"]:
        assert c in events.columns
    for c in ["year", "month", "event_count", "avg_forward_return_after_costs", "median_forward_return_after_costs", "upward_win_rate", "avg_mfe", "avg_mae"]:
        assert c in monthly.columns
    for c in ["tp_type", "tp_value", "sl_value", "max_hold_bars", "trades", "tp_first_rate", "sl_first_rate", "timeout_rate", "avg_r_multiple", "median_r_multiple", "estimated_cost_r", "avg_r_after_costs", "is_avg_r_after_costs", "oos_avg_r_after_costs", "is_oos_agree", "wf_positive_windows", "verdict"]:
        assert c in grid.columns

    assert (pd.to_datetime(events["datetime"]) < pd.to_datetime(events["datetime"]).shift(-1).fillna(pd.Timestamp.max)).all()

    md = open(out["markdown"], "r", encoding="utf-8").read().lower()
    assert "final conservative conclusion" in md
    assert "historical analysis only" in md
    for phrase in FORBIDDEN:
        assert phrase not in md

    summary = json.loads(open(out["summary_json"], "r", encoding="utf-8").read())
    assert isinstance(summary, dict)

    spec = json.loads(open(out["forward_test_strategy_spec_json"], "r", encoding="utf-8").read())
    assert isinstance(spec, dict)
    for c in ["pattern_id", "direction", "forward_test_readiness", "sample", "recommended_tp_sl", "risk_flags", "execution_guardrails"]:
        assert c in spec
