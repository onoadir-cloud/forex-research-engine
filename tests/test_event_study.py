import numpy as np
import pandas as pd

from src.event_study import run_event_study
from src.features import add_features
from src.regimes import add_regime_features
from src.sessions import add_session_features
from src.timeframe_builder import build_timeframes


def _prep(n=2000):
    dt = pd.date_range("2024-01-01", periods=n, freq="15min")
    base = 1 + np.cumsum(np.random.default_rng(0).normal(0, 0.0005, n))
    df = pd.DataFrame({"datetime": dt, "open": base, "high": base+0.001, "low": base-0.001, "close": base})
    df = add_regime_features(add_session_features(add_features(df)))
    tfs = build_timeframes(df[["datetime","open","high","low","close"]], "M15")
    for k in tfs:
        tfs[k] = add_regime_features(add_session_features(add_features(tfs[k])))
    return df, tfs


def test_event_study_produces_rows_and_walk_forward_columns():
    df, tfs = _prep()
    out = run_event_study(df, tfs, 1.2, 0.2)
    assert isinstance(out, pd.DataFrame)
    assert set([
        "pattern_id", "sample_size", "oos_sample_size",
        "wf_1_ev_after_costs", "wf_2_ev_after_costs", "wf_3_ev_after_costs",
        "wf_positive_windows", "wf_total_windows", "wf_consistency_ratio",
    ]).issubset(out.columns)


def test_oos_split_chronological():
    df, tfs = _prep()
    out = run_event_study(df, tfs)
    assert (out["is_sample_size"] + out["oos_sample_size"] == out["sample_size"]).all()
