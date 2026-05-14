import pandas as pd

from src.timeframe_builder import get_completed_htf_row, resample_ohlc


def test_m15_to_h1_resampling_ohlc_correctness():
    dt = pd.date_range("2024-01-01", periods=4, freq="15min")
    df = pd.DataFrame({"datetime": dt, "open": [1,2,3,4], "high": [2,3,4,5], "low": [0.5,1.5,2.5,3.5], "close": [1.5,2.5,3.5,4.5]})
    h1 = resample_ohlc(df, "H1")
    r = h1.iloc[0]
    assert r["open"] == 1 and r["high"] == 5 and r["low"] == 0.5 and r["close"] == 4.5


def test_no_lookahead_higher_timeframe_lookup():
    htf = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]), "x": [1, 2]})
    assert get_completed_htf_row(htf, pd.Timestamp("2024-01-01 10:15"), 60) is None
    row = get_completed_htf_row(htf, pd.Timestamp("2024-01-01 11:00"), 60)
    assert row["x"] == 1
