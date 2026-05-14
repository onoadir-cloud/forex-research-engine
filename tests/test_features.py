import numpy as np
import pandas as pd

from src.features import add_features


def test_feature_columns_exist_and_finite_after_warmup():
    n = 400
    dt = pd.date_range("2024-01-01", periods=n, freq="15min")
    c = np.linspace(1.0, 1.4, n)
    df = pd.DataFrame({"datetime": dt, "open": c, "high": c+0.01, "low": c-0.01, "close": c})
    out = add_features(df)
    cols = ["sma20","sma50","sma200","atr14","rsi14","return_10","z_score_sma20","distance_sma50","range_compression_20","breakout_20_up","breakout_20_down"]
    for col in cols:
        assert col in out.columns
    warm = out.iloc[250:]
    assert np.isfinite(warm[["sma20","sma50","sma200","atr14","return_10","distance_sma50"]]).all().all()
