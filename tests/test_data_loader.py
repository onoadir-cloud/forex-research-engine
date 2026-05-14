import pandas as pd
import pytest

from src.data_loader import load_ohlc_csv


def test_valid_datetime_csv_accepted(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("datetime,open,high,low,close\n2024-01-01 00:00,1,2,0.5,1.5\n2024-01-01 00:15,1.5,2.1,1.4,2")
    df, _ = load_ohlc_csv(str(p))
    assert list(df.columns) == ["datetime", "open", "high", "low", "close"]


def test_date_time_csv_accepted(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("date,time,open,high,low,close\n2024-01-01,00:00,1,2,0.5,1.5\n2024-01-01,00:15,1.5,2.1,1.4,2")
    df, _ = load_ohlc_csv(str(p))
    assert len(df) == 2


def test_invalid_ohlc_rejected(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("datetime,open,high,low,close\n2024-01-01 00:00,1,0.8,0.5,1.5")
    with pytest.raises(ValueError):
        load_ohlc_csv(str(p))


def test_duplicate_timestamps_rejected(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("datetime,open,high,low,close\n2024-01-01 00:00,1,2,0.5,1.5\n2024-01-01 00:00,1,2,0.5,1.5")
    with pytest.raises(ValueError):
        load_ohlc_csv(str(p))
