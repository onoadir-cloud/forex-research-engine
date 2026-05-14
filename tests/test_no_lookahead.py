import pandas as pd

from src.timeframe_builder import get_completed_htf_row


def test_higher_timeframe_not_available_before_completion():
    htf = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-01 10:00"]), "value": [1]})
    assert get_completed_htf_row(htf, pd.Timestamp("2024-01-01 10:45"), 60) is None
    r = get_completed_htf_row(htf, pd.Timestamp("2024-01-01 11:00"), 60)
    assert r["value"] == 1
