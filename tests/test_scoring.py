import pandas as pd

from src.scoring import score_patterns


def test_negative_oos_is_penalized():
    df = pd.DataFrame([{"sample_size":200,"oos_sample_size":80,"ev_after_costs":0.01,"oos_ev_after_costs":-0.01,"oos_agrees_with_is":False}])
    out = score_patterns(df, 100)
    assert out.iloc[0]["verdict"] == "Reject"


def test_small_sample_cannot_be_strong_candidate():
    df = pd.DataFrame([{"sample_size":50,"oos_sample_size":40,"ev_after_costs":0.02,"oos_ev_after_costs":0.02,"oos_agrees_with_is":True}])
    out = score_patterns(df, 100)
    assert out.iloc[0]["verdict"] != "Strong Candidate for Further Research"
