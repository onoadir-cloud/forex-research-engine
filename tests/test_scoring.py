import pandas as pd

from src.scoring import score_patterns


def _base_row(**kw):
    row = {
        "sample_size": 350,
        "oos_sample_size": 180,
        "avg_forward_return": 0.003,
        "estimated_cost_return": 0.0005,
        "ev_after_costs": 0.0025,
        "is_ev_after_costs": 0.002,
        "oos_ev_after_costs": 0.002,
        "oos_agrees_with_is": True,
        "wf_positive_windows": 3,
    }
    row.update(kw)
    return row


def test_insufficient_evidence_cannot_be_interesting():
    out = score_patterns(pd.DataFrame([_base_row(primary_rejection_reason="Insufficient evidence")]), 100)
    assert out.iloc[0]["verdict"] in {"Weak Evidence", "Reject"}


def test_oos_below_100_cannot_be_interesting():
    out = score_patterns(pd.DataFrame([_base_row(oos_sample_size=90)]), 100)
    assert out.iloc[0]["verdict"] != "Interesting"


def test_negative_is_positive_oos_cannot_exceed_weak():
    out = score_patterns(pd.DataFrame([_base_row(is_ev_after_costs=-0.001, oos_ev_after_costs=0.001)]), 100)
    assert out.iloc[0]["verdict"] in {"Weak Evidence", "Reject"}


def test_strong_candidate_requires_no_rejection_reason():
    out = score_patterns(pd.DataFrame([_base_row(primary_rejection_reason="Low sample size")]), 100)
    assert out.iloc[0]["verdict"] != "Strong Candidate for Further Research"


def test_multiple_testing_penalty_reduces_score():
    low = score_patterns(pd.DataFrame([_base_row()]), 10).iloc[0]
    high = score_patterns(pd.DataFrame([_base_row()]), 10000).iloc[0]
    assert high["multiple_testing_penalty"] > low["multiple_testing_penalty"]
    assert high["research_score"] < high["score_before_penalties"]


def test_2x_cost_failure_prevents_interesting_or_strong():
    out = score_patterns(pd.DataFrame([_base_row(avg_forward_return=0.001, estimated_cost_return=0.0006)]), 100)
    assert out.iloc[0]["cost_stress_2x_pass"] == False
    assert out.iloc[0]["verdict"] in {"Weak Evidence", "Reject"}


def test_reject_or_weak_always_have_non_empty_rejection_reason():
    rows = [
        _base_row(sample_size=20, oos_sample_size=10, primary_rejection_reason=""),
        _base_row(oos_sample_size=90, primary_rejection_reason=None),
        _base_row(avg_forward_return=0.001, estimated_cost_return=0.0006, primary_rejection_reason=""),
    ]
    out = score_patterns(pd.DataFrame(rows), 100)
    subset = out[out["verdict"].isin(["Reject", "Weak Evidence"])]
    assert not subset.empty
    assert subset["primary_rejection_reason"].notna().all()
    assert (subset["primary_rejection_reason"].astype(str).str.strip() != "").all()


def test_weak_evidence_without_blocking_reason_gets_default_reason():
    out = score_patterns(pd.DataFrame([_base_row(oos_sample_size=90, primary_rejection_reason="")]), 100)
    row = out.iloc[0]
    assert row["verdict"] == "Weak Evidence"
    assert row["primary_rejection_reason"] == "Below Interesting thresholds"
