from __future__ import annotations

import math
import pandas as pd

MIN_SAMPLE_WEAK = 80
MIN_SAMPLE_INTERESTING = 200
MIN_SAMPLE_STRONG = 300
MIN_OOS_WEAK = 50
MIN_OOS_INTERESTING = 100
MIN_OOS_STRONG = 150


def _clean_reason(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def score_patterns(patterns_df: pd.DataFrame, tested_patterns_count: int) -> pd.DataFrame:
    if patterns_df.empty:
        return patterns_df

    out = patterns_df.copy()
    scores = []
    scores_before_penalty = []
    penalties = []
    verdicts = []
    reasons = []

    # Multiple testing adjustment: as tested pattern count increases, false-positive
    # risk rises. Penalize score using a capped log10 function (sub-linear growth).
    base_penalty = min(25.0, math.log10(max(tested_patterns_count, 1)) * 5.0)

    for _, r in out.iterrows():
        sample_size = int(r.get("sample_size", 0) or 0)
        oos_sample_size = int(r.get("oos_sample_size", 0) or 0)
        is_ev = float(r.get("is_ev_after_costs", float('nan')))
        oos_ev = float(r.get("oos_ev_after_costs", float('nan')))
        gross_avg = float(r.get("avg_forward_return", 0.0) or 0.0)
        cost = float(r.get("estimated_cost_return", 0.0) or 0.0)
        base_reason = _clean_reason(r.get("primary_rejection_reason", ""))

        ev_2x = gross_avg - 2 * cost
        ev_3x = gross_avg - 3 * cost
        pass_2x = ev_2x > 0
        pass_3x = ev_3x > 0

        score = 50.0
        if sample_size >= MIN_SAMPLE_INTERESTING:
            score += 10
        if sample_size >= MIN_SAMPLE_STRONG:
            score += 10
        if oos_sample_size >= MIN_OOS_INTERESTING:
            score += 10
        if oos_sample_size >= MIN_OOS_STRONG:
            score += 10
        if oos_ev > 0:
            score += 10
        else:
            score -= 20
        if r.get("oos_agrees_with_is", False):
            score += 5
        else:
            score -= 10
        if pass_3x:
            score += 3

        reason = base_reason
        if not reason and (sample_size < MIN_SAMPLE_WEAK or oos_sample_size < MIN_OOS_WEAK):
            reason = "Insufficient evidence"
        if not reason and oos_ev <= 0:
            reason = "Non-positive OOS EV after costs"
        if not reason and pd.notna(is_ev) and pd.notna(oos_ev) and is_ev < 0 < oos_ev:
            reason = "IS/OOS instability"
        if not reason and not pass_2x:
            reason = "Fails 2x cost stress"

        verdict = "Reject"
        if score >= 45:
            verdict = "Weak Evidence"
        if (
            score >= 65
            and sample_size >= MIN_SAMPLE_INTERESTING
            and oos_sample_size >= MIN_OOS_INTERESTING
            and oos_ev > 0
            and pass_2x
            and int(r.get("wf_positive_windows", 0) or 0) >= 2
        ):
            verdict = "Interesting"
        if (
            score >= 80
            and sample_size >= MIN_SAMPLE_STRONG
            and oos_sample_size >= MIN_OOS_STRONG
            and pd.notna(is_ev)
            and is_ev > 0
            and oos_ev > 0
            and r.get("oos_agrees_with_is", False)
            and pass_2x
            and int(r.get("wf_positive_windows", 0) or 0) == 3
        ):
            verdict = "Strong Candidate for Further Research"

        # Hard reliability clamps.
        if reason:
            if verdict in {"Interesting", "Strong Candidate for Further Research"}:
                verdict = "Weak Evidence"
        if reason == "Insufficient evidence":
            verdict = "Reject" if score < 45 else "Weak Evidence"
        if sample_size < MIN_SAMPLE_WEAK or oos_sample_size < MIN_OOS_WEAK:
            verdict = "Reject" if score < 45 else "Weak Evidence"
        if oos_sample_size < MIN_OOS_INTERESTING and verdict == "Interesting":
            verdict = "Weak Evidence"
        if pd.notna(is_ev) and pd.notna(oos_ev) and is_ev < 0 < oos_ev:
            verdict = "Reject" if score < 45 else "Weak Evidence"
        if oos_ev <= 0:
            verdict = "Reject" if score < 45 else "Weak Evidence"
        if not pass_2x and verdict in {"Interesting", "Strong Candidate for Further Research"}:
            verdict = "Weak Evidence"
        if _clean_reason(reason) and verdict == "Strong Candidate for Further Research":
            verdict = "Weak Evidence"

        scores_before_penalty.append(score)
        penalties.append(base_penalty)
        scores.append(score - base_penalty)
        verdicts.append(verdict)
        reasons.append(reason)

    out["ev_after_costs_2x"] = out["avg_forward_return"] - 2 * out["estimated_cost_return"]
    out["ev_after_costs_3x"] = out["avg_forward_return"] - 3 * out["estimated_cost_return"]
    out["cost_stress_2x_pass"] = out["ev_after_costs_2x"] > 0
    out["cost_stress_3x_pass"] = out["ev_after_costs_3x"] > 0
    out["score_before_penalties"] = scores_before_penalty
    out["multiple_testing_penalty"] = penalties
    out["research_score"] = scores
    out["verdict"] = verdicts
    out["primary_rejection_reason"] = reasons
    return out
