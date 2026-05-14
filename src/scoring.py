from __future__ import annotations

import math
import pandas as pd


def score_patterns(patterns_df: pd.DataFrame, tested_patterns_count: int) -> pd.DataFrame:
    if patterns_df.empty:
        return patterns_df
    out = patterns_df.copy()
    penalty = math.log(max(tested_patterns_count, 1) + 1)
    scores = []
    verdicts = []
    reasons = []
    for _, r in out.iterrows():
        score = 50.0 - penalty * 3
        reason = ""
        if r["sample_size"] >= 150:
            score += 20
        elif r["sample_size"] < 80:
            score -= 20
            reason = "Low sample size"
        if r["oos_sample_size"] < 30:
            score -= 15
            reason = reason or "Low OOS sample size"
        if r["oos_ev_after_costs"] > 0:
            score += 20
        else:
            score -= 25
            reason = reason or "Non-positive OOS EV after costs"
        if r["ev_after_costs"] > 0 and r["oos_ev_after_costs"] < 0:
            score -= 25
            reason = "IS positive but OOS negative"
        if r["oos_agrees_with_is"]:
            score += 10
        else:
            score -= 15
            reason = reason or "IS/OOS direction conflict"

        verdict = "Reject"
        if score >= 80 and r["sample_size"] >= 150 and r["oos_sample_size"] >= 50 and r["ev_after_costs"] > 0 and r["oos_ev_after_costs"] > 0 and r["oos_agrees_with_is"]:
            verdict = "Strong Candidate for Further Research"
        elif score >= 60 and r["oos_ev_after_costs"] > 0:
            verdict = "Interesting"
        elif score >= 40:
            verdict = "Weak Evidence"

        if r["sample_size"] < 80 or r["oos_sample_size"] < 30:
            if verdict in ("Interesting", "Strong Candidate for Further Research"):
                verdict = "Weak Evidence"
        if r["ev_after_costs"] > 0 and r["oos_ev_after_costs"] < 0:
            verdict = "Reject"

        scores.append(score)
        verdicts.append(verdict)
        reasons.append(reason or "Insufficient evidence")

    out["research_score"] = scores
    out["verdict"] = verdicts
    out["primary_rejection_reason"] = reasons
    return out
