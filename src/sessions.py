from __future__ import annotations

import pandas as pd


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hours = out["datetime"].dt.hour
    out["hour"] = hours
    out["day_of_week"] = out["datetime"].dt.dayofweek

    def session(h: int) -> str:
        if 0 <= h < 7:
            return "Asia"
        if 7 <= h < 12:
            return "London"
        if 12 <= h < 16:
            return "LondonNewYorkOverlap"
        if 16 <= h < 21:
            return "NewYork"
        return "Other"

    out["session"] = hours.map(session)
    return out
