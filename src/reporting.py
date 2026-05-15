from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _simple_markdown_table(df: pd.DataFrame, columns: list[str], empty_text: str) -> str:
    if df.empty:
        return empty_text
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return empty_text
    view = df[existing].copy()
    header = "| " + " | ".join(existing) + " |"
    separator = "| " + " | ".join(["---"] * len(existing)) + " |"
    rows = ["| " + " | ".join(_format_cell(row[col]) for col in existing) + " |" for _, row in view.iterrows()]
    return "\n".join([header, separator, *rows])


def write_reports(symbol, base_timeframe, data_summary, scored_patterns, output_dir="reports"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{symbol}_patterns.csv"
    md_path = out / f"{symbol}_behavior_report.md"
    json_path = out / f"{symbol}_best_candidates.json"

    scored_patterns.to_csv(csv_path, index=False)

    strong = scored_patterns[scored_patterns.get("verdict", pd.Series(dtype=str)) == "Strong Candidate for Further Research"]
    watchlist = scored_patterns[scored_patterns.get("verdict", pd.Series(dtype=str)) == "Interesting"]
    weak = scored_patterns[scored_patterns.get("verdict", pd.Series(dtype=str)).isin(["Reject", "Weak Evidence"])]

    if "research_score" in scored_patterns.columns:
        strong = strong.sort_values("research_score", ascending=False)
        watchlist = watchlist.sort_values("research_score", ascending=False)

    best_records = pd.concat([strong, watchlist], axis=0).head(20).to_dict(orient="records")
    json_path.write_text(json.dumps(best_records, indent=2, default=str))

    if not weak.empty and "primary_rejection_reason" in weak.columns:
        reasons = weak["primary_rejection_reason"].astype(str).str.strip()
        filtered = weak.loc[reasons != ""].copy()
        by_reason = filtered.groupby("primary_rejection_reason").size().to_dict() if not filtered.empty else {}
    else:
        by_reason = {}
    no_strong_msg = "No strong research candidate was found." if strong.empty else ""
    only_weak_msg = "Only weak historical tendencies were found. These are not sufficient for strategy construction." if strong.empty and watchlist.empty else ""

    md = f"""# {symbol} behavior research report

## Data summary
- Date range: {data_summary['start']} to {data_summary['end']}
- Base timeframe: {base_timeframe}
- Candles: {data_summary['rows']}
- Generated timeframes: {', '.join(data_summary['generated_timeframes'])}
- Patterns scanned: {len(scored_patterns)}

## Rejection summary
{json.dumps(by_reason, indent=2)}

## Behavior profile
This report reviews historical tendency only and identifies conservative research candidates under reliability constraints.

## Strong candidates
{no_strong_msg}
{_simple_markdown_table(strong.head(10), ["pattern_name", "verdict", "research_score"], "None.")}

## Research watchlist
{_simple_markdown_table(watchlist.head(10), ["pattern_name", "verdict", "research_score"], "None.")}

## Weak/rejected patterns
{only_weak_msg}
{_simple_markdown_table(weak.head(20), ["pattern_name", "verdict", "primary_rejection_reason"], "None.")}

## Pattern definitions
- Downtrend: `close < sma200` and `sma50_slope < 0` (with trend inputs available).
- Uptrend: `close > sma200` and `sma50_slope > 0` (with trend inputs available).
- Range: trend inputs available, but neither Uptrend nor Downtrend.
- Low volatility: `atr_percentile_200 < 0.33`.
- Normal volatility: `0.33 <= atr_percentile_200 <= 0.66`.
- High volatility: `atr_percentile_200 > 0.66`.
- Compression: `range_compression_20` below its rolling 100-bar 20th percentile.
- Neutral: default setup/trigger state when no rule overrides.
- MomentumPositive: `return_1 > 0`.
- MomentumNegative: `return_1 < 0`.
- RsiRecoveryLow: `rsi14 < 30`.
- RsiRejectionHigh: `rsi14 > 70`.
- BreakoutUp20: `breakout_20_up` is true.
- BreakdownDown20: `breakout_20_down` is true.
- Sessions: Asia (00:00-06:59), London (07:00-11:59), LondonNewYorkOverlap (12:00-15:59), NewYork (16:00-20:59), Other (remaining hours).

## Limitations
- This is a historical event study, not proof of future performance.
- No tick data.
- No variable spread by news/session unless included in data.
- No swap/overnight financing yet.
- Multiple testing risk exists.
- A pattern can only become a strategy after further validation.
- Results are not suitable for live trading yet and require further validation.
"""
    md_path.write_text(md)
    return {"markdown": str(md_path), "csv": str(csv_path), "json": str(json_path)}
