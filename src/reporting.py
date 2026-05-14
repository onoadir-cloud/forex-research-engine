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
    """Render a small markdown table without pandas' optional tabulate dependency."""
    if df.empty:
        return empty_text
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return empty_text

    view = df[existing].copy()
    header = "| " + " | ".join(existing) + " |"
    separator = "| " + " | ".join(["---"] * len(existing)) + " |"
    rows = []
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(_format_cell(row[col]) for col in existing) + " |")
    return "\n".join([header, separator, *rows])


def write_reports(symbol, base_timeframe, data_summary, scored_patterns, output_dir="reports"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{symbol}_patterns.csv"
    md_path = out / f"{symbol}_behavior_report.md"
    json_path = out / f"{symbol}_best_candidates.json"

    scored_patterns.to_csv(csv_path, index=False)

    if "verdict" in scored_patterns.columns:
        best = scored_patterns[
            scored_patterns["verdict"].isin(["Interesting", "Strong Candidate for Further Research"])
        ]
        if "research_score" in best.columns:
            best = best.sort_values("research_score", ascending=False)
        weak = scored_patterns[scored_patterns["verdict"].isin(["Reject", "Weak Evidence"])]
    else:
        best = scored_patterns.iloc[0:0].copy()
        weak = scored_patterns.iloc[0:0].copy()

    best_records = best.head(20).to_dict(orient="records")
    json_path.write_text(json.dumps(best_records, indent=2, default=str))

    if not weak.empty and "primary_rejection_reason" in weak.columns:
        by_reason = weak.groupby("primary_rejection_reason").size().to_dict()
    else:
        by_reason = {}

    best_table = _simple_markdown_table(
        best.head(10),
        ["pattern_name", "verdict", "research_score"],
        "No candidates found.",
    )
    weak_table = _simple_markdown_table(
        weak.head(20),
        ["pattern_name", "verdict", "primary_rejection_reason"],
        "None.",
    )

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
This report reviews historical tendency only and identifies research candidate patterns with weak evidence handling.

## Best patterns
{best_table}

## Rejected or weak patterns
{weak_table}

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
