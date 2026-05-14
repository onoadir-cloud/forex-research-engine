from __future__ import annotations

import json
from pathlib import Path


def write_reports(symbol, base_timeframe, data_summary, scored_patterns, output_dir="reports"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{symbol}_patterns.csv"
    md_path = out / f"{symbol}_behavior_report.md"
    json_path = out / f"{symbol}_best_candidates.json"

    scored_patterns.to_csv(csv_path, index=False)
    best = scored_patterns[scored_patterns["verdict"].isin(["Interesting", "Strong Candidate for Further Research"])].sort_values("research_score", ascending=False)
    best_records = best.head(20).to_dict(orient="records")
    json_path.write_text(json.dumps(best_records, indent=2))

    weak = scored_patterns[scored_patterns["verdict"].isin(["Reject", "Weak Evidence"])]
    by_reason = weak.groupby("primary_rejection_reason").size().to_dict() if not weak.empty else {}
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
{best[['pattern_name','verdict','research_score']].head(10).to_markdown(index=False) if not best.empty else 'No candidates found.'}

## Rejected or weak patterns
{weak[['pattern_name','verdict','primary_rejection_reason']].head(20).to_markdown(index=False) if not weak.empty else 'None.'}

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
