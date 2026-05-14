from __future__ import annotations

import argparse

from src.data_loader import load_ohlc_csv
from src.event_study import run_event_study
from src.features import add_features
from src.regimes import add_regime_features
from src.reporting import write_reports
from src.scoring import score_patterns
from src.sessions import add_session_features
from src.timeframe_builder import build_timeframes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--base-timeframe", required=True, choices=["M5", "M15", "M30"])
    p.add_argument("--spread-pips", type=float, default=0.0)
    p.add_argument("--slippage-pips", type=float, default=0.0)
    p.add_argument("--output-dir", default="reports")
    args = p.parse_args()

    base_df, warns = load_ohlc_csv(args.csv)
    base_df = add_regime_features(add_session_features(add_features(base_df)))
    tfs = build_timeframes(base_df[["datetime", "open", "high", "low", "close"]], args.base_timeframe)
    for k, df in list(tfs.items()):
        tfs[k] = add_regime_features(add_session_features(add_features(df)))

    patterns = run_event_study(base_df, tfs, args.spread_pips, args.slippage_pips)
    scored = score_patterns(patterns, len(patterns)) if not patterns.empty else patterns

    summary = {
        "rows": len(base_df),
        "start": base_df["datetime"].min(),
        "end": base_df["datetime"].max(),
        "generated_timeframes": list(tfs.keys()),
        "warnings": warns,
    }
    outputs = write_reports(args.symbol, args.base_timeframe, summary, scored, args.output_dir)

    interesting = int((scored["verdict"] == "Interesting").sum()) if not scored.empty else 0
    strong = int((scored["verdict"] == "Strong Candidate for Further Research").sum()) if not scored.empty else 0
    print(f"rows loaded: {len(base_df)}")
    print(f"date range: {summary['start']} -> {summary['end']}")
    print(f"patterns scanned: {len(scored)}")
    print(f"interesting: {interesting}, strong candidate: {strong}")
    print(f"output paths: {outputs}")


if __name__ == "__main__":
    main()
