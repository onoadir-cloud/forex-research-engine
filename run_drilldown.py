from __future__ import annotations

import argparse

from src.candidate_drilldown import run_candidate_drilldown
from src.data_loader import load_ohlc_csv
from src.features import add_features
from src.regimes import add_regime_features
from src.sessions import add_session_features
from src.timeframe_builder import build_timeframes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--base-timeframe", required=True, choices=["M5", "M15", "M30"])
    p.add_argument("--pattern-id", required=True)
    p.add_argument("--spread-pips", type=float, default=0.0)
    p.add_argument("--slippage-pips", type=float, default=0.0)
    p.add_argument("--output-dir", default="drilldown_reports")
    args = p.parse_args()

    base_df, _ = load_ohlc_csv(args.csv)
    base_df["symbol"] = args.symbol
    base_df = add_regime_features(add_session_features(add_features(base_df)))
    tfs = build_timeframes(base_df[["datetime", "open", "high", "low", "close"]], args.base_timeframe)
    for k, df in list(tfs.items()):
        tfs[k] = add_regime_features(add_session_features(add_features(df)))

    outputs = run_candidate_drilldown(base_df, tfs, args.pattern_id, args.spread_pips, args.slippage_pips, args.output_dir)
    print(f"output paths: {outputs}")


if __name__ == "__main__":
    main()
