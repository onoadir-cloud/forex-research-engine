# Forex Research Engine (Phase 1)

A Python CLI research engine for mathematical forex behavior analysis using historical OHLC candles.

## What this project does
- Loads historical OHLC CSV data.
- Builds higher timeframes from a base timeframe.
- Computes deterministic mathematical features.
- Classifies sessions and market regimes.
- Runs a historical event study over state combinations.
- Scores patterns as research candidates.
- Exports markdown/csv/json research reports.

## What this project does not do
- Not a trading robot.
- Not a signal bot.
- Not an MT5 EA yet.
- No UI/dashboard.
- No broker or MT5 connection.
- No web API usage.
- No machine learning in phase 1.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python run_research.py --csv data/EURUSD_M15.csv --symbol EURUSD --base-timeframe M15 --spread-pips 1.2 --slippage-pips 0.2
```


## Testing
```bash
pip install -r requirements.txt
python -m pytest
```

## CSV formats
Supported inputs:
1. `datetime,open,high,low,close`
2. `date,time,open,high,low,close`

Timestamps are treated as UTC-naive unless timezone information exists.

## Generated reports
- `reports/{symbol}_behavior_report.md`
- `reports/{symbol}_patterns.csv`
- `reports/{symbol}_best_candidates.json`

## Research limitations
- Historical tendency only; weak evidence requires further validation.
- This is not suitable for live trading yet.
- No tick-level modeling.
- No variable spread modeling unless provided indirectly in data.
- No swap/overnight financing in phase 1.
- Multiple testing risk exists.

## Future path to MT5 EA
- Select stable research candidates after additional validation.
- Define strict execution/risk rules.
- Then map candidate definitions into MT5 EA logic in a later phase.



## End-to-end smoke run
```bash
python run_research.py --csv data/EURUSD_M15.csv --symbol EURUSD --base-timeframe M15 --spread-pips 1.2 --slippage-pips 0.2 --output-dir reports
```
=======

