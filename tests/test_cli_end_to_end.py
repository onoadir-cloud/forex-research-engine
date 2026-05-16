import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN = [
    "profitable strategy",
    "guaranteed",
    "ready for live trading",
    "works in all markets",
]


def test_cli_end_to_end(tmp_path):
    n = 2500
    dt = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(42)
    drift = 0.00002
    noise = rng.normal(0.0, 0.0002, size=n)
    close = 1.08 + np.cumsum(drift + noise)
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0.0004, 0.0001, size=n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    df = pd.DataFrame(
        {
            "datetime": dt,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )

    csv_path = tmp_path / "EURUSD_M15.csv"
    out_dir = tmp_path / "reports"
    df.to_csv(csv_path, index=False)

    cmd = [
        sys.executable,
        "run_research.py",
        "--csv",
        str(csv_path),
        "--symbol",
        "EURUSD",
        "--base-timeframe",
        "M15",
        "--spread-pips",
        "1.2",
        "--slippage-pips",
        "0.2",
        "--output-dir",
        str(out_dir),
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    report_md = out_dir / "EURUSD_behavior_report.md"
    patterns_csv = out_dir / "EURUSD_patterns.csv"
    candidates_json = out_dir / "EURUSD_best_candidates.json"

    assert report_md.exists()
    assert patterns_csv.exists()
    assert candidates_json.exists()

    patterns = pd.read_csv(patterns_csv)
    for col in ["pattern_id", "sample_size", "oos_sample_size", "ev_after_costs", "verdict"]:
        assert col in patterns.columns

    report_text = report_md.read_text(encoding="utf-8").lower()
    assert "## limitations" in report_text
    assert "## pattern definitions" in report_text
    assert "this is a historical event study, not proof of future performance." in report_text

    for phrase in FORBIDDEN:
        assert phrase not in report_text

    loaded_candidates = json.loads(candidates_json.read_text(encoding="utf-8"))
    assert isinstance(loaded_candidates, list)
    if not any((patterns["verdict"] == "Strong Candidate for Further Research")):
        assert "no strong research candidate was found" in report_text


def test_drilldown_cli_end_to_end(tmp_path):
    n = 2600
    dt = pd.date_range("2023-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(11)
    close = 1.07 + np.cumsum(rng.normal(0.00001, 0.0002, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0.0004, 0.0001, size=n))
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    df = pd.DataFrame({"datetime": dt, "open": open_, "high": high, "low": low, "close": close})

    csv_path = tmp_path / "EURUSD_M15.csv"
    out_dir = tmp_path / "drilldown_reports"
    df.to_csv(csv_path, index=False)

    from src.event_study import run_event_study
    from src.features import add_features
    from src.regimes import add_regime_features
    from src.sessions import add_session_features
    from src.timeframe_builder import build_timeframes
    prep = add_regime_features(add_session_features(add_features(df.copy())))
    tfs = build_timeframes(prep[["datetime", "open", "high", "low", "close"]], "M15")
    for k in tfs:
        tfs[k] = add_regime_features(add_session_features(add_features(tfs[k])))
    pid = run_event_study(prep, tfs, 1.2, 0.2).iloc[0]["pattern_id"]

    cmd = [sys.executable, "run_drilldown.py", "--csv", str(csv_path), "--symbol", "EURUSD", "--base-timeframe", "M15", "--pattern-id", pid, "--spread-pips", "1.2", "--slippage-pips", "0.2", "--output-dir", str(out_dir)]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    stem = out_dir / f"EURUSD_{pid}"
    md = Path(str(stem) + "_drilldown.md")
    events = Path(str(stem) + "_events.csv")
    monthly = Path(str(stem) + "_monthly.csv")
    grid = Path(str(stem) + "_tp_sl_grid.csv")
    summary = Path(str(stem) + "_summary.json")

    assert md.exists() and events.exists() and monthly.exists() and grid.exists() and summary.exists()
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data.get("pattern_id") == pid
