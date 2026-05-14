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
    assert "this is a historical event study, not proof of future performance." in report_text

    for phrase in FORBIDDEN:
        assert phrase not in report_text

    loaded_candidates = json.loads(candidates_json.read_text(encoding="utf-8"))
    assert isinstance(loaded_candidates, list)
