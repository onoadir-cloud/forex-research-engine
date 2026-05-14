from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

EXPECTED_FILES = [
    "README.md",
    "requirements.txt",
    "run_research.py",
    "src/__init__.py",
    "src/data_loader.py",
    "src/timeframe_builder.py",
    "src/features.py",
    "src/sessions.py",
    "src/regimes.py",
    "src/event_study.py",
    "src/scoring.py",
    "src/reporting.py",
    "src/export_mt5.py",
    "tests/test_data_loader.py",
    "tests/test_timeframe_builder.py",
    "tests/test_features.py",
    "tests/test_event_study.py",
    "tests/test_no_lookahead.py",
    "tests/test_scoring.py",
]

FORBIDDEN = [
    "profitable strategy",
    "guaranteed",
    "ready for live trading",
    "works in all markets",
]


def main() -> int:
    errors = []

    for rel in EXPECTED_FILES:
        if not (ROOT / rel).exists():
            errors.append(f"missing file: {rel}")

    req = (ROOT / "requirements.txt")
    if not req.exists():
        errors.append("requirements.txt missing")
    else:
        text = req.read_text(encoding="utf-8").lower()
        for pkg in ["pandas", "numpy", "pytest"]:
            if pkg not in text:
                errors.append(f"requirements missing package: {pkg}")

    for rel in ["README.md", "src/reporting.py"]:
        path = ROOT / rel
        if path.exists():
            txt = path.read_text(encoding="utf-8").lower()
            for phrase in FORBIDDEN:
                if phrase in txt:
                    errors.append(f"forbidden phrase found in {rel}: {phrase}")

    if errors:
        print("SMOKE TEST FAILED")
        for e in errors:
            print(f"- {e}")
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
