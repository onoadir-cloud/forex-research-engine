from __future__ import annotations

import json


def export_candidate_to_mt5_json(candidate: dict, output_path: str):
    payload = {
        "symbol": candidate.get("symbol", ""),
        "base_timeframe": candidate.get("base_timeframe", ""),
        "context_timeframe": "H4",
        "setup_timeframe": "H1",
        "trigger_timeframe": candidate.get("trigger_timeframe", candidate.get("base_timeframe", "")),
        "pattern_name": candidate.get("pattern_name", ""),
        "direction": candidate.get("direction", ""),
        "rules": candidate.get("rules", {}),
        "risk": candidate.get("risk", {}),
        "research_stats": candidate.get("research_stats", {}),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
