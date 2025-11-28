#!/usr/bin/env python3
import json
from pathlib import Path

ATT_PROMPT_COUNT = 427  # divisor for average_cost_usd
RESULT_DIR = Path("result")

for path in RESULT_DIR.glob("*_att_metrics.json"):
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    total_cost = data.get("total_cost_usd")
    if total_cost is not None:
        data["average_cost_usd"] = total_cost / ATT_PROMPT_COUNT

    data.pop("average_cost_per_set_usd", None)
    data.pop("average_time_per_set_seconds", None)

    if "total_time_seconds" in data:
        data["total_latency_seconds"] = data.pop("total_time_seconds")

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")