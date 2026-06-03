"""
AdvBench (Zou et al. 2023) — 520 harmful behaviors as instructions.
Source: https://github.com/llm-attacks/llm-attacks
"""
import csv
import io
import urllib.request
from functools import lru_cache
from typing import List, Dict

URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


@lru_cache(maxsize=1)
def load() -> List[Dict]:
    with urllib.request.urlopen(URL, timeout=30) as r:
        text = r.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    items = []
    for i, row in enumerate(reader):
        items.append({
            "dataset": "advbench",
            "behavior_id": f"advbench_{i:04d}",
            "behavior_category": "harmful_behavior",
            "prompt": row["goal"].strip(),
            "target": row.get("target", "").strip(),  # optional: desired complying response
        })
    return items
