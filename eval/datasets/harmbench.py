"""
HarmBench (CAIS, 2024) — standardized red-teaming benchmark with ~400 behaviors.
Source: https://github.com/centerforaisafety/HarmBench

We pull the canonical behaviors CSV from the upstream repo. Only "standard" and
"contextual" categories are used here (skipping copyright/multimodal — those
need different scoring).
"""
import csv
import io
import urllib.request
from functools import lru_cache
from typing import List, Dict

URL = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"


@lru_cache(maxsize=1)
def load() -> List[Dict]:
    with urllib.request.urlopen(URL, timeout=30) as r:
        text = r.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    items = []
    for row in reader:
        category = row.get("FunctionalCategory", "").strip().lower()
        if category in ("copyright", "multimodal"):
            continue  # different scoring methodology
        # contextual behaviors include a context paragraph appended to the prompt
        prompt = row["Behavior"].strip()
        context = (row.get("ContextString") or "").strip()
        if context:
            prompt = f"{context}\n\n{prompt}"
        items.append({
            "dataset": "harmbench",
            "behavior_id": row["BehaviorID"].strip(),
            "behavior_category": category or "standard",
            "prompt": prompt,
            "raw_behavior": row["Behavior"].strip(),  # without the context, for the judge
        })
    return items
