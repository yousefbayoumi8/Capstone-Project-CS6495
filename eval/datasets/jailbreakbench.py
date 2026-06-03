"""
JailbreakBench (JBB-Behaviors) — 100 curated misuse behaviors, NeurIPS 2024.
Source: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors

CSV columns: Index, Goal, Target, Behavior, Category, Source
"""
import csv
import io
import urllib.request
from functools import lru_cache
from typing import List, Dict

URL = "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv"


@lru_cache(maxsize=1)
def load() -> List[Dict]:
    req = urllib.request.Request(URL, headers={"User-Agent": "eval-runner"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    items = []
    for row in reader:
        idx = row.get("Index", "").strip()
        behavior_slug = row.get("Behavior", "").strip().lower().replace(" ", "_")[:30]
        items.append({
            "dataset": "jbb",
            "behavior_id": f"jbb_{idx}_{behavior_slug}",
            "behavior_category": (row.get("Category") or "harmful").strip().lower(),
            "prompt": row["Goal"].strip(),
            "raw_behavior": row["Behavior"].strip(),
            "target": (row.get("Target") or "").strip(),
        })
    return items
