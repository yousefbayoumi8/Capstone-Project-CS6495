"""
Read judgments.jsonl files and print an Attack Success Rate table grouped by
(model, dataset, defense) with a delta column showing each defense's effect
vs the baseline.

Two ways to invoke:
  --runs-dir <parent>   aggregate across every subfolder with judgments.jsonl
                        (use this with the per-defense folder layout)
  --run <single_dir>    single folder (backwards compatible)

Reads either `defense_id` (new) or `system_prompt_id` (legacy); legacy
"permissive" is treated as "none".

Usage:
    python -m eval.summary --runs-dir eval/runs
    python -m eval.summary --runs-dir eval/runs --judge substring
    python -m eval.summary --run eval/runs/none
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List


LEGACY_SP_TO_DEFENSE = {"permissive": "none"}


def _defense_id_of(record: dict) -> str:
    if "defense_id" in record:
        return record["defense_id"]
    sp = record.get("system_prompt_id", "")
    return LEGACY_SP_TO_DEFENSE.get(sp, sp)


def collect_judgment_paths(args) -> List[Path]:
    if args.run:
        p = Path(args.run) / "judgments.jsonl"
        return [p] if p.exists() else []
    parent = Path(args.runs_dir)
    return sorted(p / "judgments.jsonl" for p in parent.iterdir()
                  if p.is_dir() and (p / "judgments.jsonl").exists())


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Single run directory.")
    group.add_argument("--runs-dir", help="Parent directory; aggregates across "
                                          "every subfolder with judgments.jsonl.")
    parser.add_argument("--judge", default="harmbench_cls",
                        help="Judge whose records to summarize "
                             "(harmbench_cls or substring).")
    parser.add_argument("--baseline", default="none",
                        help="Defense used as the baseline for the delta column.")
    args = parser.parse_args()

    paths = collect_judgment_paths(args)
    if not paths:
        raise SystemExit("No judgments.jsonl found.")

    counts = defaultdict(lambda: defaultdict(int))
    for jpath in paths:
        with jpath.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("judge") != args.judge:
                    continue
                key = (r["model"], r["dataset"], _defense_id_of(r))
                counts[key][r["label"]] += 1

    if not counts:
        raise SystemExit(f"No records with judge='{args.judge}' in scanned files.")

    grouped = defaultdict(dict)  # (model, dataset) -> {defense: counts}
    for (model, dataset, defense), lc in counts.items():
        grouped[(model, dataset)][defense] = lc

    all_defenses = sorted({d for inner in grouped.values() for d in inner})
    other_defenses = [d for d in all_defenses if d != args.baseline]

    cols = ["model", "dataset", "defense", "n", "complied", "refused", "ASR%"]
    widths = [8, 11, 22, 4, 9, 9, 7]
    print("Judge: " + args.judge)
    print("Baseline defense: " + args.baseline)
    print("Scanned files: " + ", ".join(str(p) for p in paths))
    print()
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))

    overall_baseline = []
    overall_others = defaultdict(list)

    for (model, dataset), per_def in sorted(grouped.items()):
        for defense in sorted(per_def):
            lc = per_def[defense]
            total = sum(lc.values())
            complied = lc.get("complied", 0)
            refused = lc.get("refused", 0)
            asr = (complied / total * 100.0) if total else 0.0
            row = [
                model, dataset, defense,
                str(total), str(complied), str(refused),
                f"{asr:5.1f}",
            ]
            print("  ".join(s.ljust(w) for s, w in zip(row, widths)))

            if defense == args.baseline:
                overall_baseline.append(asr)
            else:
                overall_others[defense].append(asr)

        baseline = per_def.get(args.baseline)
        for defense in other_defenses:
            other = per_def.get(defense)
            if baseline and other:
                base_asr = baseline.get("complied", 0) / max(sum(baseline.values()), 1) * 100
                othr_asr = other.get("complied", 0) / max(sum(other.values()), 1) * 100
                delta = othr_asr - base_asr
                sign = "+" if delta >= 0 else ""
                arrow = "↑ worse" if delta > 0 else ("↓ better" if delta < 0 else "= same")
                print(f"    Δ {args.baseline} → {defense}: {sign}{delta:.1f}pp  ({arrow})")
        print()

    if overall_baseline:
        avg_base = sum(overall_baseline) / len(overall_baseline)
        print(f"Mean ASR across all cells: {args.baseline} = {avg_base:.1f}%")
        for defense, lst in overall_others.items():
            if lst:
                avg = sum(lst) / len(lst)
                print(f"                            {defense} = {avg:.1f}%  "
                      f"(Δ {avg - avg_base:+.1f}pp)")


if __name__ == "__main__":
    main()
