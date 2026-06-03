"""
Read judgments.jsonl and print an Attack Success Rate table grouped by
(model, dataset, system_prompt) with a delta column showing what the
defensive system prompt added.

Usage:
    python -m eval.summary --run eval/runs/2026-05-28
    python -m eval.summary --run eval/runs/2026-05-28 --judge substring
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="Run directory containing judgments.jsonl.")
    parser.add_argument("--judge", default="harmbench_mistral7b",
                        help="Which judge's records to summarize "
                             "(harmbench_mistral7b or substring).")
    parser.add_argument("--baseline", default="permissive",
                        help="System prompt used as the baseline for the delta column.")
    args = parser.parse_args()

    in_path = Path(args.run) / "judgments.jsonl"
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}")

    # counts[(model, dataset, sp)] -> {label: count}
    counts = defaultdict(lambda: defaultdict(int))
    with in_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("judge") != args.judge:
                continue
            key = (r["model"], r["dataset"], r["system_prompt_id"])
            counts[key][r["label"]] += 1

    if not counts:
        raise SystemExit(f"No records with judge='{args.judge}' in {in_path}.")

    # Group by (model, dataset) so we can compute deltas between sps.
    grouped = defaultdict(dict)  # (model, dataset) -> {sp: counts}
    for (model, dataset, sp), lc in counts.items():
        grouped[(model, dataset)][sp] = lc

    # Collect all sps we saw (excluding baseline) for the delta column header.
    all_sps = sorted({sp for inner in grouped.values() for sp in inner})
    other_sps = [sp for sp in all_sps if sp != args.baseline]

    # Header.
    cols = ["model", "dataset", "sp", "n", "complied", "refused", "ASR%"]
    widths = [8, 11, 22, 4, 9, 9, 7]
    print("Judge: " + args.judge)
    print()
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))

    overall_baseline = []
    overall_others = defaultdict(list)

    for (model, dataset), per_sp in sorted(grouped.items()):
        for sp in sorted(per_sp):
            lc = per_sp[sp]
            total = sum(lc.values())
            complied = lc.get("complied", 0)
            refused = lc.get("refused", 0)
            asr = (complied / total * 100.0) if total else 0.0
            row = [
                model, dataset, sp,
                str(total), str(complied), str(refused),
                f"{asr:5.1f}",
            ]
            print("  ".join(s.ljust(w) for s, w in zip(row, widths)))

            if sp == args.baseline:
                overall_baseline.append(asr)
            else:
                overall_others[sp].append(asr)

        # delta line per (model, dataset) if both sps present
        baseline = per_sp.get(args.baseline)
        for sp in other_sps:
            other = per_sp.get(sp)
            if baseline and other:
                base_asr = baseline.get("complied", 0) / max(sum(baseline.values()), 1) * 100
                othr_asr = other.get("complied", 0) / max(sum(other.values()), 1) * 100
                delta = othr_asr - base_asr
                sign = "+" if delta >= 0 else ""
                arrow = "↑ worse" if delta > 0 else ("↓ better" if delta < 0 else "= same")
                print(f"    Δ {args.baseline} → {sp}: {sign}{delta:.1f}pp  ({arrow})")
        print()

    # ─── overall summary ───
    if overall_baseline:
        avg_base = sum(overall_baseline) / len(overall_baseline)
        print(f"Mean ASR across all cells: {args.baseline} = {avg_base:.1f}%")
        for sp, lst in overall_others.items():
            if lst:
                avg = sum(lst) / len(lst)
                print(f"                            {sp} = {avg:.1f}%  "
                      f"(Δ {avg - avg_base:+.1f}pp)")


if __name__ == "__main__":
    main()
