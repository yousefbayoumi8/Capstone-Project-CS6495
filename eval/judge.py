"""
Phase 2 — judging.

Reads generations.jsonl, scores each response with the chosen judge, and
writes judgments.jsonl. Runs AFTER all target models are unloaded; the
HarmBench classifier is the only thing in VRAM (~5 GB at 4-bit).

Live monitoring:
  - tqdm bar shows progress + ETA.
  - `tail -f <run>/judgments.jsonl` for record-by-record live log.
  - `tail -f <run>/judge.log` for the same lines that go to the console.

Examples:
    python -m eval.judge --run eval/runs/2026-05-28
    python -m eval.judge --run eval/runs/2026-05-28 --judge substring
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.judges import substring as substring_judge


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def already_judged_keys(path: Path, judge_name: str) -> Set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("judge") == judge_name:
                    keys.add((r["model"], r["system_prompt_id"], r["dataset"],
                              r["behavior_id"]))
            except Exception:
                continue
    return keys


def write_judgment(out_f, gen: Dict, judge_name: str, verdict: Dict):
    record = {
        "ts": iso_now(),
        "run_id": gen.get("run_id"),
        "model": gen["model"],
        "system_prompt_id": gen["system_prompt_id"],
        "dataset": gen["dataset"],
        "behavior_id": gen["behavior_id"],
        "behavior_category": gen.get("behavior_category"),
        "judge": judge_name,
        "label": verdict["label"],
        "judge_raw": verdict.get("raw") or verdict.get("matched") or "",
        "judge_confidence": verdict.get("confidence"),
        "latency_ms_generation": gen.get("latency_ms"),
    }
    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_f.flush()


class TeeLogger:
    def __init__(self, path: Path):
        self.f = path.open("a", encoding="utf-8")

    def log(self, msg: str):
        line = f"[{iso_now()}] {msg}"
        tqdm.write(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="Run directory (contains generations.jsonl).")
    parser.add_argument("--judge", default="harmbench_cls",
                        choices=["harmbench_cls", "substring"])
    parser.add_argument("--no-quantize", action="store_true",
                        help="Load judge in FP16 (HarmBench classifier only).")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide the tqdm bar.")
    args = parser.parse_args()

    run_dir = Path(args.run)
    in_path = run_dir / "generations.jsonl"
    out_path = run_dir / "judgments.jsonl"
    log_path = run_dir / "judge.log"

    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}")

    logger = TeeLogger(log_path)

    logger.log(f"Run: {run_dir}")
    logger.log(f"Judge: {args.judge}")

    done = already_judged_keys(out_path, args.judge)
    if done:
        logger.log(f"Resuming: {len(done)} records already judged by {args.judge}.")

    gens = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                gens.append(json.loads(line))
            except Exception:
                continue
    logger.log(f"Loaded {len(gens)} generations.")

    pending = [g for g in gens
               if (g["model"], g["system_prompt_id"], g["dataset"],
                   g["behavior_id"]) not in done
               and "response" in g]
    logger.log(f"Pending judgments: {len(pending)}")

    judge_obj = None
    if args.judge == "harmbench_cls":
        from eval.judges.harmbench_cls import HarmBenchJudge
        load_t0 = time.perf_counter()
        judge_obj = HarmBenchJudge(quantize=not args.no_quantize)
        logger.log(f"Judge loaded in {int(time.perf_counter() - load_t0)}s.")

    run_start = time.perf_counter()
    label_counts: Dict[str, int] = {}

    try:
        with out_path.open("a", encoding="utf-8") as out_f:
            bar = tqdm(
                pending,
                desc=f"judging ({args.judge})",
                unit="rec",
                smoothing=0.1,
                disable=args.quiet,
                ncols=110,
            )
            for gen in bar:
                behavior = gen.get("raw_behavior") or gen["prompt"]
                response = gen["response"]

                if args.judge == "substring":
                    verdict = substring_judge.classify(behavior, response)
                else:
                    try:
                        verdict = judge_obj.classify(behavior, response)
                    except Exception as e:
                        verdict = {"label": "error",
                                   "raw": f"{type(e).__name__}: {e}"}

                label_counts[verdict["label"]] = label_counts.get(verdict["label"], 0) + 1
                write_judgment(out_f, gen, args.judge, verdict)

                # rolling tally on the bar
                bar.set_postfix_str(
                    " ".join(f"{k}={v}" for k, v in sorted(label_counts.items())),
                    refresh=False,
                )
            bar.close()
    finally:
        if judge_obj is not None:
            judge_obj.unload()
        elapsed = int(time.perf_counter() - run_start)
        logger.log(f"Done in {elapsed//60}m {elapsed%60}s.")
        logger.log(f"Label tally: {label_counts}")
        logger.log(f"Output: {out_path}")
        logger.close()


if __name__ == "__main__":
    main()
