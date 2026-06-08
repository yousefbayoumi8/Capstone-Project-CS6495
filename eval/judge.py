"""
Phase 2 — judging.

Two ways to invoke:
  --run <single_defense_dir>   judges just that one folder's generations.jsonl
  --runs-dir <parent_dir>      finds every subfolder with generations.jsonl
                               and judges each, loading the judge model ONCE

Writes `judgments.jsonl` next to each `generations.jsonl`. Resumable —
re-running skips records already judged by the same judge.

Examples:
    python -m eval.judge --runs-dir eval/runs
    python -m eval.judge --run eval/runs/rules
    python -m eval.judge --run eval/runs/none --judge substring
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.judges import substring as substring_judge


LEGACY_SP_TO_DEFENSE = {"permissive": "none"}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _defense_id_of(record: Dict) -> str:
    if "defense_id" in record:
        return record["defense_id"]
    sp = record.get("system_prompt_id", "")
    return LEGACY_SP_TO_DEFENSE.get(sp, sp)


def already_judged_keys(path: Path, judge_name: str) -> Set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("judge") == judge_name:
                    keys.add((r["model"], _defense_id_of(r), r["dataset"],
                              r["behavior_id"]))
            except Exception:
                continue
    return keys


def write_judgment(out_f, gen: Dict, judge_name: str, verdict: Dict):
    record = {
        "ts": iso_now(),
        "run_id": gen.get("run_id"),
        "model": gen["model"],
        "defense_id": _defense_id_of(gen),
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


def collect_pending(run_dir: Path, judge_name: str) -> List[Dict]:
    """Read generations.jsonl from one folder; return list of pending generations
    (not yet judged by judge_name, and not error records)."""
    gen_path = run_dir / "generations.jsonl"
    jud_path = run_dir / "judgments.jsonl"
    if not gen_path.exists():
        return []

    done = already_judged_keys(jud_path, judge_name)
    pending = []
    with gen_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                g = json.loads(line)
            except Exception:
                continue
            if "response" not in g:
                continue
            key = (g["model"], _defense_id_of(g), g["dataset"], g["behavior_id"])
            if key in done:
                continue
            pending.append(g)
    return pending


def judge_folder(run_dir: Path, judge_name: str, judge_obj, logger: TeeLogger,
                 quiet: bool, label_counts: Dict[str, int]):
    """Judge one folder's pending generations. Writes to judgments.jsonl."""
    pending = collect_pending(run_dir, judge_name)
    if not pending:
        logger.log(f"  {run_dir.name}: no pending records")
        return

    out_path = run_dir / "judgments.jsonl"
    logger.log(f"  {run_dir.name}: {len(pending)} pending")

    with out_path.open("a", encoding="utf-8") as out_f:
        bar = tqdm(
            pending,
            desc=f"{run_dir.name}",
            unit="rec",
            smoothing=0.1,
            disable=quiet,
            ncols=110,
        )
        for gen in bar:
            behavior = gen.get("raw_behavior") or gen["prompt"]
            response = gen["response"]

            if judge_name == "substring":
                verdict = substring_judge.classify(behavior, response)
            else:
                try:
                    verdict = judge_obj.classify(behavior, response)
                except Exception as e:
                    verdict = {"label": "error",
                               "raw": f"{type(e).__name__}: {e}"}

            label_counts[verdict["label"]] = label_counts.get(verdict["label"], 0) + 1
            write_judgment(out_f, gen, judge_name, verdict)
            bar.set_postfix_str(
                " ".join(f"{k}={v}" for k, v in sorted(label_counts.items())),
                refresh=False,
            )
        bar.close()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Single run directory (contains generations.jsonl).")
    group.add_argument("--runs-dir", help="Parent directory; judges every subfolder "
                                          "containing generations.jsonl, "
                                          "loading the judge model once.")
    parser.add_argument("--judge", default="harmbench_cls",
                        choices=["harmbench_cls", "substring"])
    parser.add_argument("--no-quantize", action="store_true",
                        help="Load judge in FP16 (HarmBench classifier only).")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide the tqdm bar.")
    args = parser.parse_args()

    # Resolve the list of folders to judge.
    if args.run:
        folders = [Path(args.run)]
        log_path = folders[0] / "judge.log"
    else:
        parent = Path(args.runs_dir)
        folders = [p for p in sorted(parent.iterdir())
                   if p.is_dir() and (p / "generations.jsonl").exists()]
        log_path = parent / "judge.log"

    if not folders:
        raise SystemExit("No folders with generations.jsonl found.")

    logger = TeeLogger(log_path)
    logger.log(f"Judge: {args.judge}")
    logger.log(f"Folders to judge: {[f.name for f in folders]}")

    judge_obj = None
    if args.judge == "harmbench_cls":
        from eval.judges.harmbench_cls import HarmBenchJudge
        load_t0 = time.perf_counter()
        judge_obj = HarmBenchJudge(quantize=not args.no_quantize)
        logger.log(f"Judge loaded in {int(time.perf_counter() - load_t0)}s.")

    run_start = time.perf_counter()
    label_counts: Dict[str, int] = {}

    try:
        for folder in folders:
            judge_folder(folder, args.judge, judge_obj, logger, args.quiet,
                         label_counts)
    finally:
        if judge_obj is not None:
            judge_obj.unload()
        elapsed = int(time.perf_counter() - run_start)
        logger.log(f"Done in {elapsed//60}m {elapsed%60}s.")
        logger.log(f"Label tally: {label_counts}")
        logger.close()


if __name__ == "__main__":
    main()
