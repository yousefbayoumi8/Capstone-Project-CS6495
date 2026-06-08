"""
Phase 1 — generation.

Per-defense folder layout: each defense gets its own subdirectory under
`--output`, containing `generations.jsonl` (and later `judgments.jsonl` from
the judge phase). A single `run.log` lives at the parent for the whole
invocation.

    <output>/
      <defense_id>/
        generations.jsonl
      run.log

Models run sequentially — only ONE in VRAM at a time. When more models are
run later, their records get appended to the existing per-defense
generations.jsonl files. Re-running the same `--output` is safe and skips
records already written.

Backwards compatibility: records that used the pre-refactor `system_prompt_id`
field are accepted on read, with `system_prompt_id="permissive"` treated as
`defense_id="none"`. Rename your existing `full_permissive/` directory to
`none/` to plug it into the new layout without re-running.

Live monitoring:
  - tqdm bar shows per-model progress + ETA.
  - `tail -f <output>/<defense>/generations.jsonl` for record-by-record log.
  - `tail -f <output>/run.log` for the console log.

Examples:
    python -m eval.runner --output eval/runs --models qwen
    python -m eval.runner --output eval/runs --models qwen --defenses rules,sandwich
    python -m eval.runner --output eval/runs --models qwen --auto-judge harmbench_cls
"""
import argparse
import gc
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_loader import load_model, generate_response, MODELS
from eval import datasets as ds_module
from eval import defenses as defenses_module


# Map legacy system_prompt_id values to the new defense_id namespace so old
# data can be reused. Rename eval/runs/full_permissive → eval/runs/none and
# the resume logic + judge will pick it up as the "none" baseline.
LEGACY_SP_TO_DEFENSE = {
    "permissive": "none",
}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _defense_id_of(record: dict) -> str | None:
    if "defense_id" in record:
        return record["defense_id"]
    sp = record.get("system_prompt_id")
    if sp is None:
        return None
    return LEGACY_SP_TO_DEFENSE.get(sp, sp)


def already_done_keys(path: Path) -> Set[tuple]:
    """Return (model, defense_id, dataset, behavior_id) tuples already in this file."""
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                defense_id = _defense_id_of(r)
                if defense_id is None:
                    continue
                keys.add((r["model"], defense_id, r["dataset"], r["behavior_id"]))
            except Exception:
                continue
    return keys


def run_one(model, tokenizer, system_prompt: str,
            messages: List[Dict[str, str]],
            max_new_tokens: int, temperature: float) -> Dict:
    t0 = time.perf_counter()
    response = generate_response(
        model, tokenizer, system_prompt, messages,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {"response": response, "latency_ms": elapsed_ms}


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_pending_tasks(
    model_name: str,
    defenses_to_run: List[str],
    loaded_datasets: Dict[str, list],
    done_by_defense: Dict[str, Set[tuple]],
) -> List[Tuple[str, str, dict]]:
    """Flatten (defense, dataset, item) tasks for one model, skipping completed ones."""
    tasks = []
    for defense_name in defenses_to_run:
        done = done_by_defense[defense_name]
        for ds_name, items in loaded_datasets.items():
            for item in items:
                key = (model_name, defense_name, ds_name, item["behavior_id"])
                if key not in done:
                    tasks.append((defense_name, ds_name, item))
    return tasks


class TeeLogger:
    """Print to stdout AND append to a log file. Plays nicely with tqdm."""
    def __init__(self, path: Path):
        self.path = path
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
    parser.add_argument("--output", required=True,
                        help="Parent directory. One subfolder per defense is "
                             "created/used inside.")
    parser.add_argument("--models", default="qwen,llama,gemma",
                        help="Comma-separated subset of models to run.")
    parser.add_argument("--defenses",
                        default=",".join(defenses_module.DEFENSES.keys()),
                        help="Comma-separated defenses to evaluate. "
                             f"Available: {list(defenses_module.DEFENSES)}")
    parser.add_argument("--datasets", default="advbench,harmbench,jbb",
                        help="Comma-separated subset of datasets to run.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap each dataset to N prompts (for smoke tests).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-quantize", action="store_true",
                        help="Load in FP16 instead of 4-bit.")
    parser.add_argument("--run-id", default=None,
                        help="Override the auto-generated run id (use to resume).")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide the tqdm bar (e.g., when redirecting to a file).")
    parser.add_argument("--auto-judge", default=None,
                        choices=[None, "harmbench_cls", "substring"],
                        help="If set, run Phase 2 (judging) automatically across "
                             "all defense subfolders after generation completes.")
    args = parser.parse_args()

    out_parent = Path(args.output)
    out_parent.mkdir(parents=True, exist_ok=True)
    log_path = out_parent / "run.log"
    logger = TeeLogger(log_path)

    run_id = args.run_id or uuid.uuid4().hex[:12]
    quantize = not args.no_quantize

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    defenses_to_run = [d.strip() for d in args.defenses.split(",") if d.strip()]
    datasets_to_run = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for m in models_to_run:
        if m not in MODELS:
            raise SystemExit(f"Unknown model '{m}'. Available: {list(MODELS)}")
    for d in defenses_to_run:
        if d not in defenses_module.DEFENSES:
            raise SystemExit(f"Unknown defense '{d}'. Available: {list(defenses_module.DEFENSES)}")
    for d in datasets_to_run:
        if d not in ds_module.LOADERS:
            raise SystemExit(f"Unknown dataset '{d}'. Available: {list(ds_module.LOADERS)}")

    # Create per-defense subfolders and load each one's already-done set.
    defense_dirs: Dict[str, Path] = {}
    done_by_defense: Dict[str, Set[tuple]] = {}
    for d in defenses_to_run:
        ddir = out_parent / d
        ddir.mkdir(parents=True, exist_ok=True)
        defense_dirs[d] = ddir
        done_by_defense[d] = already_done_keys(ddir / "generations.jsonl")

    logger.log(f"Run ID: {run_id}")
    logger.log(f"Output parent: {out_parent}")
    logger.log(f"Models: {models_to_run}")
    logger.log(f"Defenses (one subfolder each): {defenses_to_run}")
    logger.log(f"Datasets: {datasets_to_run}")
    logger.log(f"Quantize: {quantize}, max_new_tokens: {args.max_new_tokens}, "
               f"temperature: {args.temperature}")

    logger.log("Loading datasets...")
    loaded_datasets = {d: ds_module.load(d, limit=args.limit) for d in datasets_to_run}
    for name, items in loaded_datasets.items():
        logger.log(f"  {name}: {len(items)} prompts")

    already_total = sum(len(s) for s in done_by_defense.values())
    if already_total:
        logger.log(f"Resuming: {already_total} records already written across "
                   f"defense folders, will skip those.")
        for d, done in done_by_defense.items():
            if done:
                logger.log(f"  {d}: {len(done)} done")

    total_planned = sum(
        len(items) * len(defenses_to_run)
        for items in loaded_datasets.values()
    ) * len(models_to_run)
    total_pending = total_planned - already_total
    logger.log(f"Planned: {total_planned} generations. Pending: {total_pending}.")

    # Open one append-mode file handle per defense.
    out_files = {
        d: (defense_dirs[d] / "generations.jsonl").open("a", encoding="utf-8")
        for d in defenses_to_run
    }

    run_start = time.perf_counter()
    overall_done = 0

    try:
        for model_name in models_to_run:
            tasks = build_pending_tasks(model_name, defenses_to_run,
                                        loaded_datasets, done_by_defense)
            if not tasks:
                logger.log(f"[{model_name}] all done, skipping load.")
                continue

            logger.log(f"=== Loading {model_name} "
                       f"({'4-bit' if quantize else 'FP16'}) — "
                       f"{len(tasks)} tasks pending ===")
            load_t0 = time.perf_counter()
            model, tokenizer = load_model(model_name, quantize=quantize)
            logger.log(f"=== {model_name} loaded in "
                       f"{int(time.perf_counter() - load_t0)}s ===")

            bar = tqdm(
                total=len(tasks),
                desc=f"[{model_name}]",
                unit="gen",
                smoothing=0.1,
                disable=args.quiet,
                ncols=110,
            )
            ok = err = 0

            try:
                for defense_name, ds_name, item in tasks:
                    bar.set_postfix_str(f"defense={defense_name} ds={ds_name}",
                                        refresh=False)
                    key = (model_name, defense_name, ds_name, item["behavior_id"])
                    system_prompt, messages = defenses_module.apply(
                        defense_name, item["prompt"])
                    try:
                        result = run_one(
                            model, tokenizer, system_prompt, messages,
                            args.max_new_tokens, args.temperature,
                        )
                    except Exception as e:
                        err += 1
                        record = {
                            "ts": iso_now(), "run_id": run_id,
                            "model": model_name, "defense_id": defense_name,
                            "dataset": ds_name,
                            "behavior_id": item["behavior_id"],
                            "behavior_category": item["behavior_category"],
                            "prompt": item["prompt"],
                            "error": f"{type(e).__name__}: {e}",
                        }
                        tqdm.write(f"  ERROR [{model_name}/{defense_name}/{ds_name}] "
                                   f"{item['behavior_id']}: {e}")
                    else:
                        ok += 1
                        record = {
                            "ts": iso_now(), "run_id": run_id,
                            "model": model_name, "defense_id": defense_name,
                            "dataset": ds_name,
                            "behavior_id": item["behavior_id"],
                            "behavior_category": item["behavior_category"],
                            "prompt": item["prompt"],
                            "raw_behavior": item.get("raw_behavior", item["prompt"]),
                            "response": result["response"],
                            "latency_ms": result["latency_ms"],
                        }
                    out_files[defense_name].write(
                        json.dumps(record, ensure_ascii=False) + "\n")
                    out_files[defense_name].flush()
                    done_by_defense[defense_name].add(key)
                    overall_done += 1
                    bar.update(1)
            finally:
                bar.close()
                logger.log(f"[{model_name}] done: ok={ok}, err={err}")
                logger.log(f"=== Unloading {model_name} ===")
                unload_model(model, tokenizer)
                model, tokenizer = None, None
    finally:
        for f in out_files.values():
            f.close()
        elapsed = int(time.perf_counter() - run_start)
        logger.log(f"Run finished in {elapsed//60}m {elapsed%60}s. "
                   f"Wrote {overall_done} new records.")
        logger.log(f"Output: {out_parent}")
        logger.close()

    if args.auto_judge:
        import subprocess
        print(f"\n=== Auto-judging all defense folders with {args.auto_judge} ===")
        cmd = [sys.executable, "-m", "eval.judge",
               "--runs-dir", str(out_parent), "--judge", args.auto_judge]
        if args.quiet:
            cmd.append("--quiet")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
