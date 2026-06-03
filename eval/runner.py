"""
Phase 1 — generation.

For each (model, system_prompt, dataset, behavior), generate a response and
append a JSONL record. Only ONE target model is in VRAM at a time; we unload
and free CUDA memory between models.

Resumable: re-running the same --output skips records already written
(matched by run_id + model + system_prompt + dataset + behavior_id).

Live monitoring:
  - tqdm bar in the terminal shows per-model progress + ETA.
  - `tail -f <output>/generations.jsonl` for record-by-record live log.
  - `tail -f <output>/run.log` for the same lines that go to the console.

Example:
    python -m eval.runner --output eval/runs/2026-05-28
    python -m eval.runner --models qwen --datasets advbench --limit 10
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
from eval import system_prompts as sp_module


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def already_done_keys(path: Path) -> Set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r["model"], r["system_prompt_id"], r["dataset"], r["behavior_id"]))
            except Exception:
                continue
    return keys


def run_one(model, tokenizer, system_prompt: str, prompt: str,
            max_new_tokens: int, temperature: float) -> Dict:
    messages = [{"role": "user", "content": prompt}]
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
    sps_to_run: List[str],
    loaded_datasets: Dict[str, list],
    done: Set[tuple],
) -> List[Tuple[str, str, dict]]:
    """Flatten (sp, dataset, item) tasks for one model, skipping completed ones."""
    tasks = []
    for sp_name in sps_to_run:
        for ds_name, items in loaded_datasets.items():
            for item in items:
                key = (model_name, sp_name, ds_name, item["behavior_id"])
                if key not in done:
                    tasks.append((sp_name, ds_name, item))
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
                        help="Output directory. generations.jsonl is appended inside.")
    parser.add_argument("--models", default="qwen,llama,gemma",
                        help="Comma-separated subset of models to run.")
    parser.add_argument("--system-prompts", default="permissive,securebank_defensive",
                        help="Comma-separated subset of system prompts to run.")
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
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.jsonl"
    log_path = out_dir / "run.log"
    logger = TeeLogger(log_path)

    run_id = args.run_id or uuid.uuid4().hex[:12]
    quantize = not args.no_quantize

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    sps_to_run    = [s.strip() for s in args.system_prompts.split(",") if s.strip()]
    datasets_to_run = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for m in models_to_run:
        if m not in MODELS:
            raise SystemExit(f"Unknown model '{m}'. Available: {list(MODELS)}")
    for s in sps_to_run:
        sp_module.get(s)
    for d in datasets_to_run:
        if d not in ds_module.LOADERS:
            raise SystemExit(f"Unknown dataset '{d}'. Available: {list(ds_module.LOADERS)}")

    logger.log(f"Run ID: {run_id}")
    logger.log(f"Output: {out_path}")
    logger.log(f"Models: {models_to_run}")
    logger.log(f"System prompts: {sps_to_run}")
    logger.log(f"Datasets: {datasets_to_run}")
    logger.log(f"Quantize: {quantize}, max_new_tokens: {args.max_new_tokens}, "
               f"temperature: {args.temperature}")

    logger.log("Loading datasets...")
    loaded_datasets = {d: ds_module.load(d, limit=args.limit) for d in datasets_to_run}
    for name, items in loaded_datasets.items():
        logger.log(f"  {name}: {len(items)} prompts")

    done = already_done_keys(out_path)
    if done:
        logger.log(f"Resuming: {len(done)} records already in {out_path.name}, skipping those.")

    total_planned = sum(
        len(items) * len(sps_to_run)
        for items in loaded_datasets.values()
    ) * len(models_to_run)
    total_pending = total_planned - len(done)
    logger.log(f"Planned: {total_planned} generations. Pending: {total_pending}.")

    run_start = time.perf_counter()
    overall_done = 0

    try:
        with out_path.open("a", encoding="utf-8") as out_f:
            for model_name in models_to_run:
                tasks = build_pending_tasks(model_name, sps_to_run,
                                            loaded_datasets, done)
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
                    for sp_name, ds_name, item in tasks:
                        bar.set_postfix_str(f"sp={sp_name} ds={ds_name}",
                                            refresh=False)
                        key = (model_name, sp_name, ds_name, item["behavior_id"])
                        try:
                            result = run_one(
                                model, tokenizer,
                                sp_module.get(sp_name), item["prompt"],
                                args.max_new_tokens, args.temperature,
                            )
                        except Exception as e:
                            err += 1
                            record = {
                                "ts": iso_now(), "run_id": run_id,
                                "model": model_name, "system_prompt_id": sp_name,
                                "dataset": ds_name,
                                "behavior_id": item["behavior_id"],
                                "behavior_category": item["behavior_category"],
                                "prompt": item["prompt"],
                                "error": f"{type(e).__name__}: {e}",
                            }
                            tqdm.write(f"  ERROR [{model_name}/{sp_name}/{ds_name}] "
                                       f"{item['behavior_id']}: {e}")
                        else:
                            ok += 1
                            record = {
                                "ts": iso_now(), "run_id": run_id,
                                "model": model_name, "system_prompt_id": sp_name,
                                "dataset": ds_name,
                                "behavior_id": item["behavior_id"],
                                "behavior_category": item["behavior_category"],
                                "prompt": item["prompt"],
                                "raw_behavior": item.get("raw_behavior", item["prompt"]),
                                "response": result["response"],
                                "latency_ms": result["latency_ms"],
                            }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        done.add(key)
                        overall_done += 1
                        bar.update(1)
                finally:
                    bar.close()
                    logger.log(f"[{model_name}] done: ok={ok}, err={err}")
                    logger.log(f"=== Unloading {model_name} ===")
                    unload_model(model, tokenizer)
                    model, tokenizer = None, None
    finally:
        elapsed = int(time.perf_counter() - run_start)
        logger.log(f"Run finished in {elapsed//60}m {elapsed%60}s. "
                   f"Wrote {overall_done} new records.")
        logger.log(f"Output: {out_path}")
        logger.close()


if __name__ == "__main__":
    main()
