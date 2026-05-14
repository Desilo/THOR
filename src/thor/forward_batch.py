import argparse
import contextlib
import json
import multiprocessing as mp
import os
import queue
from pathlib import Path
from types import SimpleNamespace

import torch

from .forward import run_forward


def parse_args():
    parser = argparse.ArgumentParser(description="Run forward across a range of target indices.")
    parser.add_argument("--dataset-type", default="mrpc", help="Dataset type used by the forward script.")
    parser.add_argument("--start-idx", type=int, default=0, help="First target index to run.")
    parser.add_argument("--end-idx", type=int, default=407, help="Inclusive upper bound for target indices.")
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=None,
        help="CUDA device indices to use. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--output-dir",
        default="./forward-batch-results",
        help="Output directory. Each target index gets its own subdirectory with plots and result.json.",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Use a memory-optimized execution mode with reduced memory footprint."
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip saving per-layer comparison plots.")
    return parser.parse_args()


def get_devices(requested_devices):
    if requested_devices is not None:
        return requested_devices
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices available.")
    return list(range(device_count))


def load_completed_indices(output_dir):
    completed = set()
    if not output_dir.exists():
        return completed

    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        result_path = child / "result.json"
        if not result_path.exists():
            continue
        try:
            record = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            continue
        target_idx = record.get("target_idx")
        if isinstance(target_idx, int) and "pred" in record and "label" in record and "plain_pred" in record:
            completed.add(target_idx)
    return completed


def get_target_output_dir(output_dir, target_idx):
    return output_dir / f"{target_idx}"


def worker(device, tasks, results, worker_args):
    while True:
        try:
            target_idx = tasks.get_nowait()
        except queue.Empty:
            return

        try:
            run_args = SimpleNamespace(
                dataset_type=worker_args.dataset_type,
                target_idx=target_idx,
                device=device,
                compact=worker_args.compact,
                no_plots=worker_args.no_plots,
                output_dir=str(get_target_output_dir(worker_args.output_dir, target_idx)),
                print_rotate_levels=False,
            )
            with (
                open(os.devnull, "w") as devnull,
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                result = run_forward(run_args)
                results.put(
                    dict(
                        target_idx=target_idx,
                        pred=result["pred"],
                        plain_pred=result["plain_pred"],
                        label=result["label"],
                        he_logits=result["he_logits"],
                        plain_logits=result["plain_logits"],
                    )
                )
        except Exception as exc:  # noqa: BLE001
            results.put(
                dict(
                    target_idx=target_idx,
                    device=device,
                    error=str(exc),
                )
            )


def main():
    args = parse_args()

    if args.start_idx < 0:
        raise ValueError("--start-idx must be non-negative")
    if args.end_idx < args.start_idx:
        raise ValueError("--end-idx must be greater than or equal to --start-idx")

    devices = get_devices(args.devices)
    output_dir = Path(args.output_dir or "forward-batch-results")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    completed = load_completed_indices(output_dir)
    completed_in_range = [idx for idx in completed if args.start_idx <= idx <= args.end_idx]
    target_indices = [idx for idx in range(args.start_idx, args.end_idx + 1) if idx not in completed]

    print(f"dataset_type={args.dataset_type}")
    print(f"requested_range=[{args.start_idx}, {args.end_idx}]")
    print(f"devices={devices}")
    print(f"output_dir={output_dir}")
    print(f"already_completed={len(completed_in_range)}")
    print(f"remaining_targets={len(target_indices)}")

    if not target_indices:
        return

    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    results = ctx.Queue()

    for target_idx in target_indices:
        tasks.put(target_idx)

    workers = []
    for device in devices:
        process = ctx.Process(target=worker, args=(device, tasks, results, args))
        process.start()
        workers.append(process)

    received = 0
    while received < len(target_indices):
        record = results.get()
        received += 1
        if "error" in record:
            print(
                f"[{received}/{len(target_indices)}] "
                f"target_idx={record['target_idx']} device={record['device']} error={record['error']}"
            )
            continue

        print(f"[{received}/{len(target_indices)}] target_idx={record['target_idx']} completed")

    for process in workers:
        process.join()

    print(f"final_result: completed={received} requested={len(target_indices)}")


if __name__ == "__main__":
    main()
