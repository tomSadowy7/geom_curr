import concurrent.futures
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
from pathlib import Path

#multi-threaded batch training script for running multiple training jobs in parallel
#training for difficulty labeling -> saves lot of statistics for downstream binning
DEBUG = False

#writes JSON to a file atomically using a temp file -> prevents corruption if race condition
def atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

#appends a JSON object as a line to a JSONL file
def append_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")

#find all asset directories containing a params.json file
def find_assets(asset_root):
    assets = sorted(p.parent for p in asset_root.rglob("params.json"))
    if asset_root.joinpath("params.json").exists():
        assets = [asset_root]
    return assets

#creates an initial run record with all fields needed to track a training job
def empty_run(asset, seed, timesteps, command, log_path, event_path):
    return {"asset": asset.name, "seed": seed,
        "timesteps": timesteps, "status": "running",
        "command": command, "log_path": str(log_path), "events_path": str(event_path),
        "mean_ep_reward_by_log": [], "mean_ep_reward_by_10k": {},
        "successes_by_10k": {}, "first_success_step": None, "first_success_fraction": None,
        "total_successes": 0, "max_success_lift_cm": 0.0, "final_mean_ep_reward": None,
        "best_mean_ep_reward": None, "mean_reward_auc": None}

#buckets a step number into 10k ranges
def bucket_10k(step):
    lo = (step // 10_000) * 10_000
    hi = lo + 10_000
    return f"{lo}-{hi}"

#calculates derived metrics for a run (final reward, best reward, AUC, success fraction)
#used downstream for difficulty function + difficulty binning
def update_derived(run):
    logs = run["mean_ep_reward_by_log"]
    if logs:
        rewards = [x["mean_ep_reward"] for x in logs]
        run["final_mean_ep_reward"] = rewards[-1]
        run["best_mean_ep_reward"] = max(rewards)
        if len(logs) == 1:
            run["mean_reward_auc"] = rewards[0]
        else:
            area = 0.0
            span = max(1, logs[-1]["step"] - logs[0]["step"])
            for prev, cur in zip(logs, logs[1:]):
                width = cur["step"] - prev["step"]
                area += width * 0.5 * (
                    prev["mean_ep_reward"] + cur["mean_ep_reward"]
                )
            run["mean_reward_auc"] = area / span

    first = run["first_success_step"]
    if first is not None:
        run["first_success_fraction"] = first / max(1, run["timesteps"])


#parses a line of training output and extracts mean reward and success metrics
def parse_line(line, run, event_path):
    if "mean ep reward" in line and "|" in line:
        try:
            parts = line.split("|")
            step_part = parts[0].split()[-1].replace(",", "")
            step = int(step_part)

            reward_part = parts[1].split("=")[1].split("(")[0].strip()
            reward = float(reward_part)

            episodes_part = parts[1].split("(")[1].split()[0]
            episodes = int(episodes_part)

            entry = {"step": step, "mean_ep_reward": reward, "episodes": episodes}
            run["mean_ep_reward_by_log"].append(entry)
            run["mean_ep_reward_by_10k"][bucket_10k(step)] = reward
            append_jsonl(event_path, {"type": "mean_ep_reward", **entry})
            update_derived(run)
            return True
        except (ValueError, IndexError):
            return False

    if "SUCCESS #" in line and "at step" in line:
        try:
            count_part = line.split("SUCCESS #")[1].split()[0]
            count = int(count_part)

            step_part = line.split("at step")[1].split(":")[0].strip().replace(",", "")
            step = int(step_part)

            lift_part = line.split("lifted")[1].split()[0]
            lift_cm = float(lift_part)

            run["total_successes"] = max(run["total_successes"], count)
            run["max_success_lift_cm"] = max(run["max_success_lift_cm"], lift_cm)
            if run["first_success_step"] is None:
                run["first_success_step"] = step
            key = bucket_10k(step)
            run["successes_by_10k"][key] = run["successes_by_10k"].get(key, 0) + 1
            append_jsonl(event_path, {"type": "success", "step": step, "success_count": count, "lift_cm": lift_cm})
            update_derived(run)
            return True
        except (ValueError, IndexError):
            return False

    return False


#aggregates individual runs by asset per seed and calculates summary statistics
def aggregate(results):
    out = {}
    for run in results["runs"]:
        asset = run["asset"]
        row = out.setdefault(asset, {"asset": asset,
            "runs": 0, "finished_runs": 0, "successful_seeds": 0, "first_success_steps": [],
            "total_successes": [], "final_mean_ep_rewards": [], "best_mean_ep_rewards": [], "mean_reward_aucs": []})
        row["runs"] += 1
        if run["status"] == "finished":
            row["finished_runs"] += 1
        if run["first_success_step"] is not None:
            row["successful_seeds"] += 1
            row["first_success_steps"].append(run["first_success_step"])
        row["total_successes"].append(run["total_successes"])
        for src, dest in (("final_mean_ep_reward", "final_mean_ep_rewards"),
            ("best_mean_ep_reward", "best_mean_ep_rewards"),
            ("mean_reward_auc", "mean_reward_aucs")):
            if run[src] is not None:
                row[dest].append(run[src])

    #aggregate statistics per row
    for row in out.values():
        firsts = row["first_success_steps"]
        totals = row["total_successes"]
        finals = row["final_mean_ep_rewards"]
        bests = row["best_mean_ep_rewards"]
        aucs = row["mean_reward_aucs"]
        
        row["success_seed_fraction"] = row["successful_seeds"] / max(1, row["runs"])
        row["median_first_success_step"] = float(sorted(firsts)[len(firsts) // 2]) if firsts else None
        row["mean_total_successes"] = sum(totals) / len(totals) if totals else 0.0
        row["mean_final_reward"] = sum(finals) / len(finals) if finals else None
        row["mean_best_reward"] = sum(bests) / len(bests) if bests else None
        row["mean_reward_auc"] = sum(aucs) / len(aucs) if aucs else None

    return dict(sorted(out.items()))


#saves results to two CSV files: one for individual runs and one for aggregated asset summaries
def save_csv(results, output_dir):
    run_csv = output_dir / "runs.csv"
    with open(run_csv, "w", newline="") as f:
        fields = ["asset", "seed", "status", "timesteps",
            "first_success_step", "first_success_fraction", "total_successes",
            "max_success_lift_cm", "final_mean_ep_reward",
            "best_mean_ep_reward", "mean_reward_auc", "log_path"]
        
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in results["runs"]:
            writer.writerow({k: run.get(k) for k in fields})

    asset_csv = output_dir / "assets.csv"
    with open(asset_csv, "w", newline="") as f:
        fields = ["asset", "runs", "finished_runs", "successful_seeds",
            "success_seed_fraction", "median_first_success_step",
            "mean_total_successes", "mean_final_reward", "mean_best_reward",
            "mean_reward_auc"]
        
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results["assets"].values():
            writer.writerow({k: row.get(k) for k in fields})


#saves results to JSON and CSV files after aggregating by asset
def save_results(results_path, results):
    results["assets"] = aggregate(results)
    atomic_write_json(results_path, results)
    save_csv(results, results_path.parent)


#thread safe wrapper for save_results using a lock
def save_results_locked(results_path, results, save_lock):
    if save_lock is None:
        save_results(results_path, results)
        return
    with save_lock:
        save_results(results_path, results)


#runs a single training job, streams output, parses metrics, and saves results
def run_one(args, asset, seed, results, results_path, save_lock=None):
    run_id = f"{asset.name}_seed{seed}"
    run_dir = Path(args.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    event_path = run_dir / "events.jsonl"
    work_dir = run_dir / "scene"
    asset_abs = Path(asset).resolve()
    #ensure you have proper conda env activated
    command = [sys.executable, "-u",  "train_mujoco.py", "--mug", str(asset_abs),
        "--timesteps", str(args.timesteps), "--log_freq", str(args.log_freq),
        "--rollout_freq", str(args.rollout_freq), "--mug_scale", str(args.mug_scale),
        "--num_envs", str(args.num_envs_per_run), "--work_dir", str(work_dir)]

    run = empty_run(asset, seed, args.timesteps, command, log_path, event_path)
    if save_lock is None:
        results["runs"].append(run)
    else:
        with save_lock:
            results["runs"].append(run)
    save_results_locked(results_path, results, save_lock)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1" #for output to actually go through

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(command, cwd=Path(__file__).parent,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env)

        lines_since_save = 0
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="") #print result
            parsed = parse_line(line, run, event_path)
            lines_since_save += 1
            if parsed or lines_since_save >= 50:
                save_results_locked(results_path, results, save_lock)
                lines_since_save = 0

        exit_code = proc.wait()

    run["status"] = "finished" if exit_code == 0 else "failed"
    update_derived(run)
    save_results_locked(results_path, results, save_lock)
    return run


#set up training tasks and runs them in parallel
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_dir") #dir containing CupFactory_* asset folders
    parser.add_argument("--output_dir", default="../train_all_results")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--log_freq", type=int, default=10_000)
    parser.add_argument("--rollout_freq", type=int, default=0)
    parser.add_argument("--mug_scale", type=float, default=0.50)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--num_envs_per_run", type=int, default=16)
    args = parser.parse_args()

    asset_root = Path(args.asset_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir = output_dir.resolve()
    results_path = output_dir / "results.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = find_assets(asset_root)
    seeds = random.sample(range(1_000_000), args.num_seeds)

    results = {"asset_dir": str(asset_root), "output_dir": str(output_dir),
        "timesteps": args.timesteps, "seeds": seeds, "runs": [], "assets": {}}
    save_results(results_path, results)
    print(f"results path: {results_path}")

    if DEBUG:
        print(f"assets: {len(assets)}")
        print(f"seeds: {seeds}")
        print(f"jobs: {args.jobs}")
        print(f"envs/run: {args.num_envs_per_run}")

    tasks = []
    for asset in assets:
        for seed in seeds:
            tasks.append((asset, seed))

    #create thread pool if jobs > 1
    if args.jobs == 1:
        for asset, seed in tasks:
            save_results(results_path, results)
            run_one(args, asset, seed, results, results_path)
    else:
        save_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            future_to_task = {}
            for asset, seed in tasks:
                with save_lock:
                    save_results(results_path, results)
                future = pool.submit(
                    run_one, args, asset, seed, results, results_path, save_lock)
                future_to_task[future] = (asset, seed)

            for future in concurrent.futures.as_completed(future_to_task):
                asset, seed = future_to_task[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"ERROR {asset.name} seed={seed}: {exc}")
                    with save_lock:
                        results.setdefault("errors", []).append({
                            "asset": asset.name,
                            "seed": seed,
                            "error": repr(exc),
                        })
                        save_results(results_path, results)

    save_results(results_path, results)

if __name__ == "__main__":
    main()
