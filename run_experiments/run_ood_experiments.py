import argparse
import json
import os
import sys
import tempfile
import csv
import numpy as np
from collections import defaultdict
from pathlib import Path
from stable_baselines3 import PPO

from run_experiments.utils import RunSpec, evaluate_policy_on_assets, load_assets

#evaluate one model on all OOD assets
def evaluate_model_on_ood(model, ood_assets, run_dir, mug_scale, episodes_per_asset,
                          seed, progress_prefix=""):
    return evaluate_policy_on_assets(model, ood_assets, run_dir, mug_scale,
        None, episodes_per_asset, seed, result_type=None, success_key="ood_success_rate",
        scene_folder="ood_evaluation_scenes", progress_prefix=progress_prefix)


#find trained final models grouped by method
#Expected structure: <experiment_dir>/<method>/<run_dir>/final_model.zip
def get_method_runs(experiment_dir):
    runs_by_method = defaultdict(list)

    for model_path in sorted(experiment_dir.rglob("final_model.zip")):
        run_dir = model_path.parent
        method_dir = run_dir.parent
        if method_dir == experiment_dir:
            continue
        method = method_dir.name
        run_number = len(runs_by_method[method]) + 1
        if run_dir.name.startswith("seed"):
            try:
                seed = int(run_dir.name[4:])
            except ValueError:
                seed = run_number
        else:
            seed = run_number
        runs_by_method[method].append(
            RunSpec(method=method, seed=seed,
                run_dir=run_dir, model_path=model_path))

    #keep run order stable for each method
    for method in runs_by_method:
        runs_by_method[method].sort(key=lambda run: str(run.run_dir))

    return runs_by_method


#compute summary statistics for a list of values
def numeric_stats(values):
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)),
        "max": float(np.max(arr)), "median": float(np.median(arr))}


#sdummarize OOD results for one method across trained runs
def method_summary(method, method_results):
    success_rates = [float(r["ood_success_rate"]) for r in method_results]
    asset_successes = [float(r["asset_successes"]) for r in method_results]
    average_rewards = [
        float(r["average_reward"])
        for r in method_results
        if r.get("average_reward") is not None
    ]
    return {"method": method, "num_runs": len(method_results), "ood_success_rate": numeric_stats(success_rates), 
            "asset_successes": numeric_stats(asset_successes), "average_reward": numeric_stats(average_rewards)}


#flatten nested summary stats into one CSV row
def flatten_summary_row(summary):
    row = {"method": summary["method"], "num_runs": summary["num_runs"]}
    for metric in ("ood_success_rate", "asset_successes", "average_reward"):
        stats = summary[metric]
        for name in ("mean", "std", "min", "max", "median"):
            row[f"{metric}_{name}"] = stats[name]
    return row


#parse arguments, evaluate models, and write OOD result files
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment_dir", required=True) #dir containing final_model.zip runs
    parser.add_argument("--ood_dir", required=True) #directory containing OOD cup variants
    parser.add_argument("--output_dir", default="ood_evaluation_results")
    #number of episodes to try per OOD cup (attempts) -> if at least one episode succeeds, then we say 
    #it's a success
    parser.add_argument("--episodes_per_asset", type=int, default=10)
    parser.add_argument("--mug_scale", type=float, default=0.50)
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    ood_dir = Path(args.ood_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    ood_assets = load_assets(ood_dir)
    #get all experiment runs
    runs_by_method = get_method_runs(experiment_dir)

    print(f"Found runs for methods: {', '.join(runs_by_method.keys())}", flush=True)
    for method, runs in runs_by_method.items():
        print(f"{method}: {len(runs)} runs", flush=True)

    #eval each model
    results_by_method = {}
    all_results = []

    for method in sorted(runs_by_method.keys()):
        method_results = []

        for run_index, run in enumerate(runs_by_method[method], 1):
            print(f"Evaluating {method} run{run_index}", flush=True)

            try:
                model = PPO.load(str(run.model_path), device="cpu")
            except Exception as e:
                print(f"ERROR: Failed to load model: {e}", flush=True)
                continue

            #evaluate on OOD assets
            result = evaluate_model_on_ood(model, ood_assets,
                output_dir,
                args.mug_scale,
                args.episodes_per_asset,
                run.seed,
                progress_prefix=f"  [{method} run{run_index}]",
            )
            result["method"] = method
            result["run_number"] = run_index

            method_results.append(result)
            all_results.append(result)

            print(f"OOD success rate: {result['ood_success_rate']:.3f} | ({result['asset_successes']}/{result['num_assets']} cups)", flush=True)

        if method_results:
            results_by_method[method] = method_results

    #compute aggregate statistics + save
    summaries_by_method = {}
    summary_rows = []
    for method in sorted(results_by_method.keys()):
        method_results = results_by_method[method]
        summary = method_summary(method, method_results)
        summaries_by_method[method] = summary
        summary_rows.append(flatten_summary_row(summary))

        print(f"{method}:")
        success = summary["ood_success_rate"]
        solved = summary["asset_successes"]
        reward = summary["average_reward"]
        print(f"Success rate: mean={success['mean']:.3f}, std={success['std']:.3f} min={success['min']:.3f}, max={success['max']:.3f}")
        print(f"Cups solved: mean={solved['mean']:.1f}/{len(ood_assets)} std={solved['std']:.1f}, min={solved['min']:.0f}, max={solved['max']:.0f}")
        print(f"Avg reward: mean={reward['mean']:.3f}, std={reward['std']:.3f} min={reward['min']:.3f}, max={reward['max']:.3f}")

        #show each trained run separately
        for result in method_results:
            print(f"    run{result['run_number']}: {result['ood_success_rate']:.3f} "
                  f"({result['asset_successes']}/{result['num_assets']})")

    #save results
    results_path = output_dir / "ood_results.json"
    with open(results_path, "w") as f:
        json.dump({"experiment_dir": str(experiment_dir), "ood_dir": str(ood_dir),
            "num_ood_assets": len(ood_assets), "episodes_per_asset": args.episodes_per_asset,
            "results_by_method": results_by_method, "summaries_by_method": summaries_by_method}, f, indent=2)
    print(f"wrote to results path: {results_path}")

    #save summary CSV
    summary_path = output_dir / "ood_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "num_runs",
            "ood_success_rate_mean", "ood_success_rate_std",
            "ood_success_rate_min", "ood_success_rate_max",
            "ood_success_rate_median",
            "asset_successes_mean", "asset_successes_std",
            "asset_successes_min", "asset_successes_max",
            "asset_successes_median",
            "average_reward_mean", "average_reward_std",
            "average_reward_min", "average_reward_max",
            "average_reward_median"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"saved summary to path: {summary_path}")


if __name__ == "__main__":
    main()
