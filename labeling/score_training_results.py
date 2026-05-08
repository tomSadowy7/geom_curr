import argparse
import csv
import json
import math
from pathlib import Path

#SCORING PROCESS
#asset_score = 0.45 * success_score + 0.35 * late_reward_score + 0.20 * reward_auc_score

#write JSON to file atomically using a temp file to avoid corruption if race condition exists among threads
def atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

#normalize scores to [0, 1] range
def minmax_scores(raw_by_asset):
    values = [v for v in raw_by_asset.values() if v is not None]
    if not values:
        return {asset: 0.0 for asset in raw_by_asset}

    lo = min(values)
    hi = max(values)
    if hi == lo:
        #if for some reason all tied, return 0.5 or else we div by zero err
        return {asset: 0.5 if value is not None else 0.0 for asset, value in raw_by_asset.items()}

    return {asset: ((value - lo) / (hi - lo)) if value is not None else 0.0 for asset, value in raw_by_asset.items()}


#select the best event attempt based on completion and success metrics
def choose_event_attempt(run, attempts):
    if not attempts:
        return []

    timesteps = max(1, int(run.get("timesteps") or 1))
    target_successes = int(run.get("total_successes") or 0)

    def attempt_score(events):
        steps = [event["step"] for event in events if "step" in event]
        step = max(steps) if steps else 0
        counts = [event["success_count"] for event in events if event.get("type") == "success" and "success_count" in event]
        successes = max(counts) if counts else 0
        completed = int(step >= 0.95 * timesteps)
        success_match = int(successes == target_successes)
        closeness = -abs(step - min(step, timesteps))
        return completed, success_match, closeness, step

    return max(attempts, key=attempt_score)


#select one run per asset/seed pair
def choose_runs(results):
    chosen = {}
    for run in results.get("runs", []):
        key = (run.get("asset"), int(run.get("seed", 0)))
        previous = chosen.get(key)
        if previous is None:
            chosen[key] = run
            continue

        run_finished = run.get("status") == "finished"
        previous_finished = previous.get("status") == "finished"
        if run_finished and not previous_finished:
            chosen[key] = run
        elif run_finished == previous_finished:
            chosen[key] = run

    return list(chosen.values())

#compute time-normalized AUC reward
#approx using trapezoids
def reward_auc(reward_events):
    points = sorted((int(event["step"]), float(event["mean_ep_reward"]))
        for event in reward_events if event.get("type") == "mean_ep_reward" and event.get("step") is not None and event.get("mean_ep_reward") is not None)
    
    #bad file or wrong path
    if not points:
        return None

    if len(points) == 1:
        return points[0][1]

    area = 0.0
    span = max(1, points[-1][0] - points[0][0])
    for (prev_step, prev_reward), (step, reward) in zip(points, points[1:]):
        #normalize + approx via trapezoid
        area += (step - prev_step) * 0.5 * (prev_reward + reward)
    return area / span

#avg reward in the final third of training (step >= 2/3 * timesteps)
def late_reward(reward_events, timesteps):
    cutoff = (2.0 / 3.0) * max(1, timesteps)
    rewards = [float(event["mean_ep_reward"]) for event in reward_events if event.get("type") == "mean_ep_reward"  and event.get("step", 0) >= cutoff and event.get("mean_ep_reward") is not None]
    return sum(rewards) / len(rewards) if rewards else None

#compute weighted final scores and build output structure
def finalize_scores(grouped, per_run, source, success_weight, late_reward_weight, reward_auc_weight):
    weight_sum = success_weight + late_reward_weight + reward_auc_weight


    raw_success = {}
    raw_late_reward = {}
    raw_auc = {}
    for asset, row in grouped.items():
        successes = row["successes"]
        mean_successes = sum(successes) / len(successes) if successes else None
        raw_success[asset] = math.log1p(mean_successes) if mean_successes is not None else None
        late_rewards = row["late_rewards"]
        raw_late_reward[asset] = sum(late_rewards) / len(late_rewards) if late_rewards else None
        reward_aucs = row["reward_aucs"]
        raw_auc[asset] = sum(reward_aucs) / len(reward_aucs) if reward_aucs else None

    success_scores = minmax_scores(raw_success)
    late_scores = minmax_scores(raw_late_reward)
    auc_scores = minmax_scores(raw_auc)

    assets = {}
    for asset, row in sorted(grouped.items()):
        score = (success_weight * success_scores[asset] + late_reward_weight * late_scores[asset] + reward_auc_weight * auc_scores[asset]) / weight_sum
        successes = row["successes"]
        mean_successes = sum(successes) / len(successes) if successes else None
        assets[asset] = {"asset": asset, "seeds": row["seeds"],
            "seed_count": len(row["seeds"]), "mean_total_successes": mean_successes,
            "raw_success_log1p_mean": raw_success[asset], "raw_late_reward": raw_late_reward[asset], "raw_reward_auc": raw_auc[asset], "success_score": success_scores[asset],
            "late_reward_score": late_scores[asset], "reward_auc_score": auc_scores[asset], "asset_score": score}

    return {"source": str(source),
        "weights": {"success_weight": success_weight,"late_reward_weight": late_reward_weight ,"reward_auc_weight": reward_auc_weight}, "assets": assets, "runs": per_run}

#score assets from results.json by computing metrics from event logs
def score_results(results, results_path, success_weight = 0.45, late_reward_weight = 0.35, reward_auc_weight = 0.20):
    grouped = {}
    per_run = []

    for run in choose_runs(results):
        asset = run.get("asset")
        if not asset:
            continue
        
        #how its saved in json
        path = Path(run.get("events_path", ""))
        if not (path.is_absolute() or path.exists()):
            for candidate in [results_path.parent / path, results_path.parent.parent / path]:
                if candidate.exists():
                    path = candidate
                    break
        events_path = path

        if events_path.exists():
            all_events = []
            for line in events_path.read_text().splitlines():
                if line.strip():
                    try:
                        all_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        else:
            all_events = []

        attempts = []
        current = []
        last_step = None
        for event in all_events:
            step = event["step"]
            if last_step is not None and step < last_step:
                if current:
                    attempts.append(current)
                current = []
            current.append(event)
            last_step = step
        if current:
            attempts.append(current)

        events = choose_event_attempt(run, attempts)
        reward_events = [event for event in events if event.get("type") == "mean_ep_reward"]
        timesteps = max(1, int(run.get("timesteps") or results.get("timesteps") or 1))

        success_counts = [int(event["success_count"]) for event in events if event.get("type") == "success" and event.get("success_count") is not None]
        run_successes = max(success_counts) if success_counts else 0

        run_late_reward = late_reward(reward_events, timesteps)
        run_auc = reward_auc(reward_events)

        grouped.setdefault(asset, {"asset": asset, "seeds": [], 
            "successes": [], "late_rewards": [], "reward_aucs": []})
        grouped[asset]["seeds"].append(run.get("seed"))
        grouped[asset]["successes"].append(float(run_successes))
        if run_late_reward is not None:
            grouped[asset]["late_rewards"].append(run_late_reward)
        if run_auc is not None:
            grouped[asset]["reward_aucs"].append(run_auc)

        max_step = max([event["step"] for event in events]) if events else 0
        per_run.append({"asset": asset, "seed": run.get("seed"),
            "status": run.get("status"), "events_path": str(events_path),
            "event_attempts_found": len(attempts), "selected_event_count": len(events), "selected_max_step": max_step, "successes": run_successes, "late_reward": run_late_reward, "reward_auc": run_auc})

    return finalize_scores(grouped, per_run, results_path, success_weight, late_reward_weight, reward_auc_weight)

#write scored results to CSV with all metrics per asset
def save_csv(scored, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["asset", "seed_count", "mean_total_successes",
        "raw_success_log1p_mean", "raw_late_reward", "raw_reward_auc",
        "success_score", "late_reward_score", "reward_auc_score",
        "asset_score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in scored["assets"].values():
            writer.writerow({field: row.get(field) for field in fields})

#parse input, score results, and write JSON/CSV output files
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input") #path to results.json from train_all.py output
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_csv", default="out_csv") #SAVE THIS FOR label_assets_from_scores.py
    #WEIGHTS FOR REWARD FUNCTION
    parser.add_argument("--success_weight", type=float, default=0.45)
    parser.add_argument("--late_reward_weight", type=float, default=0.35)
    parser.add_argument("--reward_auc_weight", type=float, default=0.20)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        results_path = input_path / "results.json"
    else:
        results_path = input_path

    results = json.loads(results_path.read_text())
    scored = score_results(results, results_path, success_weight=args.success_weight,
        late_reward_weight=args.late_reward_weight, reward_auc_weight=args.reward_auc_weight)
    output_dir = results_path.parent

    out_json = Path(args.out_json) if args.out_json else output_dir / "asset_reward_scores.json"
    out_csv = Path(args.out_csv) if args.out_csv else output_dir / "asset_reward_scores.csv"
    
    atomic_write_json(out_json, scored)
    save_csv(scored, out_csv)

    print(f"num of assets: {len(scored['assets'])}")
    print(f"wrote json to path: {out_json}")
    print(f"wrote csv to path: {out_csv}")


if __name__ == "__main__":
    main()
