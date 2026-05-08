import argparse
import concurrent.futures
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import (DIFFICULTIES, AssetSpec, CheckpointEvalCallback,
    JsonTrainLogger, MultiAssetEnvBase, append_jsonl, atomic_write_json,
    load_assets, random_seeds, save_runs_csv)

#curriculum is ours and other three are basleines
METHODS = ("curriculum", "fixed", "uniform", "hard_only")

#set up env that samples assets according to one training method
class MultiAssetMugLiftEnv(MultiAssetEnvBase):
    def __init__(self, assets, method, total_timesteps, run_dir, mug_scale, seed=0,
                 real_time=False, max_cached_asset_envs=1):
        self.method = method
        super().__init__(assets=assets, total_timesteps=total_timesteps,
            run_dir=run_dir, mug_scale=mug_scale, seed=seed, real_time=real_time,
            max_cached_asset_envs=max_cached_asset_envs)

    #pick an asset based on the selected curriculum or baseline method.
    def choose_asset(self):
        progress = min(1.0, self.training_step / self.total_timesteps)
        if self.method == "uniform" or self.method == "fixed":
            choices = self.assets
        elif self.method == "hard_only":
            choices = [asset for asset in self.assets if asset.difficulty == "hard"]
        elif self.method == "curriculum":
            if progress < 1.0 / 3.0:
                allowed = {"easy"}
            elif progress < 2.0 / 3.0:
                allowed = {"easy", "medium"}
            else:
                allowed = {"easy", "medium", "hard"}
            choices = [asset for asset in self.assets if asset.difficulty in allowed]
        #coudl raise exception here if method unknown
        return choices[int(self.rng.integers(0, len(choices)))]


#build a subprocess vector environment for PPO training
#dummy env is useless for speedups bc it's sequential
def make_vec_env(args, method, assets, run_dir, seed):
    def make_env(rank):
        #build env
        def init_env():
            env = MultiAssetMugLiftEnv(assets=assets, method=method,
                total_timesteps=args.timesteps, run_dir=str(run_dir / "train_envs" / f"env_{rank}"),
                mug_scale=args.mug_scale, seed=seed + rank, real_time=False,
                max_cached_asset_envs=args.max_cached_asset_envs)
            return Monitor(env)
        return init_env

    env_fns = [make_env(rank) for rank in range(args.num_envs_per_run)]
    return SubprocVecEnv(env_fns)


#train one curriculum or baseline run for a single random seed
def run_one(serialized):
    args = argparse.Namespace(**serialized["args"])
    method = serialized["method"]
    seed = serialized["seed"]
    train_assets = [AssetSpec(**item) for item in serialized["train_assets"]]
    val_assets = [AssetSpec(**item) for item in serialized["val_assets"]]
    run_dir = Path(serialized["run_dir"]) #ensure dir exists
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {"method": method, "timesteps": args.timesteps,
        "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes,
        "mug_scale": args.mug_scale, "train_assets": [vars(asset) for asset in train_assets]}
    atomic_write_json(run_dir / "run_config.json", run_config)

    events_path = run_dir / "events.jsonl"
    append_jsonl(events_path, {"type": "run_start", "method": method,
        "train_asset_count": len(train_assets), "validation_asset_count": len(val_assets)})

    vec_env = make_vec_env(args, method, train_assets, run_dir, seed)
    
    #same hyperparams as train_mujoco.py
    model = PPO("MlpPolicy", vec_env, seed=seed,
        verbose=0, device="cpu", n_steps=2048, batch_size=64,
        n_epochs=10, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)

    callbacks = [JsonTrainLogger(events_path, log_freq=args.log_freq),
        CheckpointEvalCallback(run_dir=run_dir, val_assets=val_assets,
        mug_scale=args.mug_scale, eval_interval=args.eval_interval, eval_episodes=args.eval_episodes,
        seed=seed)]

    #just in case PPO backend fails
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    finally:
        vec_env.close()

    final_model = run_dir / "final_model.zip"
    model.save(final_model)
    run_summary = {"method": method, "status": "finished",
        "run_dir": str(run_dir), "events_path": str(events_path),
        "eval_path": str(run_dir / "eval.jsonl"), "final_model_path": str(final_model)}
    append_jsonl(events_path, {"type": "run_end", **run_summary})
    atomic_write_json(run_dir / "run_summary.json", run_summary)
    return run_summary


#select the training assets used by one baseline or curriculum method
def method_assets(method, train_assets, fixed_asset):
    if method == "fixed":
        return [fixed_asset]
    if method == "hard_only":
        return [asset for asset in train_assets if asset.difficulty == "hard"]
    return train_assets


#add a run result to the output log and CSV summary
def record_result(result, runs, output_log, output_log_path, output_dir):
    runs.append(result)
    output_log["runs"] = runs
    atomic_write_json(output_log_path, output_log)
    save_runs_csv(output_dir, runs)


#parse arguments, create processes + threads, and run the curriculum experiment
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--fixed_asset", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--output_dir", default="curriculum_experiment_runs")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--eval_interval", type=int, default=200_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=10_000)
    parser.add_argument("--mug_scale", type=float, default=0.50)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--num_envs_per_run", type=int, default=4)
    #maximum full MuJoCo asset envs cached inside each PPO env - was running intro crazy 
    #memory growth w/o this crashing my VM
    parser.add_argument("--max_cached_asset_envs", type=int, default=1)
    args = parser.parse_args()
    

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    fixed_path = Path(args.fixed_asset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_assets = load_assets(train_dir, require_difficulty=True)
    val_assets = load_assets(val_dir, require_difficulty=False)
    fixed_candidates = load_assets(fixed_path, require_difficulty=False)
   
    fixed_asset = fixed_candidates[0]
    
    counts = Counter(asset.difficulty for asset in train_assets)
    output_log_path = output_dir / "output_log.json"
    seeds = random_seeds(args.num_seeds)
    runs = []

    #saved output log
    output_log = {"train_dir": str(train_dir), "val_dir": str(val_dir),
        "fixed_asset": vars(fixed_asset), "methods": args.methods, "timesteps": args.timesteps,
        "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes,
        "validation_asset_count": len(val_assets), "output_dir": str(output_dir), "runs": runs}
    atomic_write_json(output_log_path, output_log)

    tasks = []
    for method in args.methods:
        assets = method_assets(method, train_assets, fixed_asset)

        
        for index, seed in enumerate(seeds, 1):
            run_dir = output_dir / method / f"run{index}"
            tasks.append({"method": method, "seed": seed, "run_dir": str(run_dir),
                "train_assets": [vars(asset) for asset in assets],
                "val_assets": [vars(asset) for asset in val_assets], "args": vars(args)})

    if args.jobs == 1:
        for task in tasks:
            print(f"running {task['method']} seed={task['seed']}")
            record_result(run_one(task), runs, output_log, output_log_path, output_dir)
    else: #else start multiprocessing
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_task = {pool.submit(run_one, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"method": task["method"], "status": "failed",
                        "error": repr(exc), "run_dir": task["run_dir"]}
                    print(f"ERROR | {task['method']} seed={task['seed']}: {exc}")
                record_result(result, runs, output_log, output_log_path, output_dir)

    print(f"done wrote to: {output_log_path}")


if __name__ == "__main__":
    main()
