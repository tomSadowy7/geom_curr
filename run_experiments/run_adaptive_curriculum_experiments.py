import argparse
import concurrent.futures
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from run_experiments.utils import (DIFFICULTIES, AssetSpec, CheckpointEvalCallback,
    JsonTrainLogger, MultiAssetEnvBase, append_jsonl, atomic_write_json,
    load_assets, random_seeds, save_runs_csv)

#sets up an environment that starts with only easy assets unlocked
class AdaptiveMultiAssetMugLiftEnv(MultiAssetEnvBase):
    def __init__(self, assets, total_timesteps, run_dir, mug_scale, seed=0,
                 real_time=False, max_cached_asset_envs=1):
        
        super().__init__(assets=assets, total_timesteps=total_timesteps,
            run_dir=run_dir, mug_scale=mug_scale, seed=seed,
            real_time=real_time, max_cached_asset_envs=max_cached_asset_envs)
        
        self.unlocked_difficulties = {"easy"}

    #update which difficulty levels are sampled
    def set_unlocked_difficulties(self, difficulties):
        self.unlocked_difficulties = difficulties

    #pick random asset from the currently unlocked difficulties
    def choose_asset(self):
        choices = [asset for asset in self.assets if asset.difficulty in self.unlocked_difficulties]
        return choices[int(self.rng.integers(0, len(choices)))]

#track reward plateaus and unlock harder assets over time
class AdaptiveCurriculumCallback(BaseCallback):
    def __init__(self, events_path, grace_period=300_000,
                 plateau_window=100_000, stagnation_lower=-0.01,
                 stagnation_upper=0.01):
        super().__init__()
        self.events_path = events_path
        self.grace_period = grace_period
        self.plateau_window = plateau_window
        self.stagnation_lower = stagnation_lower #-1% for experiments
        self.stagnation_upper = stagnation_upper #+1% for experiments
        self.step_to_reward = []
        self.unlocked_difficulties = {"easy"}
        self.steps_since_unlock = 0
        self.difficulty_order = ["easy", "medium", "hard"]
        self.last_trim_step = 0

    #collect rewards and check whether the next difficulty should unlock
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                reward = float(info["episode"]["r"])
                self.step_to_reward.append((self.num_timesteps, reward))

        #trim old data to prevent unbounded memory frowth
        if self.num_timesteps - self.last_trim_step >= 100_000:
            cutoff_step = self.num_timesteps - (self.plateau_window * 3)
            self.step_to_reward = [(s, r) for s, r in self.step_to_reward if s >= cutoff_step]
            self.last_trim_step = self.num_timesteps

        self.steps_since_unlock += self.model.env.num_envs

        if (self.steps_since_unlock >= self.grace_period and len(self.unlocked_difficulties) < 3):
            self.check_and_unlock_difficulty()

        return True

    #cmpare recent reward windows and unlock the next difficulty on plateau
    def check_and_unlock_difficulty(self):
        recent_start = self.num_timesteps - self.plateau_window
        recent_rewards = [r for s, r in self.step_to_reward if s >= recent_start]

        if len(recent_rewards) < 5:
            return

        prev_start = recent_start - self.plateau_window
        prev_rewards = [r for s, r in self.step_to_reward if prev_start <= s < recent_start]

        if len(prev_rewards) < 5:
            return

        prev_avg = float(np.mean(prev_rewards))
        recent_avg = float(np.mean(recent_rewards))

        if prev_avg > 0:
            improvement = (recent_avg - prev_avg) / prev_avg
        else:
            improvement = 0.0

        print(f"Step {self.num_timesteps:,} | Checking plateau (prev_avg={prev_avg:.3f}, recent_avg={recent_avg:.3f} | improvement={improvement:.2%})")

        #only unlock when performance is stagnating (between -1% and +1% improvement)
        is_stagnating = self.stagnation_lower < improvement < self.stagnation_upper

        if is_stagnating:
            next_difficulty = None
            for d in self.difficulty_order:
                if d not in self.unlocked_difficulties:
                    next_difficulty = d
                    break

            if next_difficulty:
                self.unlocked_difficulties.add(next_difficulty)
                self.steps_since_unlock = 0
                append_jsonl(self.events_path, {"type": "curriculum_unlock",
                    "step": self.num_timesteps, "unlocked_difficulty": next_difficulty,
                    "unlocked_difficulties": sorted(self.unlocked_difficulties),
                    "prev_avg_reward": prev_avg, "recent_avg_reward": recent_avg,
                    "improvement": improvement})
                print(f"UNLOCKING {next_difficulty.upper()} at step {self.num_timesteps:,} |  Now unlocked: {', '.join(sorted(self.unlocked_difficulties))}")
            else:
                print(f"Step {self.num_timesteps:,}: All difficulties already unlocked")
        elif improvement >= self.stagnation_upper:
            print(f"Step {self.num_timesteps:,}: Still improving ({improvement:.2%}) -> not unlocking yet")
        elif improvement <= self.stagnation_lower:
            print(f"Step {self.num_timesteps:,}: Performance declining ({improvement:.2%}) keeping current difficulty")
        #Detailed prints to see what's going on
        
        self.training_env.env_method("set_unlocked_difficulties", self.unlocked_difficulties)


#build subprocess vector environment for PPO training (allows for faster training)
def make_vec_env(args, assets, run_dir, seed):
    #create one monitored environment factory for subprocess worker
    def make_env(rank):
        #build the actual env inside the subprocess
        def init_env():
            env = AdaptiveMultiAssetMugLiftEnv(assets=assets,
                total_timesteps=args.timesteps, run_dir=str(run_dir / "train_envs" / f"env_{rank}"),
                mug_scale=args.mug_scale, seed=seed + rank, real_time=False,
                max_cached_asset_envs=args.max_cached_asset_envs)
            return Monitor(env)
        return init_env

    env_fns = [make_env(rank) for rank in range(args.num_envs_per_run)]
    return SubprocVecEnv(env_fns)


#train one adaptive curriculum run for a single random seed
def run_one(serialized):
    args = argparse.Namespace(**serialized["args"])
    seed = serialized["seed"]
    train_assets = [AssetSpec(**item) for item in serialized["train_assets"]]
    val_assets = [AssetSpec(**item) for item in serialized["val_assets"]]
    run_dir = Path(serialized["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {"method": "adaptive_curriculum",
        "timesteps": args.timesteps, "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "mug_scale": args.mug_scale, "train_assets": [vars(asset) for asset in train_assets],
    }
    atomic_write_json(run_dir / "run_config.json", run_config)

    events_path = run_dir / "events.jsonl"
    append_jsonl(events_path, {"type": "run_start", "method": "adaptive_curriculum",
        "train_asset_count": len(train_assets), "validation_asset_count": len(val_assets)})

    vec_env = make_vec_env(args, train_assets, run_dir, seed)
    
    #same hyperparams as train_mujoco.py
    model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0, device="cpu", n_steps=2048,
        batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)

    callbacks = [JsonTrainLogger(events_path, log_freq=args.log_freq),       
                 AdaptiveCurriculumCallback(events_path=events_path,
                grace_period=300_000, plateau_window=100_000,
                stagnation_lower=-0.01, stagnation_upper=0.01),
                CheckpointEvalCallback(run_dir=run_dir, val_assets=val_assets,
                mug_scale=args.mug_scale, eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes, seed=seed)]

    #just in case PPO fails, for some reason was failing in VM when first writing the code
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    finally:
        vec_env.close()

    final_model = run_dir / "final_model.zip"
    model.save(final_model)
    run_summary = {"method": "adaptive_curriculum", "status": "finished",
        "run_dir": str(run_dir), "events_path": str(events_path),
        "eval_path": str(run_dir / "eval.jsonl"), "final_model_path": str(final_model)}
    
    append_jsonl(events_path, {"type": "run_end", **run_summary})
    atomic_write_json(run_dir / "run_summary.json", run_summary)
    return run_summary


#add run result to the output log and CSV summary.
def record_result(result, runs, output_log, output_log_path, output_dir):
    runs.append(result)
    output_log["runs"] = runs
    atomic_write_json(output_log_path, output_log)
    save_runs_csv(output_dir, runs)


#parse arguments and run the adaptive exp
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True) #experiment_variants
    parser.add_argument("--val_dir", required=True) #validation_variants
    parser.add_argument("--output_dir", default="adaptive_curriculum_runs")
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
    parser.add_argument("--max_cached_asset_envs", type=int, default=16)
    args = parser.parse_args()


    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_assets = load_assets(train_dir, require_difficulty=True)
    val_assets = load_assets(val_dir, require_difficulty=False)


    output_log_path = output_dir / "output_log.json"
    seeds = random_seeds(args.num_seeds)
    runs = []

    output_log = {"train_dir": str(train_dir), "val_dir": str(val_dir),
        "method": "adaptive_curriculum", "timesteps": args.timesteps,
        "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes,
        "validation_asset_count": len(val_assets), "output_dir": str(output_dir), 
        "runs": runs}
    atomic_write_json(output_log_path, output_log)

    tasks = []
    for index, seed in enumerate(seeds, 1):
        run_dir = output_dir / f"run{index}"
        tasks.append({"seed": seed, "run_dir": str(run_dir),
            "train_assets": [vars(asset) for asset in train_assets],
            "val_assets": [vars(asset) for asset in val_assets],
            "args": vars(args)})

    #if one seed
    if args.jobs == 1:
        for task in tasks:
            print(f"running seed={task['seed']}")
            record_result(run_one(task), runs, output_log, output_log_path, output_dir)
    else: #if multiple seed start process pool
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_task = {pool.submit(run_one, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"method": "adaptive_curriculum", "status": "failed",
                        "error": repr(exc), "run_dir": task["run_dir"]}
                    print(f" ERROR seed={task['seed']}: {exc}")
                record_result(result, runs, output_log, output_log_path, output_dir)

    failed = [run for run in runs if run.get("status") != "finished"]
    print(f"wrote to output path: {output_log_path}")


if __name__ == "__main__":
    main()
    

