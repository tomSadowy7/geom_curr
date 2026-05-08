import csv
import gc
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from train_mujoco import MujocoMugLiftEnv, build_scene

#UTILS for run_adaptive_curriculum, run_curriculum_experiments, run_ood_experiments
#ALL Scripts of all classes must be run from top scope project dir

DIFFICULTIES = ("easy", "medium", "hard")

#store info needed to build and label one mug asset in memory
class AssetSpec:
    def __init__(self, name, path, params, difficulty=None):
        self.name = name
        self.path = path
        self.params = params
        self.difficulty = difficulty

    # Compare assets by their stored fields.
    def __eq__(self, other):
        return isinstance(other, AssetSpec) and vars(self) == vars(other)

#store the model location for one completed method and seed in memory
class RunSpec:
    def __init__(self, method, seed, run_dir, model_path):
        self.method = method
        self.seed = seed
        self.run_dir = run_dir
        self.model_path = model_path


#write JSON to temporary file so partial writes don't cause corruption if race condition occurs
def atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


#append one JSON object to a JSONL file
def append_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


#find dir under root that look like generated mug assets
def find_asset_dirs(root):
    if (root / "params.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("params.json"))


#read the difficulty value from a labels dictionary
def difficulty_from_label(label):
    difficulty = label.get("difficulty")
    if difficulty in DIFFICULTIES:
        return difficulty
    return None


#load all mug assets in a a directory.
def load_assets(root, require_difficulty=False):
    assets = []
    for path in find_asset_dirs(root):
        params = json.loads((path / "params.json").read_text())
        label = json.loads((path / "labels.json").read_text())
        difficulty = difficulty_from_label(label)
        if require_difficulty and difficulty is None:
            continue
        assets.append(AssetSpec(path.name, str(path), params, difficulty))
    return assets


#generate a list of random seeds for repeated runs
def random_seeds(count):
    rng = np.random.default_rng()
    return [int(seed) for seed in rng.choice(2**31 - 1, size=count, replace=False)]


#compute the end effector gripper scale needed for a mug geometry
def grip_scale_for(params, mug_scale):
    scale = params.get("scale", 0.20)
    x_end = params.get("x_end", 0.25)
    actual_diam = 2 * x_end * scale
    return min(1.0, 0.076 / actual_diam) * mug_scale


#Used to keep built Mujoco scene XML files for reuse -> increased speed
class SceneCache:
    def __init__(self, run_dir, mug_scale, max_cache_size=20):
        self.run_dir = Path(run_dir)
        self.mug_scale = mug_scale
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_order = []

    #return a cached scene or build it if needed
    def scene_for(self, asset):
        if asset.path in self.cache:
            return self.cache[asset.path]

        grip_scale = grip_scale_for(asset.params, self.mug_scale)
        scene_dir = self.run_dir / "scenes" / asset.name
        scene_xml = build_scene(Path(asset.path), asset.params, grip_scale, scene_dir)

        if len(self.cache) >= self.max_cache_size:
            oldest = self.access_order.pop(0)
            self.cache.pop(oldest, None)
            gc.collect()

        self.cache[asset.path] = (scene_xml, grip_scale)
        self.access_order.append(asset.path)
        return scene_xml, grip_scale

#set up common state for environments that rotate between mug assets.
class MultiAssetEnvBase(gym.Env):
    def __init__(self, assets, total_timesteps, run_dir, mug_scale, seed=0,
                 real_time=False, max_cached_asset_envs=1):
        super().__init__()
        self.assets = assets
        self.total_timesteps = max(1, total_timesteps)
        self.rng = np.random.default_rng(seed)
        self.training_step = 0
        self.real_time = real_time
        self.max_cached_asset_envs = max(1, int(max_cached_asset_envs))
        self.scene_cache = SceneCache(Path(run_dir), mug_scale)
        self.env = None
        self.current_asset = None
        self.env_cache = {}
        self.env_cache_order = []
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(MujocoMugLiftEnv.OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(MujocoMugLiftEnv.ACT_DIM,), dtype=np.float32)

    #lets callbacks tell the environment the current PPO step
    def set_training_step(self, step):
        self.training_step = int(step)

    #return the name of the asset active in the wrapped environment
    def current_asset_name(self):
        return self.current_asset.name if self.current_asset else None

    #add an environment to the small per-asset environment cache
    def remember_cached_env(self, asset_path, env):
        if asset_path in self.env_cache_order:
            self.env_cache_order.remove(asset_path)
        self.env_cache[asset_path] = env
        self.env_cache_order.append(asset_path)

        while len(self.env_cache_order) > self.max_cached_asset_envs:
            evicted_path = self.env_cache_order.pop(0)
            evicted_env = self.env_cache.pop(evicted_path, None)
            if evicted_env is not None:
                evicted_env.close()
        gc.collect() #IMPORTANT w/o this VM hits OOM and breaks

    #reset the current asset environment and switch assets when needed
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        asset = self.choose_asset()
        
        if self.env is None or self.current_asset != asset:
            cached = self.env_cache.get(asset.path)
            if cached is None:
                scene_xml, grip_scale = self.scene_cache.scene_for(asset)
                cached = MujocoMugLiftEnv(scene_xml=scene_xml, cup_dir=Path(asset.path),
                    mug_params=asset.params, grip_scale=grip_scale, real_time=self.real_time)
                self.remember_cached_env(asset.path, cached)
            else:
                self.remember_cached_env(asset.path, cached)
            self.env = cached
            self.current_asset = asset
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["asset"] = asset.name
        info["difficulty"] = asset.difficulty
        
        return obs, info

    #step one action to the active asset environment
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["asset"] = self.current_asset.name
        info["difficulty"] = self.current_asset.difficulty
        return obs, reward, terminated, truncated, info

    #close cached MuJoCo environments -> crazy memory leakage without this
    def close(self):
        for env in self.env_cache.values():
            env.close()
        self.env_cache.clear()
        self.env_cache_order.clear()
        self.env = None

#track training events and write them to a JSONL file
class JsonTrainLogger(BaseCallback):
    def __init__(self, events_path, log_freq=10_000):
        super().__init__()
        self.events_path = events_path
        self.log_freq = log_freq
        self.last_log = 0
        self.last_gc = 0
        self.episode_rewards = []
        self.success_count = 0

    #log success, episode, and mean reward events during training
    def _on_step(self):
        self.training_env.env_method("set_training_step", self.num_timesteps)
        for info in self.locals.get("infos", []):
            if info.get("is_success", False):
                self.success_count += 1
                append_jsonl(self.events_path, {"type": "success",
                    "step": self.num_timesteps, "success_count": self.success_count,
                    "lift_cm": 100.0 * float(info.get("lift_m", 0.0)), "asset": info.get("asset"),
                    "difficulty": info.get("difficulty")})
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                append_jsonl(self.events_path, {"type": "episode", "step": self.num_timesteps,
                    "reward": float(info["episode"]["r"]), "length": int(info["episode"]["l"]),
                    "asset": info.get("asset"), "difficulty": info.get("difficulty"), "is_success": bool(info.get("is_success", False))})

        if self.num_timesteps - self.last_log >= self.log_freq and self.episode_rewards:
            append_jsonl(self.events_path, {"type": "mean_ep_reward", "step": self.num_timesteps,
                "mean_ep_reward": float(np.mean(self.episode_rewards)), "episodes": len(self.episode_rewards)})
            self.episode_rewards.clear()
            self.last_log = self.num_timesteps

        if self.num_timesteps - self.last_gc >= 50_000:
            gc.collect()
            self.last_gc = self.num_timesteps

        return True


#eval a policy on a list of assets and summarize success rates
def evaluate_policy_on_assets(model, assets, run_dir, mug_scale, step,
            episodes_per_asset, seed, result_type="validation",
            success_key="validation_success_rate",
            scene_folder="validation_scenes",
            progress_prefix=""):
    
    rows = []
    total_rewards = []
    successful_assets = 0
    rng = np.random.default_rng(seed + (step or 0))

    for index, asset in enumerate(assets, 1):
        if progress_prefix:
            print(f"{progress_prefix} asset {index}/{len(assets)}: {asset.name}", flush=True)

        grip_scale = grip_scale_for(asset.params, mug_scale)
        scene_dir = Path(run_dir) / scene_folder / asset.name
        scene_xml = build_scene(Path(asset.path), asset.params, grip_scale, scene_dir)
        asset_successes = 0
        asset_rewards = []
        env = MujocoMugLiftEnv(scene_xml=scene_xml, cup_dir=Path(asset.path),
            mug_params=asset.params, grip_scale=grip_scale)
        
        #go through episodes
        for episode in range(episodes_per_asset):
            obs, reset_info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = False
            episode_reward = 0.0
            episode_success = False
            while not done:
                action, state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                episode_success = episode_success or bool(info.get("is_success", False))
                done = bool(terminated or truncated)
            asset_successes += int(episode_success)
            asset_rewards.append(episode_reward)
            total_rewards.append(episode_reward)

        env.close()
        del env
        gc.collect()

        asset_success = asset_successes > 0
        successful_assets += int(asset_success)
        rows.append({"asset": asset.name, "asset_path": asset.path,
            "episodes": episodes_per_asset, "successes": asset_successes,
            "asset_success": asset_success, "mean_reward": float(np.mean(asset_rewards)) if asset_rewards else None})

    result = {"num_assets": len(assets), "episodes_per_asset": episodes_per_asset, "asset_successes": successful_assets,
        success_key: successful_assets / max(1, len(assets)), "average_reward": float(np.mean(total_rewards)) if total_rewards else None,
        "assets": rows}
    if result_type:
        result["type"] = result_type
    if step is not None:
        result["step"] = step
        
    gc.collect()
    return result

#save checkpoints and run validation on a fixed interval.
class CheckpointEvalCallback(BaseCallback):
    def __init__(self, run_dir, val_assets, mug_scale, eval_interval, eval_episodes, seed):
        super().__init__()
        self.run_dir = run_dir
        self.val_assets = val_assets
        self.mug_scale = mug_scale
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.last_eval = 0
        self.eval_path = run_dir / "eval.jsonl"
        self.checkpoint_dir = run_dir / "checkpoints"

    #run validation when the evaluation interval has elapsed
    def _on_step(self):
        if self.num_timesteps - self.last_eval < self.eval_interval:
            return True
        self.run_eval(self.num_timesteps)
        self.last_eval = self.num_timesteps
        return True

    #run one final validation pass at training end
    def _on_training_end(self):
        if self.num_timesteps != self.last_eval:
            self.run_eval(self.num_timesteps)

    #save a checkpoint and append its validation result
    def run_eval(self, step):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"step_{step:08d}.zip"
        self.model.save(checkpoint_path)
        result = evaluate_policy_on_assets(self.model, self.val_assets,
            self.run_dir, self.mug_scale, step, self.eval_episodes, self.seed)
        result["checkpoint_path"] = str(checkpoint_path)
        append_jsonl(self.eval_path, result)
        print(f"eval: step={step:,} success_rate={result['validation_success_rate']:.3f} avg_reward={result['average_reward']:.3f}")
        gc.collect()


#save run summaries in a compact CSV file
def save_runs_csv(output_dir, runs):
    path = output_dir / "runs.csv"
    fields = ["method", "status", "error", "run_dir", "events_path", "eval_path", "final_model_path"]
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run.get(field) for field in fields})
