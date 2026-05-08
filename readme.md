## Hi! 
We're Tomasz, Mihika, Shane, and Katherine. We were interested to see if 
running a curriculum on geometric object parameter variation helps with sample efficiency and robustness for manipulation tasks. We specifically chose to look at mug lifting. Unlike other settings where difficulty progression is more intuitive, such as locomotion tasks, we had to first deal with how to label different mug geometries. Although a mug can look difficult to us humans, it may actually have a very simple grasp affordance that PPO can pick up right away- we alone can't just eyeball difficulty! Thus, we came up with a quantitative way to label difficulty: we train PPO on a small subset of mugs, extract statistics from the runs, and use an aggregated learnability function to bin geometries into "easy", "medium", or "hard" based on their training statistics. To not have to run PPO on every single training mug geometry, we save compute by pseudo-labeling the rest of the mugs via KNN in object parameter space using the PPO labeled mugs as anchor points. After we have a full set of labeled training geometries, we are able to test our curriculum method against a few baselines. Interestingly, we found that our method has seems to have better sample efficiency and robustness to OOD geometries than the baselines (yay!). Our repository hierarchy is defined below:

- `gen_objects/` - scripts to generate mugs (using ProCura on top of Infinigen) and to extract collision geometries as VHACDs (needed for handle grasping)
- `labeling/` - scripts for reading training statistics, binning geometries based on them, and running KNN in object parameter space
- `run_experiments/` - scripts for running experiments (our curriculum vs three baselines) as well ablations
- `train/` - Mujoco mug lifting environment and script for PPO difficulty-labeling training
- `for_procura/` - file to copy/paste into ProCura (explained in Installation & Setup)

### Installation & Setup
To start, first git clone this repository while also recursely cloning the submodules with `git clone --recurse-submodules git@github.com:tomSadowy7/geom_curr.git`. We need `ProCura`, which is a scene reconstruction layer on top of Infinigen that Tomasz created while an undergraduate researcher at the Princeton Vision & Learning Lab, to perturb object parameters and `mujoco_menagerie` for the Franka arm asset. Then, perform `mv gen_objects/* ProCura/` as the `gen_object` scripts need to be in the ProCura top-level directory and also replace `ProCura/files_to_replace/cup.py` with `for_ProCura/cup.py` (this allows for params.json to be saved). Afterwards, please follow Infinigen's instructions (Infinigen is nested inside of ProCura) on how to install the Infinigen environment. We'll also need to `pip -m install` a few libraries to the Infinigen environment which you can find below. 

Libraries to pip install to the Infinigen environment:
- pybullet
- stable-baselines3
- gymnasium
- mujoco

Also make sure to have `zip` installed on your system. For mac users: `brew install zip` and for ubuntu users, `sudo apt-get install zip`

### Generating Mugs
Generating mugs is pretty simple. First `cd ProCura` and then run `python generate_cups.py --n 20 --output_dir mugs`. 

One last step we need to do is to convert the collision geometries into VHACDs to allow for handle grasping, so we run `python export_cup_meshes.py --cup_dir mugs`.

### Labeling
After our mugs have been generated, we'll select a subset of the mug directories (CupFactory_***) and transfer them to a new directory we call `train_variants/` and transfer the rest of the mugs to a directory we make called `to_knn_label/`. 

Then, we'll run `python -m train.train_all train_variants --output_dir ../train_all_results`. This will run PPO on the mug geometries and save training statistics to `train_all_results/`. 

Next, we'll run `python -m labeling.score_training_results train_all_results/results.json --out_csv training_stats.csv`. This aggregates the training statistics and saves them to `training_stats.csv`. 

From there, we'll run `python -m labeling.label_assets_from_scores training_stats.csv train_variants`. This will label the mugs we just PPO'd. 

Finally, we can run `python -m labeling.knn_label_assets train_variants to_knn_label`. This will label the rest of the mugs with KNN.

Now since all of our training mugs have a labels.json, we can aggregate all the mug directories back into a shared directory called `experiment_variants`

### Running Experiments
Now we can get to the fun stuff- actually running the experiments! First, we'll like to see how our curriculum performs against baselines in terms of sample efficiency. Prior to training, we need to generate a set of validation mugs that will be used to validate each method during each interval of training. We can do this in the same way we did in the Generating Mugs section, and we'll save them in `validation_variants`. Then, we can simply run 
```bash
python -m run_experiments.run_curriculum_experiments \
  --train_dir experiment_variants \
  --val_dir validation_variants \
  --fixed_asset path/to/CupFactory_***/we/want/ \
  --output_dir experiment_runs/curriculum_vs_baselines \
  --methods curriculum fixed uniform hard_only \
  --timesteps 3000000 \
  --eval_interval 200000 \
  --eval_episodes 10 \
  --num_seeds 3
```

This will print out training statistics as well as evaluation statistics per method per seed which we can analyze to determine sample efficiency!

To evaluate how our method performs on OOD geometries against the baselines, we'll first generate a set of ood distributions (we need to change the ranges in cup.py) in the same way as we did in the Generating Mugs section and we'll save them to `ood_variants\`. Then, we want to extract the saved final models from our previous experiment's output directory and then set them up in this hierarchy:

```text
for_ood_experiment/
  curriculum/
    seed0/
      final_model.zip
    seed2/
      final_model.zip
  fixed/
    seed0/
      final_model.zip
    seed2/
      final_model.zip
  hard_only/
    seed0/
      final_model.zip
    seed2/
      final_model.zip
  uniform/
    seed0/
      final_model.zip
    seed2/
      final_model.zip
```

Then, we can run 
```bash
python -m run_experiments.run_ood_experiments \
  --experiment_dir for_ood_experiment \
  --ood_dir ood_variants \
  --output_dir ood_evaluation_results
```
This will give us results on how our curriculum method performs on OOD geometries vs the baselines. 

Finally, feel free to check out `run_experiments/run_adaptive_curriculum_experiments.py` for one of our ablations where we implemented a performance-gated scheduler for our curriculum.




Have fun exploring!