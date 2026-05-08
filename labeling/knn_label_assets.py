import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


#KNN SCRIPT for object parameter space -> prod used K=3

DIFFICULTIES = {"easy", "medium", "hard"}

#find all continuous numeric parameter keys across assets
def continuous_keys(paths):
    keys = set()
    for path in paths:
        for key, value in json.loads((path / "params.json").read_text()).items():
            if key not in {"factory_seed"} and not isinstance(value, bool) and isinstance(value, int | float):
                keys.add(key) #exclude factory_seed in object param space
    return sorted(keys)


#extract continuous parameter values as feature vector
def feature_vector(path, keys):
    params = json.loads((path / "params.json").read_text())
    return [float(params[key]) if (key not in {"factory_seed"} and not isinstance(params.get(key), bool) and isinstance(params.get(key), (int, float))) else 0.0 for key in keys]


#load labeled assets and their difficulty labels
def load_labeled(labeled_dir):
    paths = []
    labels = []
    for path in sorted(p for p in labeled_dir.iterdir() if p.is_dir() and (p / "params.json").exists()):
        label_path = path / "labels.json"
        if not label_path.exists():
            continue
        difficulty = json.loads(label_path.read_text()).get("difficulty")
        if difficulty in DIFFICULTIES:
            paths.append(path)
            labels.append(difficulty)

    return paths, labels


#write difficulty label to asset directory
def write_label(path, difficulty):
    data = {
        "label_type": "knn",
        "difficulty": difficulty,
    }
    (path / "labels.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


#run KNN on labeled assets and predict labels for target assets
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labeled_dir") #asset directory with labels.json files
    parser.add_argument("target_dir") #asset directory to pseudo-label
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    #resolve paths for dirs
    labeled_dir = Path(args.labeled_dir)
    target_dir = Path(args.target_dir)
    labeled_paths, y = load_labeled(labeled_dir)
    target_paths = sorted(p for p in target_dir.iterdir() if p.is_dir() and (p / "params.json").exists())
   
    #extract features
    keys = continuous_keys(labeled_paths + target_paths)
    X_labeled = np.array([feature_vector(path, keys) for path in labeled_paths], dtype=np.float32)
    X_target = np.array([feature_vector(path, keys) for path in target_paths], dtype=np.float32)

    k = min(args.k, len(labeled_paths))
    scaler = StandardScaler()
    X_labeled = scaler.fit_transform(X_labeled)
    X_target = scaler.transform(X_target)

    #runn KNN 
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_labeled, np.array(y))
    predictions = knn.predict(X_target)

    for path, difficulty in zip(target_paths, predictions):
        write_label(path, str(difficulty))

    counts = Counter(str(label) for label in predictions)
    print(f"labeled {len(target_paths)} assets")
    print(f"difficulties: easy={counts.get('easy', 0)} | medium={counts.get('medium', 0)} | hard={counts.get('hard', 0)}")


if __name__ == "__main__":
    main()

