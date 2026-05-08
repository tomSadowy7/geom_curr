import argparse
import csv
import json
from pathlib import Path

#BIN ASSETS AND ACTUALYL WRITE TO LABELS.JSON THEIR LABEL

DIFFICULTIES = ("easy", "medium", "hard")

#load asset names and scores from CSV file
def load_score_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asset = row["asset"].strip()
            score = float(row["asset_score"])
            rows.append({"asset": asset, "score": score, "csv_row": row})
    return rows

#assign easy, medium, and hard labels based on score ranking 
#(lowest third = hard) since technically scoring function from scoring script is like learnability score
#so difficulty is complement of it
def assign_labels(rows):
    ranked = sorted(rows, key=lambda row: (row["score"], row["asset"]))
    n = len(ranked)
    first_cut = n // 3
    second_cut = (2 * n) // 3

    labeled = []
    for index, row in enumerate(ranked):
        if index < first_cut:
            difficulty = "hard"
        elif index < second_cut:
            difficulty = "medium"
        else:
            difficulty = "easy"

        labeled.append({"asset": row["asset"], "asset_score": row["score"],
            "rank_ascending": index + 1, "num_assets": n, "difficulty": difficulty,
            "csv_row": row["csv_row"]})

    return labeled

#write labels.json file to each asset directory
def write_labels(labeled, asset_dir):
    written = []

    for row in labeled:
        asset_path = asset_dir / row["asset"]
        label_path = asset_path / "labels.json"
        data = {"asset_score": row["asset_score"], "label_type": "quantitative_reward_bin", "difficulty": row["difficulty"]}
        label_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        written.append((label_path, row["difficulty"], row["asset_score"]))

    return written

#print summary of labels assigned and written
def print_summary(labeled, written):
    counts = {difficulty: 0 for difficulty in DIFFICULTIES}
    for row in labeled:
        counts[row["difficulty"]] += 1

    #print stats
    print(f"num of assets scored: {len(labeled)}")
    print(f"label split: easy={counts['easy']} | medium={counts['medium']} | hard={counts['hard']}")
    print(f"wrote to path: {len(written)} labels.json files")

#run labeling pipeline -> you need valid Infinigen generated dir to parse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score_csv") #path to asset_reward_scores.csv
    parser.add_argument("asset_dir") #directory containing CupFactory_XXX folders"
    args = parser.parse_args()

    #run through pipeline
    csv_path = Path(args.score_csv)
    asset_dir = Path(args.asset_dir)
    rows = load_score_rows(csv_path)
    labeled = assign_labels(rows)
    written = write_labels(labeled, asset_dir)
    print_summary(labeled, written)


if __name__ == "__main__":
    main()
