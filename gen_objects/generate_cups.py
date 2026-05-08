import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

#functions adapted from main.py of ProCura same structure
#still need ProCura cloned though bc it provides exposed param cup.py

CUP_INFINIGEN_PATH = Path(__file__).parent / "infinigen_tp/infinigen/assets/objects/tableware/cup.py"
CUP_REPLACEMENT_PATH = Path(__file__).parent / "files_to_replace/cup.py"
CUP_BACKUP_PATH = Path(__file__).parent / "files_to_replace/original/cup.py"

#backup original cup.py and replace with customized version
def copy_cup():
    CUP_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(CUP_INFINIGEN_PATH, CUP_BACKUP_PATH)
    shutil.copy(CUP_REPLACEMENT_PATH, CUP_INFINIGEN_PATH)
    print(f"replaced infinigen cup.py with files_to_replace/cup.py")


#restore original cup.py
def revert_cup():
    if CUP_BACKUP_PATH.exists():
        shutil.copy(CUP_BACKUP_PATH, CUP_INFINIGEN_PATH)
        print(f"reverted infinigen cup.py to original")


#generate next randomized cups using Infinigen
def generate_cups(n=20, output_dir="outputs/cup_variants"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    copy_cup()
    try:
        cmd = [sys.executable, "-m", "infinigen_examples.generate_individual_assets",
            "--output_folder", str(out),"-f", "CupFactory",
            "-n", str(n), "--save_blend"]
        env = os.environ.copy()
        env["CUP_OUTPUT_DIR"] = str(out.resolve())
        print(f"generating {n} cups -> {out}")
        subprocess.run(cmd, env=env, check=True)
    finally:
        revert_cup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="experiment_variants")
    args = parser.parse_args()
    generate_cups(n=args.n, output_dir=args.output_dir)
