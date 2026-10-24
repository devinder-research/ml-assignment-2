"""
run_all.py - run the whole pipeline in order:
    1. create + save the split (skipped if splits/split_seed.npz exists)
    2. Q1: tree baseline + bagging          (~5 min)
    3. Q2: SAMME boosting                   (~3 min)
    4. Q3: bias/variance over 30 refits     (~45 min on 2 cores)

    python src/run_all.py
"""

import time

from data import REPO, SEED, log
from pathlib import Path
import subprocess
import sys

STAGES = ["make_split", "q1", "q2", "q3"]


def main():
    logp = log("run_all")
    logp(f"ML Assignment 2 | roll P26AI0004 | seed {SEED}")
    logp(f"stages: {', '.join(STAGES)}")
    for stage in STAGES:
        t0 = time.perf_counter()
        logp(f"=== stage: {stage} " + "=" * (50 - len(stage)))
        r = subprocess.run([sys.executable, str(REPO / "src" / f"{stage}.py")])
        if r.returncode != 0:
            sys.exit(f"stage {stage} failed (exit {r.returncode})")
        logp(f"=== {stage} finished in {time.perf_counter() - t0:.0f}s")
    logp("all stages complete. Results in results/, figures in figures/, "
         "logs in logs/.")


if __name__ == "__main__":
    main()
