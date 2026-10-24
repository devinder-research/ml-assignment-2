"""
make_split.py - create and save the train/val/test split.

    python src/make_split.py            (refuses to overwrite an existing split)
    python src/make_split.py --force    (regenerate)

The split is 6 train / 2 val / 2 test images WITHIN each of the 40
identities, drawn with numpy.random.default_rng(260004) (the numeric
characters of roll number P26AI0004, in order). It is saved to splits/
and every experiment loads it from disk, so all stages share one
identical split.
"""

import argparse

from data import REPO, SEED, log, load_faces, load_split, make_split, \
    save_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing saved split")
    args = ap.parse_args()
    logp = log("make_split")

    if (REPO / "splits" / "split_seed.npz").exists() and not args.force:
        split = load_split()
        logp(f"Saved split already exists and is verified: "
             f"train {split['train'].size}, val {split['val'].size}, "
             f"test {split['test'].size}. Nothing to do.")
        return

    logp(f"Loading Olivetti faces (seed {SEED})")
    X, y = load_faces()
    logp(f"Data: X {X.shape}, y {y.shape}, values in "
         f"[{X.min():.1f}, {X.max():.1f}]")

    split, per_id = make_split(y, seed=SEED)
    payload = save_split(split, per_id, seed=SEED)
    logp(f"Saved splits/split_seed.npz and splits/split_seed.json")
    logp(f"sha256(split indices) = {payload['sha256_split_indices']}")
    logp(f"train {split['train'].size}, val {split['val'].size}, "
         f"test {split['test'].size}  (6/2/2 inside each identity)")


if __name__ == "__main__":
    main()
