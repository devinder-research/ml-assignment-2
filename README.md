# Bagging, Boosting and Bias-Variance on Face Recognition

Machine Learning - Assignment 2 (Olivetti faces).

Student: **Devinder**, Roll No. **P26AI0004**

## What is here

- `src/data.py` - data loading, the seeded 6/2/2 split (create/save/load),
  the masked test set and the small metric helpers.
- `src/make_split.py` - creates `splits/split_seed.npz` + `split_seed.json`
  once; every experiment loads the split from disk afterwards.
- `src/bagging.py` - bagging from scratch (bootstrap sampling, majority
  vote, out-of-bag accuracy, pairwise disagreement, vote margins), with
  input validation (empty/mismatched data, invalid labels, invalid
  ensemble parameters, bootstrap-index checks) and clean refit semantics.
- `src/boosting.py` - multiclass SAMME AdaBoost from scratch, including
  the explicit handling of degenerate weak learners. `fit()` resets all
  model state, so fitting the same object twice gives exactly the model
  of a fresh object; all inputs and ensemble prefixes are validated.
- `src/q1.py`, `src/q2.py`, `src/q3.py` - the three question scripts.
  Each one also writes its figures into `figures/`. `q2.py` handles
  early stopping explicitly (reports the accepted learners, marks
  unreachable prefixes unavailable, never indexes empty histories).
- `src/run_all.py` - runs everything in order.
- `scripts/validate_results.py` - independent validation: recomputes the
  tree grid, the bagging sweep, the OOB diagnostics, the SAMME toy round,
  the boosting selection, the clean/masked evaluations and all Q3
  statistics from the saved split/checkpoints, and compares them to the
  saved JSONs (PASS/FAIL per check, nonzero exit code on failure).
- `config.json` - the single source of experimental settings (seed,
  tree grid, B values, boosting depths/rounds, Q3 repetitions, depth
  study, comparison learner sizes); the scripts read everything from
  here instead of duplicating hard-coded values.
- `results/` - every number reported in the report, as JSON.
- `figures/`, `logs/` - figures and run logs (the figure scripts live in
  the `q*.py` files that produced them).
- `data/olivetti_faces.npz` - cached copy of `fetch_olivetti_faces`
  (400x4096 raw pixels, labels 0..39) so the code runs offline.

## Environment setup

Python 3.12 with the pinned dependencies in `requirements.txt`
(numpy 2.1.3, scipy 1.14.1, scikit-learn 1.5.2, matplotlib 3.9.2,
joblib 1.5.3). Either option works:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

or the conda environment file:

```bash
conda env create -f environment.yml
conda activate ml-assignment-2
```

## Reproduction

```bash
python src/run_all.py                  # full pipeline, ~55 min on 2 cores
```

Stages can also be run separately, in this order:

```bash
python src/make_split.py    # ~5 s   (refuses to overwrite an existing split)
python src/q1.py            # ~6 min (tree grid, bagging sweep, OOB, cases)
python src/q2.py            # ~4 min (toy round, boosting grid, comparison)
python src/q3.py            # ~45 min (30 bias/variance refits, checkpointed)
```

Expected outputs per stage: `make_split.py` writes `splits/split_seed.npz`
+ `splits/split_seed.json` and prints the sha256 of the split indices;
`q1.py`/`q2.py`/`q3.py` rewrite `results/q1*.json` / `q2*.json` / `q3*.json`,
the figures in `figures/` and append to the run logs in `logs/`. Only
wall-clock training times and log timestamps differ between runs - every
measured number is reproduced exactly from the seed.

`--force`: `python src/make_split.py --force` regenerates the saved split
(only needed if `splits/` was deleted or corrupted). Without `--force` the
script verifies the existing split against its stored sha256 and exits.

**Q3 checkpoint resumption**: every Q3 repetition is checkpointed to
`results/checkpoints/q3_rep{r}.npz` as soon as it finishes. If the run is
interrupted, simply re-run `python src/q3.py` (or `run_all.py`): finished
repetitions are loaded from disk and only the missing ones are computed.
To redo Q3 from scratch, delete `results/checkpoints/` first.

## Validation (test execution)

```bash
python scripts/validate_results.py
```

The script re-derives every reported quantity independently of the saved
JSONs (tree grid, bagging sweep, OOB diagnostics, SAMME toy round,
boosting grid/selection, clean/masked evaluations, Q3 statistics from the
30 checkpoints, original-split bagging refits) and additionally checks
refit determinism (fit twice == fit fresh) and the input-validation
behaviour of both ensembles. Each check prints PASS or FAIL; the exit
code is 1 when any check fails. Expected output: `summary: 23 PASS,
0 FAIL` in roughly 10-15 minutes (the 200-tree sweep and the Q3(c)
bagging refits dominate). Individual checks can be run by name, e.g.
`python scripts/validate_results.py tree_grid q3_statistics` (valid
names: input_validation, tree_grid, bagging_sweep, oob_diagnostics,
samme_toy, boosting_selection, clean_masked, q3_statistics,
refit_determinism).

## Random seeds

Every random decision comes from `numpy.random.default_rng(260004)`
(260004 = the numeric characters of the roll number P26AI0004, in order):

- split: one `default_rng(260004)` stream, one `permutation` per identity;
- bagging: one `default_rng(260004)` draw `integers(0, 240, (200, 240))`,
  the size-B ensemble is the first B trees;
- Q3 resamples: fresh `default_rng(260004)`, `integers(0, 6, (30, 40, 6))`;
- Q3 per-repetition bagging: `default_rng([260004, r])`;
- all base trees use `random_state=260004` (the fits are deterministic).

Re-running any stage on the same machine reproduces the saved results
numerically; only wall-clock times and log timestamps differ.

## Results layout

`results/q1a_tree_grid.json`, `q1b_bagging_sweep.json`,
`q1c_diagnostics.json`, `q1d_clean_masked.json` for Q1;
`q2a_synth_round.json`, `q2b_boosting_grid.json`,
`q2b_selected_dynamics.json`, `q2c_bag_vs_boost.json`,
`q2d_final_comparison.json` for Q2; `q3a_bias_variance.json`,
`q3b_depth_study.json`, `q3c_deepest_analysis.json`,
`q3d_final_table.json` for Q3 (per-repetition checkpoints under
`results/checkpoints/`).

Note on the Q3 depth-study labels: the selected tree has **unlimited**
depth, so no deeper configuration exists; the depth study therefore
compares a shallow tree (4), a deep *bounded* tree (16, deeper than the
shallow tree but not deeper than the selected one) and the selected
unlimited tree (roles `shallow` / `deeper_bounded` / `selected` in
`config.json` and `results/q3b_depth_study.json`).
