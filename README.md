# Bagging, Boosting and Bias-Variance on Face Recognition

Machine Learning - Assignment 2 (Olivetti faces).

Student: **Devinder**, Roll No. **P26AI0004**

## What is here

- `src/data.py` - data loading, the seeded 6/2/2 split (create/save/load),
  the masked test set and the small metric helpers.
- `src/make_split.py` - creates `splits/split_seed.npz` + `split_seed.json`
  once; every experiment loads the split from disk afterwards.
- `src/bagging.py` - bagging from scratch (bootstrap sampling, majority
  vote, out-of-bag accuracy, pairwise disagreement, vote margins).
- `src/boosting.py` - multiclass SAMME AdaBoost from scratch, including
  the explicit handling of degenerate weak learners.
- `src/q1.py`, `src/q2.py`, `src/q3.py` - the three question scripts.
  Each one also writes its figures into `figures/`.
- `src/run_all.py` - runs everything in order.
- `config.json` - all experimental settings (seed, grids, B values, ...).
- `results/` - every number reported in the report, as JSON.
- `figures/`, `logs/` - figures and run logs (the figure scripts live in
  the `q*.py` files that produced them).
- `data/olivetti_faces.npz` - cached copy of `fetch_olivetti_faces`
  (400x4096 raw pixels, labels 0..39) so the code runs offline.

## Reproduction

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml
python src/run_all.py                  # full pipeline, ~55 min on 2 cores
```

Stages can also be run separately, in this order:

```bash
python src/make_split.py    # ~5 s   (skipped if the split already exists)
python src/q1.py            # ~6 min (tree grid, bagging sweep, OOB, cases)
python src/q2.py            # ~4 min (toy round, boosting grid, comparison)
python src/q3.py            # ~45 min (30 bias/variance refits, checkpointed)
```

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
