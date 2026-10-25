"""
q3.py - Q3: bias and variance as measured quantities.

Protocol:
  * the final CLEAN test set stays fixed;
  * 30 new training sets: within each identity, 6 faces drawn with
    replacement from that identity's original 6 training faces. Exact
    draw: fresh rng = default_rng(260004); g = rng.integers(0, 6,
    size=(30, 40, 6)); resamples[r, c, :] = identity-c train indices.
  * on each resampled set we refit the selected single tree, the final
    bagged model and the selected boosting model (no re-tuning), plus
    single trees and 100-tree bagged ensembles at the two depth-study
    depths from config.json (q3.depths). Because the selected depth is
    unlimited, no configuration deeper than the selected tree exists:
    the study compares a shallow bounded tree (4), a deep bounded tree
    (16) and the selected unlimited tree. Bagging bootstraps inside
    repetition r use their own stream: default_rng([260004, r]).
  * p_r(x_i): class vector on the fixed test set - predict_proba for the
    tree, vote fractions for bagging, normalised weighted vote q_k for
    boosting. Then
      Bias2   = (1/N)    sum_i   || p_bar(x_i) - e_i ||^2
      Variance= (1/(N R)) sum_{i,r} || p_r(x_i) - p_bar(x_i) ||^2

Each repetition is checkpointed to results/checkpoints/q3_rep{r}.npz, so
an interrupted run resumes where it stopped.

Run:  python src/q3.py      (needs Q1 and Q2 results first)
"""

import time

import numpy as np
from joblib import Parallel, delayed

from bagging import BaggingEnsemble, make_tree
from boosting import SammeBoost
from data import (CFG, REPO, SEED, N_CLASSES, accuracy, log, load_faces,
                  load_json, load_split, one_hot, save_json, vote_fractions)

RESULTS = REPO / "results"
CKPT = RESULTS / "checkpoints"

# Q3 depth study, from config.json. The roles are named so that depth 16
# is never described as deeper than the selected tree: the selected depth
# is unlimited (null), so a deeper configuration is unavailable and depth
# 16 is the deeper BOUNDED comparison.
_Q3_DEPTHS = CFG["q3"]["depths"]
DEPTH_STUDY = [("shallow", _Q3_DEPTHS["shallow"]),
               ("selected", _Q3_DEPTHS["selected"]),
               ("deeper_bounded", _Q3_DEPTHS["deeper_bounded"])]
N_REPS = CFG["q3"]["n_repetitions"]
BAG_SIZE = CFG["q3"]["bag_size"]


def depth_key(d):
    """Checkpoint key prefix for a studied depth."""
    return "sel" if d is None else f"d{d}"


def make_resamples(split, y):
    """The 30 resampled training sets (exact documented procedure)."""
    train_idx = split["train"]
    per_id = [train_idx[y[train_idx] == c] for c in range(40)]
    rng = np.random.default_rng(SEED)
    g = rng.integers(0, 6, size=(N_REPS, 40, 6))
    resamples = np.empty((N_REPS, 40, 6), dtype=np.int64)
    for c in range(40):
        resamples[:, c, :] = per_id[c][g[:, c, :]]
    return resamples


def rep_worker(r, resamples, X, y, split, frozen_msl, boost_depth,
               boost_rounds):
    """Fit everything on resample r and save a checkpoint."""
    Xte, yte = X[split["test"]], y[split["test"]]
    idx = resamples[r].reshape(-1)                # 240 indices, with repeats
    Xr, yr = X[idx], y[idx]

    out = {}
    # single trees at the three depths (min_samples_leaf = frozen 4)
    for _, d in DEPTH_STUDY:
        key = depth_key(d)
        t = make_tree({"max_depth": d, "min_samples_leaf": frozen_msl})
        t.fit(Xr, yr)
        out[f"P_tree_{key}"] = t.predict_proba(Xte)
        out[f"acc_tree_{key}"] = np.array([accuracy(yte, t.predict(Xte))])
        out[f"leaves_tree_{key}"] = np.array([float(t.get_n_leaves())])

    # 100-tree bagged ensembles at the same three depths
    for _, d in DEPTH_STUDY:
        key = depth_key(d)
        ens = BaggingEnsemble({"max_depth": d, "min_samples_leaf": frozen_msl},
                              BAG_SIZE, seed=SEED)
        ens.fit(Xr, yr, rng=np.random.default_rng([SEED, r]))
        tp = ens.tree_predictions(Xte)
        votes = vote_fractions(tp)
        out[f"P_bag_{key}"] = votes
        out[f"acc_bag_{key}"] = np.array(
            [accuracy(yte, np.argmax(votes, axis=1))])
        iu = np.triu_indices(100, k=1)
        out[f"dis_bag_{key}"] = np.array(
            [float((tp[:, None, :] != tp[None, :, :])[iu].mean())])

    # selected boosting model, frozen depth and rounds
    boost = SammeBoost({"max_depth": boost_depth, "min_samples_leaf": 1},
                       n_rounds=boost_rounds, seed=SEED).fit(Xr, yr)
    out["P_boost"] = boost.qk(Xte)
    out["acc_boost"] = np.array([accuracy(yte, boost.predict(Xte))])

    out["rep"] = np.array([float(r)])
    np.savez(CKPT / f"q3_rep{r}.npz", **out)
    return {"rep": r, "boost_rounds_run": boost.n_rounds_run,
            "stop_reason": boost.stop_reason}


# ---------------------------------------------------------------------------
def aggregate(X, y, split, logp):
    yte = y[split["test"]]
    e = one_hot(yte)
    reps = range(N_REPS)
    data = []
    for r in reps:
        with np.load(CKPT / f"q3_rep{r}.npz") as z:
            data.append({k: z[k] for k in z.files})

    configs = ([f"tree_{depth_key(d)}" for _, d in DEPTH_STUDY]
               + [f"bag_{depth_key(d)}" for _, d in DEPTH_STUDY]
               + ["boost"])
    stats = {}
    for name in configs:
        P = np.stack([d[f"P_{name}"] for d in data])       # (R, N, K)
        acc = np.array([d[f"acc_{name}"][0] for d in data])
        p_bar = P.mean(axis=0)
        stats[name] = {
            "bias2": float(np.mean(np.sum((p_bar - e) ** 2, axis=1))),
            "variance": float(np.mean(np.sum((P - p_bar[None]) ** 2,
                                             axis=2))),
            "acc_mean": float(acc.mean()),
            "acc_std": float(acc.std()),
        }

    # ---- Q3(a) ----------------------------------------------------------
    sel_key = depth_key(dict(DEPTH_STUDY)["selected"])
    a = {"tree": stats[f"tree_{sel_key}"],
         "bagging": stats[f"bag_{sel_key}"], "boosting": stats["boost"]}
    logp("Q3(a)  model | bias^2 | variance | test acc over the 30 refits")
    for k, v in a.items():
        logp(f"  {k:<9s} | {v['bias2']:.4f} | {v['variance']:.4f} | "
             f"{v['acc_mean']:.4f} +- {v['acc_std']:.4f}")
    save_json(a, RESULTS / "q3a_bias_variance.json")

    # ---- Q3(b): depth study ----------------------------------------------
    q1a = load_json(RESULTS / "q1a_tree_grid.json")
    occ = {str(r["max_depth"]): r for r in q1a["grid"]
           if r["min_samples_leaf"] == 4}
    rows = []
    for role, d in DEPTH_STUDY:
        key = depth_key(d)
        st, sb = stats[f"tree_{key}"], stats[f"bag_{key}"]
        o = occ[str(d)]
        rows.append({
            "role": role, "max_depth": d,
            "tree_bias2": st["bias2"], "tree_var": st["variance"],
            "tree_acc": st["acc_mean"],
            "bag_bias2": sb["bias2"], "bag_var": sb["variance"],
            "bag_acc": sb["acc_mean"],
            "var_removed_frac": (st["variance"] - sb["variance"])
                                / st["variance"],
            "acc_gain": sb["acc_mean"] - st["acc_mean"],
            "n_leaves": o["n_leaves"],
            "mean_samples_per_leaf": o["mean_samples_per_leaf"],
            "disagreement_mean": float(np.mean(
                [d2[f"dis_bag_{key}"][0] for d2 in data])),
        })
    logp("Q3(b)  depth | tree b2/var/acc | bagged b2/var/acc | var removed")
    for r in rows:
        logp(f"  {str(r['max_depth']):>5s} ({r['role']:<8s}) | "
             f"{r['tree_bias2']:.4f}/{r['tree_var']:.4f}/"
             f"{r['tree_acc']:.4f} | {r['bag_bias2']:.4f}/"
             f"{r['bag_var']:.4f}/{r['bag_acc']:.4f} | "
             f"-{r['var_removed_frac'] * 100:.0f}%")
    save_json({"rows": rows}, RESULTS / "q3b_depth_study.json")

    # ---- Q3(c): is the deepest tree the best depth to bag? ----------------
    # bagged accuracy on the ORIGINAL split at the two bounded depths (the
    # unlimited-depth row comes from the Q1 sweep at B = bag_size)
    q1b = load_json(RESULTS / "q1b_bagging_sweep.json")
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    sel_depth = dict(DEPTH_STUDY)["selected"]
    orig_bag = {sel_depth: next(r["test_acc"] for r in q1b["sweep"]
                                if r["B"] == BAG_SIZE)}
    for d in [d for _, d in DEPTH_STUDY if d is not None]:
        ens = BaggingEnsemble({"max_depth": d, "min_samples_leaf": 4},
                              BAG_SIZE, seed=SEED)
        ens.fit(Xtr, ytr)          # same bootstrap stream as Q1
        orig_bag[d] = accuracy(yte, ens.predict(Xte))
        logp(f"  (original-split bagging at depth {d}: "
             f"{orig_bag[d]:.4f})")
    improvements = []
    for row in rows:
        d = row["max_depth"]
        single = occ[str(d)]["test_acc"]
        improvements.append({
            "max_depth": d, "role": row["role"],
            "single_orig": single, "bagged_orig": orig_bag[d],
            "gain_orig_pp": (orig_bag[d] - single) * 100,
            "gain_resamples_pp": row["acc_gain"] * 100,
            "n_leaves": row["n_leaves"],
            "mean_samples_per_leaf": row["mean_samples_per_leaf"],
            "disagreement_mean": row["disagreement_mean"],
            "tree_var": row["tree_var"],
            "var_removed_frac": row["var_removed_frac"],
        })
    logp("Q3(c)  depth | single -> bagged (original split) | gain")
    for m in improvements:
        logp(f"  {str(m['max_depth']):>5s} | {m['single_orig']:.4f} -> "
             f"{m['bagged_orig']:.4f} | +{m['gain_orig_pp']:.1f} pp "
             f"(+{m['gain_resamples_pp']:.1f} pp over the refits)")
    save_json({"measurements": improvements},
              RESULTS / "q3c_deepest_analysis.json")

    # ---- Q3(d): final table ----------------------------------------------
    q1d = load_json(RESULTS / "q1d_clean_masked.json")
    q2d = load_json(RESULTS / "q2d_final_comparison.json")
    final = {"single_tree": {
                 "clean_acc": q1d["clean"]["tree_acc"],
                 "masked_acc": q1d["masked"]["tree_acc"],
                 "bias2": stats[f"tree_{sel_key}"]["bias2"],
                 "variance": stats[f"tree_{sel_key}"]["variance"]},
             "bagging": {
                 "clean_acc": q1d["clean"]["bagging_acc"],
                 "masked_acc": q1d["masked"]["bagging_acc"],
                 "bias2": stats[f"bag_{sel_key}"]["bias2"],
                 "variance": stats[f"bag_{sel_key}"]["variance"]},
             "boosting": {
                 "clean_acc": q2d["boosting"]["clean_acc"],
                 "masked_acc": q2d["boosting"]["masked_acc"],
                 "bias2": stats["boost"]["bias2"],
                 "variance": stats["boost"]["variance"]}}
    logp("Q3(d)  model | clean | masked | bias^2 | variance")
    for k, v in final.items():
        logp(f"  {k:<12s} | {v['clean_acc']:.4f} | {v['masked_acc']:.4f} | "
             f"{v['bias2']:.4f} | {v['variance']:.4f}")
    save_json(final, RESULTS / "q3d_final_table.json")
    return stats


def make_figure(stats, logp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # plot order: shallow, deeper bounded, selected (unlimited) - the
    # selected unlimited depth is plotted last, never "to the left" of a
    # bounded depth
    order = ["shallow", "deeper_bounded", "selected"]
    depths_by_role = dict(DEPTH_STUDY)
    xs = np.arange(len(order))
    labels = [str(depths_by_role[role]) for role in order]
    keys = [depth_key(depths_by_role[role]) for role in order]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    for ax, metric, title in zip(
            axes, ("bias2", "variance", "acc_mean"),
            ("bias$^2$", "variance", "test accuracy")):
        ax.plot(xs, [stats[f"tree_{k}"][metric] for k in keys], "o-",
                ms=4, label="single tree")
        ax.plot(xs, [stats[f"bag_{k}"][metric] for k in keys], "s-",
                ms=4, label="bagged 100")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("max depth")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("measured value")
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(REPO / "figures" / "fig_q3_depth.png", dpi=200)
    fig.savefig(REPO / "figures" / "fig_q3_depth.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    logp = log("q3_bias_variance")
    logp(f"Q3 | seed {SEED}")
    X, y = load_faces()
    split = load_split()

    q1a = load_json(RESULTS / "q1a_tree_grid.json")
    q2b = load_json(RESULTS / "q2b_boosting_grid.json")
    frozen_msl = q1a["frozen_params"]["min_samples_leaf"]
    boost_depth = q2b["selected"]["depth"]
    boost_rounds = q2b["selected"]["rounds"]
    final_B = load_json(RESULTS / "q1b_bagging_sweep.json")["final_B"]
    assert final_B == BAG_SIZE, \
        "the depth-study bagged ensemble must equal the final bagged model"
    logp(f"frozen: min_samples_leaf={frozen_msl}, B*={final_B}, "
         f"boosting depth {boost_depth} x {boost_rounds} rounds | "
         f"depth study: "
         + ", ".join(f"{role}={d}" for role, d in DEPTH_STUDY)
         + " (selected depth is unlimited, so no deeper configuration "
           "exists; 16 is the deeper bounded comparison)")

    CKPT.mkdir(parents=True, exist_ok=True)
    resamples = make_resamples(split, y)
    pending = [r for r in range(N_REPS)
               if not (CKPT / f"q3_rep{r}.npz").exists()]
    logp(f"repetitions to run: {len(pending)}/{N_REPS}")
    t0 = time.perf_counter()
    while pending:
        batch = pending[:2]
        res = Parallel(n_jobs=len(batch))(
            delayed(rep_worker)(r, resamples, X, y, split, frozen_msl,
                                boost_depth, boost_rounds) for r in batch)
        for r in res:
            logp(f"  rep {r['rep']} done "
                 f"({time.perf_counter() - t0:.0f}s elapsed, boost rounds "
                 f"{r['boost_rounds_run']}, {r['stop_reason']})")
            pending.remove(r["rep"])

    stats = aggregate(X, y, split, logp)
    make_figure(stats, logp)
    logp("Q3 done.")


if __name__ == "__main__":
    main()
