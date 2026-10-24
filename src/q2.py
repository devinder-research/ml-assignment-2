"""
q2.py - Q2: multiclass SAMME boosting from scratch.

(a) One complete SAMME round on a 3-class, 9-sample toy: initial weights,
    weak-learner predictions, weighted error, alpha_t, updated weights.
    The face-data booster uses exactly the same update rule (boosting.py).
(b) Weak-learner depths {1,2,3,4} x rounds {10,25,50,100}. The ensemble of
    t rounds is the first t learners of one seeded 100-round run. Selection
    on clean validation accuracy only (ties: fewer rounds, then shallower
    depth). For the selected model: weighted training error, learner
    weight, maximum sample weight and the weight of the 10 heaviest
    training images, per round.
(c) Boosting vs bagging with 25/50/100 learners (accuracy + training
    time), plus one heavily weighted training face and its prediction
    history.
(d) Frozen boosting vs the Q1 tree and bagging on clean/masked test sets,
    and one face on which the two ensembles react differently to masking.

Run:  python src/q2.py     (needs Q1 results first: python src/q1.py)
"""

import numpy as np

from bagging import BaggingEnsemble
from boosting import SammeBoost, samme_alpha
from data import (REPO, SEED, accuracy, log, load_faces, load_json,
                  load_split, mask_images, save_json)

FIG = REPO / "figures"
RESULTS = REPO / "results"


# ---------------------------------------------------------------------------
def part_a(logp):
    logp("=" * 70)
    logp("Q2(a)  one complete SAMME round on a 3-class / 9-sample toy")
    logp("=" * 70)
    x = np.array([[1.0], [2.0], [12.0], [5.0], [6.0], [7.0],
                  [3.0], [14.0], [15.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    boost = SammeBoost({"max_depth": 1, "min_samples_leaf": 1},
                       n_rounds=1, n_classes=3, seed=SEED).fit(x, y)

    eps = boost.train_error[0]
    alpha = boost.alphas[0]
    w0, w1 = boost.initial_weights, boost.weight_history[0]
    preds = boost.train_predictions[0]
    order = np.argsort(x[:, 0])
    logp(f"  {'i':>2s} {'x':>4s} {'y':>2s} {'w_init':>9s} {'pred':>4s} "
         f"{'ok':>5s} {'w_updated':>9s}")
    for i in order:
        logp(f"  {i:>2d} {x[i, 0]:>4.0f} {y[i]:>2d} {w0[i]:>9.6f} "
             f"{preds[i]:>4d} {str(preds[i] == y[i]):>5s} {w1[i]:>9.6f}")
    stump = boost.trees[0].tree_
    logp(f"  stump: split x <= {stump.threshold[0]:.1f}  "
         f"eps_1 = {eps:.6f}  alpha_1 = {alpha:.6f}  "
         f"(check: log((1-eps)/eps)+log(2) = {samme_alpha(eps, 3):.6f})")
    logp(f"  misclassified mass after update: {w1[preds != y].sum():.6f}, "
         f"correct mass {w1[preds == y].sum():.6f}")

    save_json({"x": x.ravel().tolist(), "y": y.tolist(),
               "initial_weights": w0.tolist(),
               "predictions": preds.tolist(),
               "weighted_error": eps, "alpha": alpha,
               "updated_weights": w1.tolist(),
               "threshold": float(stump.threshold[0])},
              RESULTS / "q2a_synth_round.json")
    # the same table goes to a plain-text log for the report
    lines = ["Q2(a) one complete SAMME round (3 classes, 9 samples, stump)",
             f"eps_1 = {eps:.6f}, alpha_1 = {alpha:.6f}",
             f"updated weights (sorted by x): "
             f"{[round(w1[i], 6) for i in order]}"]
    (REPO / "logs" / "q2a_synth_round.txt").write_text("\n".join(lines))
    return {"eps": eps, "alpha": alpha}


# ---------------------------------------------------------------------------
def prefix_metrics(boost, t, sets):
    """Accuracy + mean winning q for the first-t prefix on each set."""
    out = {}
    for name, (Xa, ya) in sets.items():
        S = boost.class_scores(Xa, t)
        q = S / float(np.sum(boost.alphas[:t]))
        out[f"{name}_acc"] = accuracy(ya, S.argmax(axis=1))
        out[f"{name}_mean_q"] = float(q.max(axis=1).mean())
    out["train_seconds"] = boost.cum_seconds[t - 1]
    return out


def part_b(X, y, split, logp):
    logp("=" * 70)
    logp("Q2(b)  boosting grid (depth x rounds), validation-only selection")
    logp("=" * 70)
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    sets = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}

    grid, models = [], {}
    for d in [1, 2, 3, 4]:
        boost = SammeBoost({"max_depth": d, "min_samples_leaf": 1},
                           n_rounds=100, seed=SEED).fit(Xtr, ytr)
        models[d] = boost
        logp(f"  depth {d}: stop reason '{boost.stop_reason}', "
             f"{boost.n_rounds_run} learners kept, "
             f"eps range [{min(boost.train_error):.3f}, "
             f"{max(boost.train_error):.3f}]")
        for t in [10, 25, 50, 100]:
            row = {"depth": d, "rounds": t}
            row.update(prefix_metrics(boost, t, sets))
            grid.append(row)
            logp(f"    t={t:>3d} | train {row['train_acc']:.4f}  val "
                 f"{row['val_acc']:.4f}  test {row['test_acc']:.4f} "
                 f"({row['train_seconds']:.1f}s)")

    # ties -> fewer rounds, then shallower depth
    sel = sorted(grid, key=lambda r: (-r["val_acc"], r["rounds"],
                                      r["depth"]))[0]
    logp(f"  SELECTED: depth {sel['depth']}, {sel['rounds']} rounds "
         f"(val {sel['val_acc']:.4f}, test {sel['test_acc']:.4f})")

    sel_boost = models[sel["depth"]]
    hist = sel_boost
    save_json({"grid": grid, "selected": sel,
               "stop_reasons": {str(d): models[d].stop_reason
                                for d in [1, 2, 3, 4]}},
              RESULTS / "q2b_boosting_grid.json")
    save_json({"selected": sel, "eps": hist.train_error,
               "alpha": hist.alphas, "max_weight": hist.max_weight,
               "top10_share": hist.top10_share,
               "final_weights": hist.weight_history[-1].tolist()},
              RESULTS / "q2b_selected_dynamics.json")
    return grid, sel, sel_boost


# ---------------------------------------------------------------------------
def part_c(X, y, split, sel, sel_boost, logp):
    logp("=" * 70)
    logp("Q2(c)  boosting vs bagging (25/50/100 learners) + heavy face")
    logp("=" * 70)
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    sets = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
    bag_rows = {r["B"]: r for r in
                load_json(RESULTS / "q1b_bagging_sweep.json")["sweep"]}

    compare = []
    for L in [25, 50, 100]:
        b = prefix_metrics(sel_boost, L, sets)
        row = {"learners": L,
               "boost_train_acc": b["train_acc"],
               "boost_val_acc": b["val_acc"],
               "boost_test_acc": b["test_acc"],
               "boost_seconds": b["train_seconds"],
               "bag_train_acc": bag_rows[L]["train_acc"],
               "bag_val_acc": bag_rows[L]["val_acc"],
               "bag_test_acc": bag_rows[L]["test_acc"],
               "bag_seconds": bag_rows[L]["train_seconds"]}
        compare.append(row)
        logp(f"  L={L:>3d} | boost train {row['boost_train_acc']:.4f} val "
             f"{row['boost_val_acc']:.4f} test {row['boost_test_acc']:.4f} "
             f"({row['boost_seconds']:.1f}s) | bagging test "
             f"{row['bag_test_acc']:.4f} ({row['bag_seconds']:.1f}s)")

    # heaviest training face of the full 100-round selected-depth run
    w_final = sel_boost.weight_history[-1]
    pos = int(np.argmax(w_final))
    pred_hist = np.stack(sel_boost.train_predictions)      # (100, 240)
    correct = pred_hist[:, pos] == ytr[pos]
    heavy = {
        "global_index": int(split["train"][pos]),
        "identity": int(y[split["train"][pos]]),
        "final_weight": float(w_final[pos]),
        "initial_weight": float(1.0 / len(ytr)),
        "rounds_correct": int(correct.sum()),
        "rounds_total": int(len(correct)),
        "weight_trajectory": [float(w[pos]) for w in
                              sel_boost.weight_history],
        "same_identity_train": [int(i) for i in split["train"]
                                if y[i] == int(y[split["train"][pos]])],
    }
    logp(f"  heaviest face: image {heavy['global_index']} (identity "
         f"{heavy['identity']}), final weight {heavy['final_weight']:.4f} "
         f"= {heavy['final_weight'] / heavy['initial_weight']:.1f}x initial, "
         f"correct in {heavy['rounds_correct']}/{heavy['rounds_total']} "
         f"rounds")
    save_json({"comparison": compare, "heavy": heavy},
              RESULTS / "q2c_bag_vs_boost.json")
    return compare, heavy


# ---------------------------------------------------------------------------
def part_d(X, y, split, sel, sel_boost, logp):
    logp("=" * 70)
    logp("Q2(d)  tree vs bagging vs boosting on clean/masked test sets")
    logp("=" * 70)
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    Xte_m = mask_images(Xte)
    q1d = load_json(RESULTS / "q1d_clean_masked.json")
    T = sel["rounds"]

    # boosting on both test sets (first T learners = the frozen model)
    S_c = sel_boost.class_scores(Xte, T)
    S_m = sel_boost.class_scores(Xte_m, T)
    q_c = S_c / float(np.sum(sel_boost.alphas[:T]))
    q_m = S_m / float(np.sum(sel_boost.alphas[:T]))
    boosting = {"clean_acc": accuracy(yte, S_c.argmax(1)),
                "masked_acc": accuracy(yte, S_m.argmax(1)),
                "clean_conf": float(q_c.max(axis=1).mean()),
                "masked_conf": float(q_m.max(axis=1).mean())}
    tree = {"clean_acc": q1d["clean"]["tree_acc"],
            "masked_acc": q1d["masked"]["tree_acc"],
            "clean_conf": q1d["clean"]["tree_conf"],
            "masked_conf": q1d["masked"]["tree_conf"]}
    bag = {"clean_acc": q1d["clean"]["bagging_acc"],
           "masked_acc": q1d["masked"]["bagging_acc"],
           "clean_conf": q1d["clean"]["bagging_conf"],
           "masked_conf": q1d["masked"]["bagging_conf"]}
    logp(f"  CLEAN : tree {tree['clean_acc']:.4f} | bagging "
         f"{bag['clean_acc']:.4f} | boosting {boosting['clean_acc']:.4f}")
    logp(f"  MASKED: tree {tree['masked_acc']:.4f} | bagging "
         f"{bag['masked_acc']:.4f} | boosting {boosting['masked_acc']:.4f}")

    # the final bagged ensemble, refitted with the same seeded stream
    # (first 100 bootstrap rows of the same Q1 stream -> identical model)
    logp("  refitting the 100-tree bagged ensemble for the case study ...")
    ens = BaggingEnsemble(load_json(RESULTS / "q1a_tree_grid.json")
                          ["frozen_params"], 100, seed=SEED)
    ens.fit(Xtr, ytr)
    B = 100

    bag_c = np.array(q1d["per_example"]["bagging_pred_clean"])
    bag_m = np.array(q1d["per_example"]["bagging_pred_masked"])
    boost_c, boost_m = S_c.argmax(1), S_m.argmax(1)

    both_ok = (bag_c == yte) & (boost_c == yte)
    bag_stays = both_ok & (bag_m == yte) & (boost_m != yte)
    boost_stays = both_ok & (boost_m == yte) & (bag_m != yte)
    if bag_stays.any():
        pos, kind = int(np.flatnonzero(bag_stays)[0]), \
            "bagging_stays_correct_boosting_flips"
    elif boost_stays.any():
        pos, kind = int(np.flatnonzero(boost_stays)[0]), \
            "boosting_stays_correct_bagging_flips"
    else:
        diff = ((bag_m == yte) & (boost_m != yte)) | \
               ((boost_m == yte) & (bag_m != yte))
        pos, kind = int(np.flatnonzero(diff)[0]), "differing_masked_outcome"
    true_k = int(yte[pos])
    votes_c = ens.vote_fractions(Xte, B)[pos] * B
    votes_m = ens.vote_fractions(Xte_m, B)[pos] * B
    tp_c = ens.tree_predictions(Xte, B)[:, pos]
    tp_m = ens.tree_predictions(Xte_m, B)[:, pos]

    # how often do the ensembles split on pixels inside the masked block?
    def frac_splits_in_block(trees):
        fracs = []
        for tr in trees:
            f = tr.tree_.feature
            f = f[f >= 0]
            r, c = f // 64, f % 64
            fracs.append(float(((r >= 24) & (r <= 39) &
                                (c >= 24) & (c <= 39)).mean()))
        return float(np.mean(fracs))

    case = {
        "test_position": pos, "test_index": int(split["test"][pos]),
        "true": true_k, "type": kind,
        "tree_pred_clean": int(q1d["per_example"]["tree_pred_clean"][pos]),
        "tree_pred_masked": int(q1d["per_example"]["tree_pred_masked"][pos]),
        "bagging_pred_clean": int(bag_c[pos]),
        "bagging_pred_masked": int(bag_m[pos]),
        "boosting_pred_clean": int(boost_c[pos]),
        "boosting_pred_masked": int(boost_m[pos]),
        "bagging_votes_true_clean": float(votes_c[true_k]),
        "bagging_votes_true_masked": float(votes_m[true_k]),
        "bagging_disagreement_this_face_clean":
            float(np.mean(tp_c[:, None] != tp_c[None, :])),
        "bagging_disagreement_this_face_masked":
            float(np.mean(tp_m[:, None] != tp_m[None, :])),
        "boosting_q_true_clean": float(q_c[pos, true_k]),
        "boosting_q_true_masked": float(q_m[pos, true_k]),
        "bagging_frac_splits_in_block":
            frac_splits_in_block(ens.trees[:B]),
        "boosting_frac_splits_in_block":
            frac_splits_in_block(sel_boost.trees[:T]),
    }
    logp(f"  case: test image {case['test_index']} (true {true_k}) "
         f"[{kind}]")
    logp(f"    bagging votes for true class: {votes_c[true_k]:.0f}/{B} -> "
         f"{votes_m[true_k]:.0f}/{B}   boosting q_true: "
         f"{q_c[pos, true_k]:.3f} -> {q_m[pos, true_k]:.3f}")
    logp(f"    splits inside the masked block: bagging "
         f"{case['bagging_frac_splits_in_block']:.3f}, boosting "
         f"{case['boosting_frac_splits_in_block']:.3f}")

    save_json({"tree": tree, "bagging": bag, "boosting": boosting,
               "case_study": case, "frozen_boosting": sel},
              RESULTS / "q2d_final_comparison.json")
    return {"tree": tree, "bagging": bag, "boosting": boosting,
            "case": case}


# ---------------------------------------------------------------------------
def make_figures(X, y, split, sel, sel_boost, heavy, logp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logp("  drawing Q2 figures ...")
    rounds = np.arange(1, len(sel_boost.alphas) + 1)

    # --- training dynamics of the selected configuration ---------------------
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), constrained_layout=True)
    axes[0].plot(rounds, sel_boost.train_error, lw=1)
    axes[0].set_ylabel("weighted train error")
    axes[1].plot(rounds, sel_boost.alphas, lw=1)
    axes[1].set_ylabel("learner weight alpha")
    axes[2].plot(rounds, sel_boost.max_weight, lw=1)
    axes[2].set_ylabel("max sample weight")
    axes[3].plot(rounds, sel_boost.top10_share, lw=1)
    axes[3].set_ylabel("weight of 10 heaviest")
    for ax in axes:
        ax.axvline(sel["rounds"], ls=":", c="gray", lw=1)
        ax.set_xlabel("round t")
    fig.savefig(FIG / "fig_q2_dynamics.png", dpi=200)
    fig.savefig(FIG / "fig_q2_dynamics.pdf")
    plt.close(fig)

    # --- the heavily weighted face ------------------------------------------
    g = heavy["global_index"]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8), constrained_layout=True)
    axes[0].imshow(X[g].reshape(64, 64), cmap="gray")
    axes[0].set_title(f"image {g} (id {heavy['identity']})", fontsize=6.5)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].semilogy(rounds, heavy["weight_trajectory"], lw=1)
    axes[1].set_xlabel("round t")
    axes[1].set_ylabel("sample weight")
    axes[1].set_title(f"correct in {heavy['rounds_correct']}/"
                      f"{heavy['rounds_total']} rounds", fontsize=7)
    fig.savefig(FIG / "fig_q2_heavy.png", dpi=200)
    fig.savefig(FIG / "fig_q2_heavy.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    logp = log("q2_boosting")
    logp(f"Q2 | seed {SEED}")
    X, y = load_faces()
    split = load_split()

    part_a(logp)
    grid, sel, sel_boost = part_b(X, y, split, logp)
    compare, heavy = part_c(X, y, split, sel, sel_boost, logp)
    part_d(X, y, split, sel, sel_boost, logp)
    make_figures(X, y, split, sel, sel_boost, heavy, logp)
    logp("Q2 done.")


if __name__ == "__main__":
    main()
