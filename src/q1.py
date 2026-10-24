"""
q1.py - Q1: decision tree baseline and bagging from scratch.

(a) DecisionTreeClassifier grid over max_depth x min_samples_leaf. One
    setting is selected on validation accuracy only (ties: smaller depth
    first, then larger min_samples_leaf) and frozen for everything else.
(b) From-scratch bagging, B in {1,5,10,25,50,100,200}. The size-B
    ensemble is the first B trees of one seeded 200-tree run, so every
    prefix is itself a valid size-B bagged ensemble from the same RNG
    stream (one call: default_rng(260004).integers(0, 240, (200, 240))).
(c) Per B: out-of-bag accuracy (training set, coverage reported), mean
    pairwise tree disagreement and mean vote margin (clean test set).
    Final ensemble size B* is selected on validation accuracy only.
(d) Frozen single tree vs final bagged model on the clean and the masked
    test sets (accuracy + mean prediction confidence), plus two faces
    where bagging corrects the tree and two where both are wrong.

Also draws the Q1 figures into figures/.

Run:  python src/q1.py
"""

import numpy as np

from bagging import BaggingEnsemble, make_tree
from data import (REPO, SEED, accuracy, log, load_faces, load_split,
                  mask_images, pairwise_disagreement, save_json,
                  tree_leaf_stats, vote_fractions, vote_margins)

FIG = REPO / "figures"
RESULTS = REPO / "results"

DEPTH_RANK = {2: 2, 4: 4, 8: 8, 16: 16, None: float("inf")}


# ---------------------------------------------------------------------------
def part_a(X, y, split, logp):
    logp("=" * 70)
    logp("Q1(a)  decision tree grid, validation-only selection")
    logp("=" * 70)
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]
    Xte, yte = X[split["test"]], y[split["test"]]

    rows = []
    for d in [2, 4, 8, 16, None]:
        for msl in [1, 2, 4]:
            params = {"max_depth": d, "min_samples_leaf": msl}
            model = make_tree(params).fit(Xtr, ytr)
            row = {"max_depth": d, "min_samples_leaf": msl,
                   "train_acc": accuracy(ytr, model.predict(Xtr)),
                   "val_acc": accuracy(yva, model.predict(Xva)),
                   "test_acc": accuracy(yte, model.predict(Xte)),
                   **tree_leaf_stats(model)}
            rows.append(row)
            logp(f"  depth={str(d):>4s} msl={msl} | train {row['train_acc']:.4f}"
                 f"  val {row['val_acc']:.4f}  test {row['test_acc']:.4f} | "
                 f"leaves {row['n_leaves']:>3d}  mean/leaf "
                 f"{row['mean_samples_per_leaf']:6.2f}  1-sample-leaf frac "
                 f"{row['frac_single_example_leaves']:.3f}")

    # highest validation accuracy; ties -> smaller depth, then larger msl
    best = sorted(rows, key=lambda r: (-r["val_acc"], DEPTH_RANK[r["max_depth"]],
                                       -r["min_samples_leaf"]))[0]
    frozen = {"max_depth": best["max_depth"],
              "min_samples_leaf": best["min_samples_leaf"]}
    logp(f"  FROZEN: max_depth={frozen['max_depth']}, "
         f"min_samples_leaf={frozen['min_samples_leaf']} "
         f"(val {best['val_acc']:.4f})")
    out = {"grid": rows, "selected": best, "frozen_params": frozen}
    save_json(out, RESULTS / "q1a_tree_grid.json")
    return out


# ---------------------------------------------------------------------------
def parts_b_c(X, y, split, frozen, logp):
    logp("=" * 70)
    logp("Q1(b)+Q1(c)  bagging from scratch, B sweep with OOB diagnostics")
    logp("=" * 70)
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    n = Xtr.shape[0]

    logp("  fitting the seeded 200-tree ensemble "
         "(default_rng(260004).integers(0, 240, (200, 240))) ...")
    ens = BaggingEnsemble(frozen, 200, seed=SEED)
    ens.fit(Xtr, ytr)
    logp(f"  done in {ens.fit_seconds:.1f}s")

    preds_tr = ens.tree_predictions(Xtr)
    preds_va = ens.tree_predictions(Xva)
    preds_te = ens.tree_predictions(Xte)
    theory = n * (1.0 - (1.0 - 1.0 / n) ** n)

    def vote_acc(preds, yy):
        return accuracy(yy, np.argmax(vote_fractions(preds), axis=1))

    sweep, diag = [], []
    for B in [1, 5, 10, 25, 50, 100, 200]:
        pt, pv, pte = preds_tr[:B], preds_va[:B], preds_te[:B]
        oob_acc, oob_cov = ens.oob_accuracy(ytr, B)
        row = {"B": B,
               "train_acc": vote_acc(pt, ytr),
               "val_acc": vote_acc(pv, yva),
               "test_acc": vote_acc(pte, yte),
               "avg_distinct": float(ens.distinct_bootstrap_examples(B).mean()),
               "theory_distinct": float(theory),
               "train_seconds": ens.cum_seconds[B - 1]}
        sweep.append(row)
        d = {"B": B,
             "oob_accuracy": oob_acc,
             "oob_coverage": oob_cov,
             "disagreement": pairwise_disagreement(pte),
             "margin": float(np.mean(vote_margins(vote_fractions(pte)))),
             "val_acc": row["val_acc"]}
        diag.append(d)
        logp(f"  B={B:>3d} | train {row['train_acc']:.4f}  val "
             f"{row['val_acc']:.4f}  test {row['test_acc']:.4f} | distinct "
             f"{row['avg_distinct']:.2f} (theory {theory:.2f}) | OOB "
             f"{d['oob_accuracy']:.4f} (cov {d['oob_coverage']:.3f})  dis "
             f"{d['disagreement']}  margin {d['margin']:.4f}")

    # final size: highest validation accuracy, ties -> smallest B
    final_B = sorted(sweep, key=lambda r: (-r["val_acc"], r["B"]))[0]["B"]
    logp(f"  FINAL B* = {final_B} (highest validation accuracy, ties -> "
         f"smallest B)")
    save_json({"B_values": [1, 5, 10, 25, 50, 100, 200], "sweep": sweep,
               "final_B": final_B, "theory_distinct": float(theory)},
              RESULTS / "q1b_bagging_sweep.json")
    save_json({"diagnostics": diag, "final_B": final_B},
              RESULTS / "q1c_diagnostics.json")
    return {"sweep": sweep, "final_B": final_B}, {"diag": diag}, ens


# ---------------------------------------------------------------------------
def part_d(X, y, split, frozen, final_B, ens, logp):
    logp("=" * 70)
    logp(f"Q1(d)  frozen tree vs bagging (B*={final_B}) on clean/masked test")
    logp("=" * 70)
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    Xte_m = mask_images(Xte)

    tree = make_tree(frozen).fit(Xtr, ytr)
    t_pred_c, t_pred_m = tree.predict(Xte), tree.predict(Xte_m)
    t_conf_c = tree.predict_proba(Xte).max(axis=1)
    t_conf_m = tree.predict_proba(Xte_m).max(axis=1)
    b_pred_c, b_pred_m = ens.predict(Xte, final_B), ens.predict(Xte_m, final_B)
    b_conf_c = ens.confidence(Xte, final_B)
    b_conf_m = ens.confidence(Xte_m, final_B)

    clean = {"tree_acc": accuracy(yte, t_pred_c),
             "tree_conf": float(t_conf_c.mean()),
             "bagging_acc": accuracy(yte, b_pred_c),
             "bagging_conf": float(b_conf_c.mean())}
    masked = {"tree_acc": accuracy(yte, t_pred_m),
              "tree_conf": float(t_conf_m.mean()),
              "bagging_acc": accuracy(yte, b_pred_m),
              "bagging_conf": float(b_conf_m.mean())}
    logp(f"  CLEAN : tree {clean['tree_acc']:.4f} (conf {clean['tree_conf']:.3f})"
         f" | bagging {clean['bagging_acc']:.4f} (conf "
         f"{clean['bagging_conf']:.3f})")
    logp(f"  MASKED: tree {masked['tree_acc']:.4f} (conf {masked['tree_conf']:.3f})"
         f" | bagging {masked['bagging_acc']:.4f} (conf "
         f"{masked['bagging_conf']:.3f})")

    # case studies on the clean test set
    votes = ens.vote_fractions(Xte, final_B) * final_B   # vote counts
    corrected = np.flatnonzero((t_pred_c != yte) & (b_pred_c == yte))
    both_wrong = np.flatnonzero((t_pred_c != yte) & (b_pred_c != yte))

    def case(i):
        return {"test_index": int(split["test"][i]),
                "true": int(yte[i]),
                "tree_pred": int(t_pred_c[i]),
                "tree_conf": float(t_conf_c[i]),
                "bagging_pred": int(b_pred_c[i]),
                "votes_true": float(votes[i, yte[i]]),
                "votes_top": {str(k): float(votes[i, k])
                              for k in np.argsort(-votes[i])[:3]}}

    cases = {"bagging_corrects_tree": [case(i) for i in corrected[:2]],
             "both_wrong": [case(i) for i in both_wrong[:2]]}
    for tag, ids in (("corrects", corrected[:2]), ("both wrong", both_wrong[:2])):
        for i in ids:
            c = case(int(i))
            logp(f"  {tag}: img {c['test_index']} true={c['true']} tree -> "
                 f"{c['tree_pred']} (p={c['tree_conf']:.2f}) | bagging -> "
                 f"{c['bagging_pred']}  votes true "
                 f"{c['votes_true']:.0f}/{final_B}")

    save_json({"clean": clean, "masked": masked,
               "case_counts": {"bagging_corrects_tree": int(corrected.size),
                               "both_wrong": int(both_wrong.size)},
               "cases": cases,
               "per_example": {
                   "test_indices": [int(v) for v in split["test"]],
                   "true": [int(v) for v in yte],
                   "tree_pred_clean": [int(v) for v in t_pred_c],
                   "tree_pred_masked": [int(v) for v in t_pred_m],
                   "bagging_pred_clean": [int(v) for v in b_pred_c],
                   "bagging_pred_masked": [int(v) for v in b_pred_m]}},
              RESULTS / "q1d_clean_masked.json")
    return {"clean": clean, "masked": masked, "cases": cases,
            "counts": (corrected, both_wrong), "votes": votes}


# ---------------------------------------------------------------------------
def make_figures(X, y, split, b, c, d, final_B, logp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    logp("  drawing Q1 figures ...")

    # --- dataset / protocol figure -----------------------------------------
    fig = plt.figure(figsize=(7.0, 2.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 10, height_ratios=[1.0, 1.15])
    id0 = np.flatnonzero(y == 0)
    # colour each of identity 0's images by its role in the saved split
    role = {}
    for name, col in (("train", "#2e7d32"), ("val", "#1565c0"),
                      ("test", "#c62828")):
        for g in split[name]:
            role[int(g)] = col
    for j, g in enumerate(id0):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(X[g].reshape(64, 64), cmap="gray")
        ax.set_title(str(int(g)), fontsize=5, color=role[int(g)])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(role[int(g)])
            s.set_linewidth(1.6)
    # bottom: a test face of identity 0, clean vs masked
    test0 = [g for g in split["test"] if y[g] == 0][0]
    axc = fig.add_subplot(gs[1, 0:5])
    axc.imshow(X[test0].reshape(64, 64), cmap="gray")
    axc.add_patch(plt.Rectangle((23.5, 23.5), 17, 17, fill=False,
                                ec="red", lw=1.2))
    axc.set_title(f"test image {test0}, clean (red box = masked block)",
                  fontsize=7)
    axm = fig.add_subplot(gs[1, 5:10])
    axm.imshow(mask_images(X[[test0]])[0].reshape(64, 64), cmap="gray")
    axm.set_title("same image with the centred 16x16 block zeroed",
                  fontsize=7)
    for ax in (axc, axm):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Olivetti identity 0 with the saved split "
                 "(green train / blue val / red test)", fontsize=8)
    fig.savefig(FIG / "fig_dataset.png", dpi=200)
    fig.savefig(FIG / "fig_dataset.pdf")
    plt.close(fig)

    # --- B sweep -------------------------------------------------------------
    Bs = [r["B"] for r in b["sweep"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    for key, lab in (("train_acc", "train"), ("val_acc", "validation"),
                     ("test_acc", "test")):
        ax.plot(Bs, [r[key] for r in b["sweep"]], "o-", ms=3, label=lab)
    ax.axvline(final_B, ls=":", c="gray")
    ax.annotate("B*", (final_B, 0.02), fontsize=7, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("ensemble size B")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(FIG / "fig_q1_sweep.png", dpi=200)
    fig.savefig(FIG / "fig_q1_sweep.pdf")
    plt.close(fig)

    # --- diagnostics ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), constrained_layout=True)
    dd = c["diag"]
    axes[0].plot(Bs, [r["oob_accuracy"] for r in dd], "o-", ms=3,
                 label="OOB")
    axes[0].plot(Bs, [r["val_acc"] for r in dd], "s--", ms=3,
                 label="validation")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].plot(Bs, [r["disagreement"] for r in dd], "o-", ms=3)
    axes[1].set_ylabel("pairwise disagreement")
    axes[2].plot(Bs, [r["margin"] for r in dd], "o-", ms=3)
    axes[2].set_ylabel("mean vote margin")
    for ax in axes:
        ax.set_xlabel("ensemble size B")
        ax.set_xscale("log")
    fig.savefig(FIG / "fig_q1_diag.png", dpi=200)
    fig.savefig(FIG / "fig_q1_diag.pdf")
    plt.close(fig)

    # --- case studies ---------------------------------------------------------
    cases = d["cases"]["bagging_corrects_tree"] + d["cases"]["both_wrong"]
    tags = ["corrected", "corrected", "both wrong", "both wrong"]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 2.55),
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.1, 1]})
    votes = d["votes"]
    for j, (cs, tag) in enumerate(zip(cases, tags)):
        ax = axes[0, j]
        ax.imshow(X[cs["test_index"]].reshape(64, 64), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({chr(97 + j)}) {tag}", fontsize=8)
        txt = axes[1, j]
        txt.axis("off")
        vt = cs["votes_true"]
        top_k = max(cs["votes_top"], key=cs["votes_top"].get)
        txt.text(0, 0.78, f"true {cs['true']} | tree -> {cs['tree_pred']}"
                          f" (p={cs['tree_conf']:.2f})", fontsize=6.3)
        txt.text(0, 0.55, f"bagging -> {cs['bagging_pred']}", fontsize=6.3)
        txt.text(0, 0.32, f"votes true {vt:.0f}/{final_B} | top: "
                          f"id {top_k} {cs['votes_top'][top_k]:.0f}",
                 fontsize=6.3)
    fig.savefig(FIG / "fig_q1_cases.png", dpi=200)
    fig.savefig(FIG / "fig_q1_cases.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    logp = log("q1_tree_bagging")
    logp(f"Q1 | seed {SEED}")
    X, y = load_faces()
    split = load_split()

    a = part_a(X, y, split, logp)
    b, c, ens = parts_b_c(X, y, split, a["frozen_params"], logp)
    d = part_d(X, y, split, a["frozen_params"], b["final_B"], ens, logp)
    make_figures(X, y, split, b, c, d, b["final_B"], logp)
    logp("Q1 done.")


if __name__ == "__main__":
    main()
