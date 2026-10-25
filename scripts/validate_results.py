#!/usr/bin/env python3
"""
validate_results.py - independently recalculate the reported results and
compare them against the saved outputs in results/.

For every check the quantities are RECOMPUTED from the saved split and the
seeded procedures (never copied from the JSON files) and then compared to
the saved values:

  1. tree_grid          Q1(a): the full 5x3 decision-tree grid + selection
  2. bagging_sweep      Q1(b): accuracy and distinct-bootstrap statistics per B
  3. oob_diagnostics    Q1(c): OOB accuracy, coverage, disagreement, margin
  4. samme_toy          Q2(a): the one logged SAMME round on the 3-class toy
  5. boosting_selection Q2(b): the depth x rounds grid and the selected model
  6. clean_masked       Q1(d)+Q2(d): clean/masked accuracy + confidence for
                        tree, bagging and boosting, case studies included
  7. q3_statistics      Q3(a)-(d): bias2/variance/accuracy recomputed from
                        the 30 per-repetition checkpoints, the original-split
                        bagging refits, and the depth-study tables
  8. refit_determinism  fitting the same object twice == fitting a fresh
                        object (SammeBoost and BaggingEnsemble)
  9. input_validation   empty datasets / mismatched dimensions / invalid
                        labels / invalid ensemble parameters / invalid
                        bootstrap indices / predictions before fitting all
                        raise clear exceptions

Wall-clock training times are intentionally NOT compared (they differ
between runs); every measured quantity is. Each check prints PASS or FAIL;
the script exits with code 1 when any essential check fails.

Run:  python scripts/validate_results.py
(needs the saved split, results/*.json and results/checkpoints/)
Optional: pass check names to run a subset, e.g.
  python scripts/validate_results.py tree_grid bagging_sweep
check names: input_validation, tree_grid, bagging_sweep, oob_diagnostics,
samme_toy, boosting_selection, clean_masked, q3_statistics,
refit_determinism. With no arguments every check runs.
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bagging import BaggingEnsemble, make_tree          # noqa: E402
from boosting import SammeBoost, samme_alpha            # noqa: E402
from data import (CFG, SEED, accuracy, load_faces, load_json,           # noqa: E402
                  load_split, mask_images, one_hot, pairwise_disagreement,
                  tree_leaf_stats, vote_fractions, vote_margins)

RESULTS = REPO / "results"
CKPT = RESULTS / "checkpoints"
TOL = 1e-9          # recomputation should be exact; allow a last-ulp slack

_checks = []


def _close(a, b, tol=TOL):
    return bool(np.allclose(np.asarray(a, dtype=float),
                            np.asarray(b, dtype=float),
                            rtol=tol, atol=1e-12, equal_nan=True))


def check(name, ok, detail=""):
    _checks.append((name, bool(ok)))
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)
    return bool(ok)


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72, flush=True)


# ---------------------------------------------------------------------------
def check_input_validation(X, y, split):
    section("CHECK 0  input validation raises clear exceptions")
    Xtr, ytr = X[split["train"]], y[split["train"]]

    def raises(fn, exc=ValueError):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    ok = True
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, 10).fit(
        np.empty((0, 4096)), np.empty((0,))))              # empty dataset
    ok &= raises(lambda: SammeBoost({"max_depth": 2}, 10, n_classes=40).fit(
        np.empty((0, 4096)), np.empty((0,))))              # empty dataset
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, 10).fit(
        Xtr, ytr[:100]))                                   # sample mismatch
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, 10).fit(
        Xtr, ytr + 0.5))                                   # non-integer label
    ok &= raises(lambda: SammeBoost({"max_depth": 2}, 10, n_classes=3).fit(
        Xtr[:20], ytr[:20]))                               # label out of range
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, 0))
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, -5))
    ok &= raises(lambda: SammeBoost({"max_depth": 2}, 0))
    ok &= raises(lambda: SammeBoost({"max_depth": 2}, -3))
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, 10).fit(
        Xtr, ytr, bootstrap_indices=np.zeros((10, 100), dtype=int)))
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, 10).fit(
        Xtr, ytr, bootstrap_indices=np.random.default_rng(0).integers(
            0, 240, size=(10, 239))))                      # wrong shape
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, 10).fit(
        Xtr, ytr,
        bootstrap_indices=np.random.default_rng(0).integers(
            0, 240, size=(10, 240)).astype(float)))        # non-integer type
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, 10).fit(
        Xtr, ytr,
        bootstrap_indices=np.full((10, 240), 240)))        # out of range
    # predictions before fitting must fail clearly, not silently
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 4}, 10).predict(Xtr))
    ok &= raises(lambda: SammeBoost({"max_depth": 2}, 10,
                                    n_classes=40).predict(Xtr))
    # valid bootstrap indices with repeats must be ACCEPTED
    try:
        rng = np.random.default_rng(0)
        BaggingEnsemble({"max_depth": 2}, 5).fit(
            Xtr[:40], ytr[:40],
            bootstrap_indices=rng.integers(0, 40, size=(5, 40)))
        accepted = True
    except Exception:
        accepted = False
    ok &= accepted
    # an out-of-range ensemble prefix must fail clearly
    ok &= raises(lambda: BaggingEnsemble({"max_depth": 2}, 5)
                 .fit(Xtr[:40], ytr[:40]).predict(Xtr[:40], n_estimators=6))
    return check("input_validation: empty/mismatched/invalid inputs, invalid "
                 "ensemble parameters, invalid bootstrap indices and "
                 "predictions before fitting all raise ValueError; valid "
                 "bootstraps with repeated indices are accepted", bool(ok))


# ---------------------------------------------------------------------------
def check_tree_grid(X, y, split, q1a):
    section("CHECK 1  Q1(a) decision-tree grid")
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    ok = True
    for row in q1a["grid"]:
        params = {"max_depth": row["max_depth"],
                  "min_samples_leaf": row["min_samples_leaf"]}
        m = make_tree(params).fit(Xtr, ytr)
        got = {"train_acc": accuracy(ytr, m.predict(Xtr)),
               "val_acc": accuracy(yva, m.predict(Xva)),
               "test_acc": accuracy(yte, m.predict(Xte)),
               **tree_leaf_stats(m)}
        bad = [k for k, v in got.items() if not _close(v, row[k])]
        ok &= not bad
        if bad:
            print(f"  mismatch at depth={row['max_depth']} "
                  f"msl={row['min_samples_leaf']}: {bad}")
    ok = check("tree_grid: all 15 grid rows reproduce exactly", ok)

    # selection with the documented tie-break: smaller depth, then larger msl
    def depth_rank(d):
        return float("inf") if d is None else float(d)

    best = sorted(q1a["grid"],
                  key=lambda r: (-r["val_acc"], depth_rank(r["max_depth"]),
                                 -r["min_samples_leaf"]))[0]
    ok2 = (best["max_depth"] == q1a["selected"]["max_depth"]
           and best["min_samples_leaf"] == q1a["selected"]["min_samples_leaf"]
           and q1a["frozen_params"] == {"max_depth": best["max_depth"],
                                        "min_samples_leaf":
                                        best["min_samples_leaf"]})
    return check("tree_grid: frozen setting follows the documented "
                 "tie-break", ok2)


# ---------------------------------------------------------------------------
def build_bagging_ensemble(X, y, split, q1a):
    """The seeded 200-tree Q1 ensemble (shared by several checks)."""
    Xtr, ytr = X[split["train"]], y[split["train"]]
    ens = BaggingEnsemble(q1a["frozen_params"], CFG["bagging"]["n_max"],
                          seed=SEED)
    ens.fit(Xtr, ytr)
    return ens


def build_boosting_models(X, y, split, depths=None):
    """The Q2(b) boosters, one per requested weak-learner depth."""
    Xtr, ytr = X[split["train"]], y[split["train"]]
    if depths is None:
        depths = CFG["boosting"]["depths"]
    return {d: SammeBoost({"max_depth": d, "min_samples_leaf": 1},
                          n_rounds=CFG["boosting"]["max_rounds"],
                          seed=SEED).fit(Xtr, ytr) for d in depths}


def check_bagging_sweep(X, y, split, q1a, q1b, ens):
    section("CHECK 2  Q1(b) bagging sweep (one seeded 200-tree run)")
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    n = Xtr.shape[0]
    preds = {"train": ens.tree_predictions(Xtr),
             "val": ens.tree_predictions(Xva),
             "test": ens.tree_predictions(Xte)}
    theory = n * (1.0 - (1.0 - 1.0 / n) ** n)

    ok = _close(theory, q1b["theory_distinct"])
    for row in q1b["sweep"]:
        B = row["B"]
        got = {"train_acc": accuracy(ytr, np.argmax(
                   vote_fractions(preds["train"][:B]), axis=1)),
               "val_acc": accuracy(yva, np.argmax(
                   vote_fractions(preds["val"][:B]), axis=1)),
               "test_acc": accuracy(yte, np.argmax(
                   vote_fractions(preds["test"][:B]), axis=1)),
               "avg_distinct": float(
                   ens.distinct_bootstrap_examples(B).mean()),
               "theory_distinct": theory}
        bad = [k for k, v in got.items() if not _close(v, row[k])]
        if bad:
            print(f"  mismatch at B={B}: {bad}")
            ok = False
    ok = check("bagging_sweep: train/val/test accuracy and distinct counts "
               "reproduce for every B", ok)
    final_B = sorted(q1b["sweep"],
                     key=lambda r: (-r["val_acc"], r["B"]))[0]["B"]
    ok = check("bagging_sweep: final B* follows validation-only selection "
               "(ties -> smallest B)", final_B == q1b["final_B"],
               f"B*={q1b['final_B']}")
    return ens


# ---------------------------------------------------------------------------
def check_oob_diagnostics(X, y, split, q1b, q1c, ens):
    section("CHECK 3  Q1(c) OOB accuracy / coverage / disagreement / margin")
    Xte = X[split["test"]]
    ytr = y[split["train"]]
    ok = True
    for d in q1c["diagnostics"]:
        B = d["B"]
        oob_acc, oob_cov = ens.oob_accuracy(ytr, B)
        pte = ens.tree_predictions(Xte, B)
        got = {"oob_accuracy": oob_acc, "oob_coverage": oob_cov,
               "disagreement": pairwise_disagreement(pte),
               "margin": float(np.mean(vote_margins(vote_fractions(pte)))),
               "val_acc": next(r["val_acc"] for r in q1b["sweep"]
                               if r["B"] == B)}
        bad = []
        for k, v in got.items():
            if d[k] is None or v is None:
                if (d[k] is None) != (v is None):
                    bad.append(k)
            elif not _close(v, d[k]):
                bad.append(k)
        if bad:
            print(f"  mismatch at B={B}: {bad}")
            ok = False
    return check("oob_diagnostics: OOB accuracy, coverage, disagreement and "
                 "margin reproduce for every B", ok)


# ---------------------------------------------------------------------------
def check_samme_toy(q2a):
    section("CHECK 4  Q2(a) one complete SAMME round on the 3-class toy")
    x = np.array([[1.0], [2.0], [12.0], [5.0], [6.0], [7.0],
                  [3.0], [14.0], [15.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    boost = SammeBoost({"max_depth": 1, "min_samples_leaf": 1},
                       n_rounds=1, n_classes=3, seed=SEED).fit(x, y)
    got = {"weighted_error": boost.train_error[0],
           "alpha": boost.alphas[0],
           "initial_weights": boost.initial_weights,
           "updated_weights": boost.weight_history[0],
           "predictions": boost.train_predictions[0],
           "threshold": float(boost.trees[0].tree_.threshold[0])}
    bad = [k for k, v in got.items() if not _close(v, q2a[k])]
    ok = check("samme_toy: eps, alpha, weights, predictions and stump "
               "threshold reproduce", not bad,
               ", ".join(bad) if bad else
               f"eps={got['weighted_error']:.4f}, "
               f"alpha={got['alpha']:.4f}")
    ok2 = _close(samme_alpha(got["weighted_error"], 3), got["alpha"])
    return check("samme_toy: alpha equals log((1-eps)/eps) + log(K-1)", ok2)


# ---------------------------------------------------------------------------
def check_boosting_selection(X, y, split, q2b, models):
    section("CHECK 5  Q2(b) boosting grid and selection")
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    sets = {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}
    ok = True
    for row in q2b["grid"]:
        boost = models[row["depth"]]
        t = row["rounds"]
        if t > boost.n_rounds_run:
            # early stopping: numbers saved for a non-existent prefix are
            # an inconsistency
            if any(row.get(k) is not None for k in
                   ("train_acc", "val_acc", "test_acc")):
                print(f"  saved row depth={row['depth']} t={t} has numbers "
                      f"but boosting now stops after {boost.n_rounds_run} "
                      f"rounds")
                ok = False
            continue
        S = {name: boost.class_scores(Xa, t) for name, (Xa, _) in
             sets.items()}
        got = {}
        for name, (_, ya) in sets.items():
            got[f"{name}_acc"] = accuracy(ya, S[name].argmax(axis=1))
            got[f"{name}_mean_q"] = float(
                (S[name] / float(np.sum(boost.alphas[:t])))
                .max(axis=1).mean())
        bad = [k for k, v in got.items() if not _close(v, row[k])]
        if bad:
            print(f"  mismatch at depth={row['depth']} t={t}: {bad}")
            ok = False
    ok = check("boosting_selection: every grid cell reproduces", ok)

    ok = all(models[d].stop_reason == q2b["stop_reasons"][str(d)]
             for d in CFG["boosting"]["depths"])
    check("boosting_selection: stop reasons match the saved ones", ok)

    valid = [r for r in q2b["grid"] if r.get("val_acc") is not None]
    sel = sorted(valid, key=lambda r: (-r["val_acc"], r["rounds"],
                                       r["depth"]))[0]
    ok = (sel["depth"] == q2b["selected"]["depth"]
          and sel["rounds"] == q2b["selected"]["rounds"])
    check("boosting_selection: selected model follows validation-only "
          "selection (ties -> fewer rounds, shallower depth)", ok,
          f"depth {q2b['selected']['depth']} x "
          f"{q2b['selected']['rounds']} rounds")


# ---------------------------------------------------------------------------
def check_clean_masked(X, y, split, q1a, q1b, q1d, q2b, q2d, bag_ens,
                       boost_models):
    section("CHECK 6  Q1(d)+Q2(d) clean/masked evaluation and case studies")
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    Xte_m = mask_images(Xte)
    final_B = q1b["final_B"]
    ok = True

    # single tree
    tree = make_tree(q1a["frozen_params"]).fit(Xtr, ytr)
    t_pred_c, t_pred_m = tree.predict(Xte), tree.predict(Xte_m)
    t_conf_c = tree.predict_proba(Xte).max(axis=1)
    t_conf_m = tree.predict_proba(Xte_m).max(axis=1)
    for tag, acc, conf in (("clean", accuracy(yte, t_pred_c),
                            float(t_conf_c.mean())),
                           ("masked", accuracy(yte, t_pred_m),
                            float(t_conf_m.mean()))):
        if not (_close(acc, q1d[tag]["tree_acc"])
                and _close(conf, q1d[tag]["tree_conf"])):
            print(f"  single-tree mismatch on {tag}")
            ok = False

    # bagging (prefix of the same seeded 200-tree run)
    b_pred_c = bag_ens.predict(Xte, final_B)
    b_pred_m = bag_ens.predict(Xte_m, final_B)
    b_conf_c = bag_ens.confidence(Xte, final_B)
    b_conf_m = bag_ens.confidence(Xte_m, final_B)
    for tag, acc, conf in (("clean", accuracy(yte, b_pred_c),
                            float(b_conf_c.mean())),
                           ("masked", accuracy(yte, b_pred_m),
                            float(b_conf_m.mean()))):
        if not (_close(acc, q1d[tag]["bagging_acc"])
                and _close(conf, q1d[tag]["bagging_conf"])):
            print(f"  bagging mismatch on {tag}")
            ok = False
    ok = check("clean_masked: single tree and bagging accuracy/confidence "
               "reproduce on both test sets", ok)

    # Q1(d) case studies
    votes = bag_ens.vote_fractions(Xte, final_B) * final_B
    corrected = np.flatnonzero((t_pred_c != yte) & (b_pred_c == yte))
    both_wrong = np.flatnonzero((t_pred_c != yte) & (b_pred_c != yte))
    ok = (_close(int(corrected.size),
                 q1d["case_counts"]["bagging_corrects_tree"])
          and _close(int(both_wrong.size),
                     q1d["case_counts"]["both_wrong"]))
    ok = check("clean_masked: case counts reproduce "
               f"({q1d['case_counts']['bagging_corrects_tree']} corrected / "
               f"{q1d['case_counts']['both_wrong']} both wrong)", ok)

    def case(i):
        return {"test_index": int(split["test"][i]),
                "true": int(yte[i]),
                "tree_pred": int(t_pred_c[i]),
                "tree_conf": float(t_conf_c[i]),
                "bagging_pred": int(b_pred_c[i]),
                "votes_true": float(votes[i, yte[i]]),
                "votes_top": {str(k): float(votes[i, k])
                              for k in np.argsort(-votes[i])[:3]}}

    recomputed = {"bagging_corrects_tree": [case(i) for i in corrected[:2]],
                  "both_wrong": [case(i) for i in both_wrong[:2]]}
    ok = True
    for tag in recomputed:
        for got, saved in zip(recomputed[tag], q1d["cases"][tag]):
            for k in ("test_index", "true", "tree_pred", "tree_conf",
                      "bagging_pred", "votes_true"):
                if not _close(got[k], saved[k]):
                    print(f"  case mismatch [{tag}] {k}: "
                          f"{got[k]} vs {saved[k]}")
                    ok = False
            for k, v in saved["votes_top"].items():
                if not _close(got["votes_top"].get(k, 0.0), v):
                    ok = False
    ok = check("clean_masked: the four Q1(d) case studies reproduce "
               "(indices, predictions, vote counts)", ok)

    # boosting (Q2d)
    sel = q2b["selected"]
    sel_boost = boost_models[sel["depth"]]
    T = min(sel["rounds"], sel_boost.n_rounds_run)
    S_c = sel_boost.class_scores(Xte, T)
    S_m = sel_boost.class_scores(Xte_m, T)
    q_c = S_c / float(np.sum(sel_boost.alphas[:T]))
    q_m = S_m / float(np.sum(sel_boost.alphas[:T]))
    got = {"clean_acc": accuracy(yte, S_c.argmax(1)),
           "masked_acc": accuracy(yte, S_m.argmax(1)),
           "clean_conf": float(q_c.max(axis=1).mean()),
           "masked_conf": float(q_m.max(axis=1).mean())}
    ok = all(_close(v, q2d["boosting"][k]) for k, v in got.items())
    ok = check("clean_masked: boosting accuracy/confidence reproduces on "
               "both test sets", ok)

    # Q2(d) masking case study
    case = q2d["case_study"]
    ens = BaggingEnsemble(q1a["frozen_params"], CFG["q3"]["bag_size"],
                          seed=SEED)
    ens.fit(Xtr, ytr)
    B = ens.n_estimators
    bag_c = np.array(q1d["per_example"]["bagging_pred_clean"])
    bag_m = np.array(q1d["per_example"]["bagging_pred_masked"])
    boost_c, boost_m = S_c.argmax(1), S_m.argmax(1)
    both_ok = (bag_c == yte) & (boost_c == yte)
    bag_stays = both_ok & (bag_m == yte) & (boost_m != yte)
    boost_stays = both_ok & (boost_m == yte) & (bag_m != yte)
    diff = ((bag_m == yte) & (boost_m != yte)) | \
           ((boost_m == yte) & (bag_m != yte))
    if bag_stays.any():
        pos, kind = int(np.flatnonzero(bag_stays)[0]), \
            "bagging_stays_correct_boosting_flips"
    elif boost_stays.any():
        pos, kind = int(np.flatnonzero(boost_stays)[0]), \
            "boosting_stays_correct_bagging_flips"
    elif diff.any():
        pos, kind = int(np.flatnonzero(diff)[0]), "differing_masked_outcome"
    else:
        pos, kind = None, None
    ok = (kind == case["type"]
          and (pos is None or int(split["test"][pos]) == case["test_index"]))
    true_k = int(yte[pos]) if pos is not None else None
    votes_c = ens.vote_fractions(Xte, B)[pos] * B
    votes_m = ens.vote_fractions(Xte_m, B)[pos] * B
    tp_c = ens.tree_predictions(Xte, B)[:, pos]
    tp_m = ens.tree_predictions(Xte_m, B)[:, pos]

    def frac_splits_in_block(trees):
        fracs = []
        for tr in trees:
            f = tr.tree_.feature
            f = f[f >= 0]
            if f.size == 0:
                continue          # tree without internal splits
            r, c = f // 64, f % 64
            fracs.append(float(((r >= 24) & (r <= 39) &
                                (c >= 24) & (c <= 39)).mean()))
        return None if not fracs else float(np.mean(fracs))

    pairs = (("bagging_votes_true_clean", float(votes_c[true_k])),
             ("bagging_votes_true_masked", float(votes_m[true_k])),
             ("bagging_disagreement_this_face_clean",
              float(np.mean(tp_c[:, None] != tp_c[None, :]))),
             ("bagging_disagreement_this_face_masked",
              float(np.mean(tp_m[:, None] != tp_m[None, :]))),
             ("boosting_q_true_clean", float(q_c[pos, true_k])),
             ("boosting_q_true_masked", float(q_m[pos, true_k])),
             ("bagging_frac_splits_in_block",
              frac_splits_in_block(ens.trees[:B])),
             ("boosting_frac_splits_in_block",
              frac_splits_in_block(sel_boost.trees[:T])))
    bad = [k for k, v in pairs if not _close(v, case[k])]
    ok = ok and not bad
    return check("clean_masked: the Q2(d) masking case study reproduces "
                 "(face, votes, q_true, splits-in-block)", ok,
                 ", ".join(bad) if bad else f"test image {case['test_index']}")


# ---------------------------------------------------------------------------
def check_q3_statistics(X, y, split, q1a, q1b, q3a, q3b, q3c, q3d, q1d):
    section("CHECK 7  Q3 bias/variance statistics (from the 30 checkpoints)")
    if not all((CKPT / f"q3_rep{r}.npz").exists()
               for r in range(CFG["q3"]["n_repetitions"])):
        return check("q3_statistics: 30 per-repetition checkpoints present",
                     False, "run python src/q3.py first")

    yte = y[split["test"]]
    e = one_hot(yte)
    data = []
    for r in range(CFG["q3"]["n_repetitions"]):
        with np.load(CKPT / f"q3_rep{r}.npz") as z:
            data.append({k: z[k] for k in z.files})

    keys = ["d4", "sel", "d16"]
    stats = {}
    for name in ([f"tree_{k}" for k in keys] + [f"bag_{k}" for k in keys]
                 + ["boost"]):
        P = np.stack([d[f"P_{name}"] for d in data])
        acc = np.array([d[f"acc_{name}"][0] for d in data])
        p_bar = P.mean(axis=0)
        stats[name] = {
            "bias2": float(np.mean(np.sum((p_bar - e) ** 2, axis=1))),
            "variance": float(np.mean(np.sum((P - p_bar[None]) ** 2,
                                             axis=2))),
            "acc_mean": float(acc.mean()),
            "acc_std": float(acc.std())}

    ok = True
    for pre, q3key, key in (("tree", "tree", "sel"),
                            ("bag", "bagging", "sel"),
                            ("boost", "boosting", "")):
        skey = "boost" if pre == "boost" else f"{pre}_{key}"
        for k in ("bias2", "variance", "acc_mean", "acc_std"):
            if not _close(stats[skey][k], q3a[q3key][k]):
                print(f"  q3a mismatch for {q3key}.{k}")
                ok = False
    ok = check("q3_statistics: Q3(a) bias2/variance/accuracy reproduce "
               "from the checkpoints", ok)

    ok = True
    for row in q3b["rows"]:
        key = "sel" if row["max_depth"] is None else f"d{row['max_depth']}"
        for pre in ("tree", "bag"):
            for k, stat_key in (("bias2", "bias2"), ("var", "variance"),
                                ("acc", "acc_mean")):
                if not _close(stats[f"{pre}_{key}"][stat_key],
                              row[f"{pre}_{k}"]):
                    print(f"  q3b mismatch depth={row['max_depth']} "
                          f"{pre}_{k}")
                    ok = False
        if not _close((row["tree_var"] - row["bag_var"]) / row["tree_var"],
                      row["var_removed_frac"]):
            ok = False
        if not _close(row["bag_acc"] - row["tree_acc"], row["acc_gain"]):
            ok = False
        dis = float(np.mean([d[f"dis_bag_{key}"][0] for d in data]))
        if not _close(dis, row["disagreement_mean"]):
            ok = False
    ok = check("q3_statistics: Q3(b) depth-study rows reproduce (bias, "
               "variance, accuracy, variance removed, gain, disagreement)",
               ok)

    # Q3(c): original-split bagging at the bounded depths + Q1 references
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xte = X[split["test"]]
    yte = y[split["test"]]
    occ = {str(r["max_depth"]): r for r in q1a["grid"]
           if r["min_samples_leaf"] ==
           q1a["frozen_params"]["min_samples_leaf"]}
    ok = True
    for m in q3c["measurements"]:
        d = m["max_depth"]
        if d is None:
            bagged = next(r["test_acc"] for r in q1b["sweep"]
                          if r["B"] == q1b["final_B"])
        else:
            ens = BaggingEnsemble(
                {"max_depth": d,
                 "min_samples_leaf":
                 q1a["frozen_params"]["min_samples_leaf"]},
                CFG["q3"]["bag_size"], seed=SEED)
            ens.fit(Xtr, ytr)
            bagged = accuracy(yte, ens.predict(Xte))
        if not (_close(bagged, m["bagged_orig"])
                and _close(occ[str(d)]["test_acc"], m["single_orig"])
                and _close((bagged - m["single_orig"]) * 100,
                           m["gain_orig_pp"])
                and _close(m["gain_resamples_pp"],
                           [r for r in q3b["rows"]
                            if r["max_depth"] == d][0]["acc_gain"] * 100)):
            print(f"  q3c mismatch at depth={d}")
            ok = False
    ok = check("q3_statistics: Q3(c) original-split bagging gains reproduce",
               ok)

    ok = True
    for model, key in (("single_tree", "tree_sel"), ("bagging", "bag_sel"),
                       ("boosting", "boost")):
        for k in ("bias2", "variance"):
            if not _close(stats[key][k], q3d[model][k]):
                print(f"  q3d mismatch for {model}.{k}")
                ok = False
    ok = check("q3_statistics: Q3(d) final-table bias2/variance reproduce",
               ok)
    # cross-check the clean/masked accuracies quoted in the Q3(d) table
    # against the Q1(d)/Q2(d) evaluations
    ok2 = (_close(q3d["single_tree"]["clean_acc"], q1d["clean"]["tree_acc"])
           and _close(q3d["single_tree"]["masked_acc"],
                      q1d["masked"]["tree_acc"])
           and _close(q3d["bagging"]["clean_acc"],
                      q1d["clean"]["bagging_acc"])
           and _close(q3d["bagging"]["masked_acc"],
                      q1d["masked"]["bagging_acc"]))
    return check("q3_statistics: Q3(d) clean/masked accuracies agree with "
                 "the Q1(d)/Q2(d) evaluations", ok2) and ok


# ---------------------------------------------------------------------------
def check_refit_determinism(X, y, split):
    section("CHECK 8  refit determinism (fit twice == fit fresh)")
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]

    # SammeBoost on the face data (few rounds so the check stays fast)
    boost = SammeBoost({"max_depth": 2, "min_samples_leaf": 1},
                       n_rounds=10, seed=SEED).fit(Xtr, ytr)
    hist1 = (list(boost.alphas), [w.copy() for w in boost.weight_history],
             list(boost.train_error), list(boost.train_predictions),
             boost.stop_reason)
    boost.fit(Xtr, ytr)                      # refit the SAME object
    hist2 = (list(boost.alphas), [w.copy() for w in boost.weight_history],
             list(boost.train_error), list(boost.train_predictions),
             boost.stop_reason)
    fresh = SammeBoost({"max_depth": 2, "min_samples_leaf": 1},
                       n_rounds=10, seed=SEED).fit(Xtr, ytr)
    hist3 = (list(fresh.alphas), [w.copy() for w in fresh.weight_history],
             list(fresh.train_error), list(fresh.train_predictions),
             fresh.stop_reason)

    def same_hist(a, b):
        return (len(a[0]) == len(b[0])
                and all(_close(x, z) for x, z in zip(a[0], b[0]))
                and all(_close(x, z) for x, z in zip(a[1], b[1]))
                and all(_close(x, z) for x, z in zip(a[2], b[2]))
                and all(np.array_equal(x, z)
                        for x, z in zip(a[3], b[3]))
                and a[4] == b[4])

    same = same_hist(hist1, hist2)
    same_fresh = same_hist(hist1, hist3)
    same_pred = _close(boost.predict(Xva), fresh.predict(Xva))
    ok = check("refit_determinism: SammeBoost refit == fresh fit (alphas, "
               "weight histories, errors, predictions)", 
               same and same_fresh and same_pred)

    ens = BaggingEnsemble({"max_depth": 4, "min_samples_leaf": 2}, 20,
                          seed=SEED).fit(Xtr, ytr)
    p1 = ens.predict(Xva)
    idx1 = ens.bootstrap_indices.copy()
    ens.fit(Xtr, ytr)                        # refit the SAME object
    same = (_close(p1, ens.predict(Xva))
            and np.array_equal(idx1, ens.bootstrap_indices))
    fresh = BaggingEnsemble({"max_depth": 4, "min_samples_leaf": 2}, 20,
                            seed=SEED).fit(Xtr, ytr)
    same_fresh = (_close(p1, fresh.predict(Xva))
                  and np.array_equal(idx1, fresh.bootstrap_indices))
    return check("refit_determinism: BaggingEnsemble refit == fresh fit "
                 "(bootstraps, predictions)", same and same_fresh)


# ---------------------------------------------------------------------------
CHECK_NAMES = ["input_validation", "tree_grid", "bagging_sweep",
               "oob_diagnostics", "samme_toy", "boosting_selection",
               "clean_masked", "q3_statistics", "refit_determinism"]


def main():
    t0 = time.perf_counter()
    want = set(sys.argv[1:])
    unknown = want - set(CHECK_NAMES)
    if unknown:
        print(f"unknown check name(s): {sorted(unknown)}; "
              f"valid names: {CHECK_NAMES}")
        sys.exit(2)
    run_all = not want
    print(f"independent validation of the saved results against the seeded "
          f"procedures (seed {SEED})"
          + ("" if run_all else f"  [checks: {sorted(want)}]"))
    X, y = load_faces()
    split = load_split()
    q1a = load_json(RESULTS / "q1a_tree_grid.json")
    q1b = load_json(RESULTS / "q1b_bagging_sweep.json")
    q2b = load_json(RESULTS / "q2b_boosting_grid.json")
    cache = {}

    if "input_validation" in want or run_all:
        check_input_validation(X, y, split)
    if "tree_grid" in want or run_all:
        check_tree_grid(X, y, split, q1a)
    if {"bagging_sweep", "oob_diagnostics", "clean_masked"} & want or run_all:
        if "bagging_sweep" in want or run_all:
            cache["ens"] = check_bagging_sweep(
                X, y, split, q1a, q1b,
                build_bagging_ensemble(X, y, split, q1a))
        else:
            cache["ens"] = build_bagging_ensemble(X, y, split, q1a)
    if "oob_diagnostics" in want or run_all:
        check_oob_diagnostics(X, y, split, q1b,
                              load_json(RESULTS / "q1c_diagnostics.json"),
                              cache["ens"])
    if "samme_toy" in want or run_all:
        check_samme_toy(load_json(RESULTS / "q2a_synth_round.json"))
    if "boosting_selection" in want or run_all:
        cache["models"] = build_boosting_models(X, y, split)
        check_boosting_selection(X, y, split, q2b, cache["models"])
    if "clean_masked" in want or run_all:
        if "models" not in cache:
            d = q2b["selected"]["depth"]
            cache["models"] = build_boosting_models(X, y, split, depths=[d])
        check_clean_masked(X, y, split, q1a, q1b,
                           load_json(RESULTS / "q1d_clean_masked.json"),
                           q2b,
                           load_json(RESULTS / "q2d_final_comparison.json"),
                           cache["ens"], cache["models"])
    if "q3_statistics" in want or run_all:
        check_q3_statistics(X, y, split, q1a, q1b,
                            load_json(RESULTS / "q3a_bias_variance.json"),
                            load_json(RESULTS / "q3b_depth_study.json"),
                            load_json(RESULTS / "q3c_deepest_analysis.json"),
                            load_json(RESULTS / "q3d_final_table.json"),
                            load_json(RESULTS / "q1d_clean_masked.json"))
    if "refit_determinism" in want or run_all:
        check_refit_determinism(X, y, split)

    n_pass = sum(1 for _, ok in _checks if ok)
    n_fail = len(_checks) - n_pass
    print("\n" + "=" * 72)
    print(f"summary: {n_pass} PASS, {n_fail} FAIL "
          f"({time.perf_counter() - t0:.0f}s)")
    for name, ok in _checks:
        if not ok:
            print(f"  FAILED: {name}")
    print("=" * 72)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
