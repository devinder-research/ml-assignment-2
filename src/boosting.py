"""
boosting.py - multiclass SAMME AdaBoost, implemented from scratch.

Algorithm (Zhu, Zou, Rosset & Hastie, 2009) with K classes:
    alpha_t = log((1 - eps_t) / eps_t) + log(K - 1)
    S_k(x)  = sum_t alpha_t * 1[h_t(x) = k]      -> predict argmax_k S_k(x)
    q_k(x)  = S_k(x) / sum_t alpha_t             (confidence)

Weight update (reweighting, no resampling):
    w_i <- w_i * exp(alpha_t * 1[h_t(x_i) != y_i]), then renormalise.

Degenerate learners are handled explicitly:
  * eps_t >= 1 - 1/K (no better than random multiclass guessing): the
    learner is rejected, boosting stops early and the reason is logged.
  * eps_t == 0 (perfect learner): alpha_t would be log(inf), so it is
    capped at log((1-1e-12)/1e-12) + log(K-1); the learner is kept and
    boosting stops. Nothing is misclassified, so the weight update is a
    no-op and the weights stay unchanged.
  * 0 < eps_t < 1e-12 (near-perfect): same capped alpha, boosting stops,
    but here the update IS applied so the tiny misclassified mass is
    up-weighted exactly as the rule prescribes.

The weak learner is sklearn DecisionTreeClassifier fitted with the
current sample weights; everything else is implemented in this file.
"""

import time

import numpy as np

from bagging import make_tree
from data import N_CLASSES, SEED

EPS_FLOOR = 1e-12          # below this the learner counts as perfect
RANDOM_TOL = 1e-12


def samme_alpha(eps, k):
    return float(np.log((1.0 - eps) / eps) + np.log(k - 1))


class SammeBoost:
    """SAMME boosting with deterministic reweighting."""

    def __init__(self, base_params, n_rounds, n_classes=N_CLASSES,
                 seed=SEED):
        self.base_params = dict(base_params)
        self.n_rounds = int(n_rounds)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.trees = []
        self.alphas = []
        self.stop_reason = None
        self.initial_weights = None
        # per-round history (post-update weights = what the next learner sees)
        self.train_error = []        # eps_t
        self.weight_history = []     # sample weights after each round
        self.train_predictions = []  # h_t(x) on the training set
        self.max_weight = []
        self.top10_share = []
        self.cum_seconds = []     # cumulative wall-clock time after each round
        self.fit_seconds = 0.0

    def fit(self, X, y):
        self._t0 = time.perf_counter()
        X, y = np.asarray(X), np.asarray(y)
        n = X.shape[0]
        w = np.full(n, 1.0 / n)
        self.initial_weights = w.copy()
        K = self.n_classes
        alpha_cap = samme_alpha(EPS_FLOOR, K)

        for t in range(self.n_rounds):
            tree = make_tree(self.base_params, seed=self.seed)
            tree.fit(X, y, sample_weight=w)
            pred = tree.predict(X).astype(int)
            wrong = pred != y
            eps = float(w[wrong].sum())

            if eps >= 1.0 - 1.0 / K - RANDOM_TOL:
                # not better than random: reject, stop early
                self.stop_reason = "not_better_than_random"
                break

            if eps < EPS_FLOOR:
                # perfect / near-perfect learner: cap alpha and stop
                w = self._keep(tree, alpha_cap, pred, eps, w, wrong)
                self.stop_reason = "perfect_learner" if eps == 0.0 \
                    else "near_perfect_learner"
                break

            # normal round
            alpha = samme_alpha(eps, K)
            w = self._keep(tree, alpha, pred, eps, w, wrong)

        if self.stop_reason is None:
            self.stop_reason = "completed"
        self.fit_seconds = self.cum_seconds[-1] if self.cum_seconds else 0.0
        return self

    def _keep(self, tree, alpha, pred, eps, w, wrong):
        """Store one accepted round; apply the SAMME weight update and
        return the new weights (a no-op when eps == 0, since nothing was
        misclassified)."""
        self.trees.append(tree)
        self.alphas.append(float(alpha))
        self.train_predictions.append(pred)
        self.train_error.append(eps)
        if eps > 0.0:
            w = w * np.exp(alpha * wrong)
            w = w / w.sum()
        self.weight_history.append(w.copy())
        self.max_weight.append(float(w.max()))
        self.top10_share.append(float(np.sort(w)[::-1][:10].sum()))
        self.cum_seconds.append(time.perf_counter() - self._t0)
        return w

    # -- inference -----------------------------------------------------------
    def class_scores(self, X, n_rounds=None):
        """S_k(x) = sum_t alpha_t * 1[h_t(x) = k] over the first t rounds."""
        T = len(self.trees) if n_rounds is None else int(n_rounds)
        S = np.zeros((np.asarray(X).shape[0], self.n_classes))
        for tree, alpha in zip(self.trees[:T], self.alphas[:T]):
            preds = tree.predict(X).astype(int)
            for k in range(self.n_classes):
                S[:, k] += alpha * (preds == k)
        return S

    def qk(self, X, n_rounds=None):
        """Normalised weighted vote q_k(x) = S_k(x) / sum_t alpha_t."""
        T = len(self.trees) if n_rounds is None else int(n_rounds)
        S = self.class_scores(X, T)
        total = float(np.sum(self.alphas[:T]))
        if total == 0.0:                      # empty ensemble -> uniform
            return np.full((S.shape[0], self.n_classes), 1.0 / self.n_classes)
        return S / total

    def predict(self, X, n_rounds=None):
        return np.argmax(self.class_scores(X, n_rounds), axis=1)

    def confidence(self, X, n_rounds=None):
        """Winning normalised weighted vote per example."""
        return self.qk(X, n_rounds).max(axis=1)

    @property
    def n_rounds_run(self):
        return len(self.trees)
