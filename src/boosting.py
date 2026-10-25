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


def _validate_training_data(X, y, n_classes):
    """Shared input validation for fit() (clear ValueError messages)."""
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional (n_samples, n_features), "
                         f"got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
    n = X.shape[0]
    if n == 0:
        raise ValueError("empty dataset: X has no samples")
    if X.shape[1] == 0:
        raise ValueError("empty dataset: X has no features")
    if y.shape[0] != n:
        raise ValueError(f"X and y have mismatched sample counts: "
                         f"X has {n}, y has {y.shape[0]}")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"y must contain numeric class labels, got dtype "
                         f"{y.dtype}")
    y_int = y.astype(int)
    if not np.array_equal(y_int, y):
        raise ValueError("y must contain integer class labels")
    if y_int.min() < 0 or y_int.max() >= n_classes:
        raise ValueError(f"invalid class labels: labels must be integers in "
                         f"[0, {n_classes}), got range "
                         f"[{y_int.min()}, {y_int.max()}]")
    return X, y_int


def _validate_positive_int(value, name, minimum=1):
    """Reject zero/negative or non-integer ensemble parameters."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer >= {minimum}, "
                         f"got {value!r} of type {type(value).__name__}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be a positive integer >= {minimum}, "
                         f"got {value}")
    return value


class SammeBoost:
    """SAMME boosting with deterministic reweighting."""

    def __init__(self, base_params, n_rounds, n_classes=N_CLASSES,
                 seed=SEED):
        self.base_params = dict(base_params)
        self.n_rounds = _validate_positive_int(n_rounds, "n_rounds")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2 for SAMME, got "
                             f"{n_classes}")
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

    def _reset_state(self):
        """Clear every fitted attribute so that refitting the same object
        yields exactly the same model as fitting a fresh instance (repeated
        fit() must not append to the previous run's histories)."""
        self.trees = []
        self.alphas = []
        self.stop_reason = None
        self.initial_weights = None
        self.train_error = []
        self.weight_history = []
        self.train_predictions = []
        self.max_weight = []
        self.top10_share = []
        self.cum_seconds = []
        self.fit_seconds = 0.0

    def fit(self, X, y):
        # refitting the same object must start from a clean state
        self._reset_state()
        self._t0 = time.perf_counter()
        X, y = _validate_training_data(X, y, self.n_classes)
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
    def _resolve_prefix(self, n_rounds):
        """Validate an ensemble-prefix request and return the number of
        accepted rounds to use (1..n_rounds_run)."""
        n_run = len(self.trees)
        if n_run == 0:
            raise ValueError("boosting model is not fitted yet (or the first "
                             "weak learner was rejected): no predictions "
                             "available before fit() accepts a learner")
        if n_rounds is None:
            return n_run
        if isinstance(n_rounds, bool) or not isinstance(n_rounds,
                                                        (int, np.integer)):
            raise ValueError(f"n_rounds prefix must be an integer, got "
                             f"{n_rounds!r}")
        T = int(n_rounds)
        if T < 1 or T > n_run:
            raise ValueError(f"n_rounds prefix must be in [1, {n_run}] "
                             f"(accepted learners), got {T}")
        return T

    def class_scores(self, X, n_rounds=None):
        """S_k(x) = sum_t alpha_t * 1[h_t(x) = k] over the first t rounds."""
        T = self._resolve_prefix(n_rounds)
        S = np.zeros((np.asarray(X).shape[0], self.n_classes))
        for tree, alpha in zip(self.trees[:T], self.alphas[:T]):
            preds = tree.predict(X).astype(int)
            for k in range(self.n_classes):
                S[:, k] += alpha * (preds == k)
        return S

    def qk(self, X, n_rounds=None):
        """Normalised weighted vote q_k(x) = S_k(x) / sum_t alpha_t."""
        T = self._resolve_prefix(n_rounds)
        S = self.class_scores(X, T)
        total = float(np.sum(self.alphas[:T]))
        if total == 0.0:                      # defensive; cannot occur
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
