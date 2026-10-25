"""
bagging.py - bagging (Breiman, 1996) implemented from scratch.

Only sklearn.tree.DecisionTreeClassifier is used as the base learner
(allowed by the assignment). The bootstrap sampling, majority vote,
out-of-bag evaluation, pairwise tree disagreement and vote margins are
all implemented here - no sklearn ensemble class is used.

Determinism: bootstrap indices are either passed in explicitly or drawn
from np.random.default_rng(seed) as ONE call rng.integers(0, n, (B, n)).
With splitter="best" the tree fit itself is deterministic, and the base
trees get a fixed random_state, so the whole ensemble is reproducible.
"""

import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from data import N_CLASSES, SEED, pairwise_disagreement, vote_fractions


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


def _validate_bootstrap_indices(bootstrap_indices, n_estimators, n):
    """Validate externally supplied bootstrap indices: shape (B, n),
    integer dtype, values in [0, n). Repeated indices are allowed and
    expected (sampling with replacement)."""
    idx = np.asarray(bootstrap_indices)
    if idx.ndim != 2 or idx.shape != (n_estimators, n):
        raise ValueError(f"bootstrap_indices must have shape "
                         f"({n_estimators}, {n}) = (n_estimators, n_samples), "
                         f"got {idx.shape}")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"bootstrap_indices must be integer-valued, got "
                         f"dtype {idx.dtype}")
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError(f"bootstrap_indices out of range: values must be in "
                         f"[0, {n}), got [{idx.min()}, {idx.max()}]")
    return idx.astype(np.int64)


def make_tree(params, seed=SEED):
    """Base learner factory with a fixed random_state."""
    return DecisionTreeClassifier(random_state=seed, **params)


class BaggingEnsemble:
    """B trees, each trained on a bootstrap sample of the training set."""

    def __init__(self, base_params, n_estimators, seed=SEED,
                 n_classes=N_CLASSES):
        if isinstance(n_estimators, bool) or \
                not isinstance(n_estimators, (int, np.integer)):
            raise ValueError(f"n_estimators must be a positive integer, got "
                             f"{n_estimators!r} of type "
                             f"{type(n_estimators).__name__}")
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be a positive integer >= 1, "
                             f"got {n_estimators}")
        self.base_params = dict(base_params)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.n_classes = n_classes
        self.trees = []
        self.bootstrap_indices = None   # (B, n)
        self.oob_mask = None            # (B, n), True = example was OOB
        self.fit_seconds = 0.0

    def fit(self, X, y, bootstrap_indices=None, rng=None):
        # refitting the same object must start from a clean state
        self.trees = []
        self.cum_seconds = []
        self.fit_seconds = 0.0
        self.bootstrap_indices = None
        self.oob_mask = None
        X, y = _validate_training_data(X, y, self.n_classes)
        n = X.shape[0]
        if bootstrap_indices is None:
            if rng is None:
                rng = np.random.default_rng(self.seed)
            bootstrap_indices = rng.integers(0, n,
                                             size=(self.n_estimators, n))
        self.bootstrap_indices = _validate_bootstrap_indices(
            bootstrap_indices, self.n_estimators, n)
        self.train_X, self.train_y = X, y

        t0 = time.perf_counter()
        self.trees = []
        self.cum_seconds = []     # cumulative wall-clock time after each tree
        for b in range(self.n_estimators):
            idx = self.bootstrap_indices[b]
            tree = make_tree(self.base_params, seed=self.seed)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            self.cum_seconds.append(time.perf_counter() - t0)
        self.fit_seconds = self.cum_seconds[-1]

        # example i is out-of-bag for tree b iff it was not drawn into
        # bootstrap sample b
        self.oob_mask = np.ones((self.n_estimators, n), dtype=bool)
        for b in range(self.n_estimators):
            self.oob_mask[b, self.bootstrap_indices[b]] = False
        return self

    # -- prediction ----------------------------------------------------------
    def _resolve_prefix(self, n_estimators):
        """Validate an ensemble-prefix request and return the number of
        trees to use (1..n_estimators fitted)."""
        if not self.trees:
            raise ValueError("bagging ensemble is not fitted yet: no "
                             "predictions available before fit()")
        if n_estimators is None:
            return len(self.trees)
        if isinstance(n_estimators, bool) or \
                not isinstance(n_estimators, (int, np.integer)):
            raise ValueError(f"n_estimators prefix must be an integer, got "
                             f"{n_estimators!r}")
        B = int(n_estimators)
        if B < 1 or B > len(self.trees):
            raise ValueError(f"n_estimators prefix must be in [1, "
                             f"{len(self.trees)}] (fitted trees), got {B}")
        return B

    def tree_predictions(self, X, n_estimators=None):
        B = self._resolve_prefix(n_estimators)
        return np.stack([t.predict(X) for t in self.trees[:B]])

    def vote_fractions(self, X, n_estimators=None):
        return vote_fractions(self.tree_predictions(X, n_estimators),
                              self.n_classes)

    def predict(self, X, n_estimators=None):
        # np.argmax breaks ties toward the smaller class index (fixed rule)
        return np.argmax(self.vote_fractions(X, n_estimators), axis=1)

    def confidence(self, X, n_estimators=None):
        """Winning vote fraction per example."""
        return self.vote_fractions(X, n_estimators).max(axis=1)

    # -- out-of-bag ----------------------------------------------------------
    def oob_accuracy(self, y_train, n_estimators=None):
        """OOB accuracy on the training set: each training example is voted
        only by the trees whose bootstrap sample excluded it. Accuracy is
        measured over the examples with at least one OOB tree.
        Returns (accuracy, coverage)."""
        B = self._resolve_prefix(n_estimators)
        y_train = np.asarray(self.train_y if y_train is None else y_train)
        n = len(y_train)
        train_preds = np.stack([t.predict(self.train_X)
                                for t in self.trees[:B]])
        oob = self.oob_mask[:B]
        cov = oob.any(axis=0)
        if not cov.any():
            return float("nan"), 0.0
        correct = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(cov):
            sel = oob[:, i]
            counts = np.bincount(train_preds[sel, i].astype(int),
                                 minlength=self.n_classes)
            # majority OOB vote, ties toward the smaller class index
            correct[i] = int(np.argmax(counts)) == y_train[i]
        return float(correct[cov].mean()), float(cov.mean())

    # -- diagnostics ---------------------------------------------------------
    def distinct_bootstrap_examples(self, n_estimators=None):
        """Distinct training examples in each of the first B bootstraps."""
        B = self._resolve_prefix(n_estimators)
        return np.array([np.unique(self.bootstrap_indices[b]).size
                         for b in range(B)])

    def pairwise_disagreement(self, X, n_estimators=None):
        B = self._resolve_prefix(n_estimators)
        return pairwise_disagreement(self.tree_predictions(X, B))
