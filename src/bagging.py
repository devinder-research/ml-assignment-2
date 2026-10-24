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


def make_tree(params, seed=SEED):
    """Base learner factory with a fixed random_state."""
    return DecisionTreeClassifier(random_state=seed, **params)


class BaggingEnsemble:
    """B trees, each trained on a bootstrap sample of the training set."""

    def __init__(self, base_params, n_estimators, seed=SEED,
                 n_classes=N_CLASSES):
        self.base_params = dict(base_params)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.n_classes = n_classes
        self.trees = []
        self.bootstrap_indices = None   # (B, n)
        self.oob_mask = None            # (B, n), True = example was OOB
        self.fit_seconds = 0.0

    def fit(self, X, y, bootstrap_indices=None, rng=None):
        X, y = np.asarray(X), np.asarray(y)
        n = X.shape[0]
        if bootstrap_indices is None:
            if rng is None:
                rng = np.random.default_rng(self.seed)
            bootstrap_indices = rng.integers(0, n,
                                             size=(self.n_estimators, n))
        self.bootstrap_indices = np.asarray(bootstrap_indices, dtype=int)
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
    def tree_predictions(self, X, n_estimators=None):
        B = self.n_estimators if n_estimators is None else int(n_estimators)
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
        B = self.n_estimators if n_estimators is None else int(n_estimators)
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
        B = self.n_estimators if n_estimators is None else int(n_estimators)
        return np.array([np.unique(self.bootstrap_indices[b]).size
                         for b in range(B)])

    def pairwise_disagreement(self, X, n_estimators=None):
        B = self.n_estimators if n_estimators is None else int(n_estimators)
        return pairwise_disagreement(self.tree_predictions(X, B))
