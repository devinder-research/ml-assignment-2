"""
data.py - shared helpers: data loading, split handling, small metrics.

Roll number P26AI0004 -> numeric characters in order -> seed 260004.
Every random decision in the project comes from numpy.random.default_rng
seeded with this value (exact usage documented in each script).

Only raw flattened 64x64 pixels are used, no PCA or any other
dimensionality reduction.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

N_CLASSES = 40
IMG = (64, 64)


def load_config():
    return json.loads((REPO / "config.json").read_text())


CFG = load_config()
ROLL = CFG["roll_number"]
SEED = int("".join(ch for ch in ROLL if ch.isdigit()))  # P26AI0004 -> 260004
assert SEED == CFG["seed"], "config.json seed does not match the roll number"


# ---------------------------------------------------------------------------
def log(name):
    """Return a logger that prints and appends to logs/<name>.log.

    Appending (instead of overwriting) keeps one continuous log across
    the checkpointed resume invocations that Q3 documents.
    """
    import time as _time
    LOGS = REPO / "logs"
    LOGS.mkdir(exist_ok=True)
    f = open(LOGS / f"{name}.log", "a")
    stamp = _time.strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"\n===== {name} | run start {stamp} =====\n")
    f.flush()

    def log(msg):
        print(msg, flush=True)
        f.write(str(msg) + "\n")
        f.flush()
    return log


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_faces():
    """Olivetti faces: X (400, 4096) float in [0,1], y (400,) labels 0..39.

    Uses the local copy in data/ when present (so the repo runs offline),
    otherwise falls back to sklearn.datasets.fetch_olivetti_faces and
    caches it there.
    """
    npz = REPO / "data" / "olivetti_faces.npz"
    if npz.exists():
        with np.load(npz) as z:
            X = z["data"].astype(np.float64)
            y = z["target"].astype(np.int64)
    else:
        from sklearn.datasets import fetch_olivetti_faces
        bunch = fetch_olivetti_faces(data_home=str(REPO / "data"),
                                     shuffle=False, random_state=0)
        X = np.asarray(bunch.data, dtype=np.float64)
        y = np.asarray(bunch.target, dtype=np.int64)
        np.savez_compressed(npz, data=X, target=y)

    assert X.shape == (400, 64 * 64) and y.shape == (400,)
    for c in range(N_CLASSES):
        assert (y == c).sum() == 10
    return X, y


# ---------------------------------------------------------------------------
# Split (created once by src/make_split.py, loaded from disk everywhere else)
# ---------------------------------------------------------------------------
SPLIT_NPZ = REPO / "splits" / "split_seed.npz"
SPLIT_JSON = REPO / "splits" / "split_seed.json"


def make_split(y, seed=SEED):
    """Random 6/2/2 split within each identity.

    Exact procedure (deterministic for a given seed):
        rng = np.random.default_rng(seed)
        for each identity c = 0..39:
            perm = rng.permutation(indices of identity c)
            train += perm[:6], val += perm[6:8], test += perm[8:10]
    """
    rng = np.random.default_rng(seed)
    tr, va, te, per_id = [], [], [], {}
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        perm = rng.permutation(idx)
        tr += perm[:6].tolist()
        va += perm[6:8].tolist()
        te += perm[8:10].tolist()
        per_id[c] = {"train": perm[:6].tolist(),
                     "val": perm[6:8].tolist(),
                     "test": perm[8:10].tolist()}
    split = {"train": np.array(tr, dtype=np.int64),
             "val": np.array(va, dtype=np.int64),
             "test": np.array(te, dtype=np.int64)}
    validate_split(split, y)
    return split, per_id


def validate_split(split, y):
    tr, va, te = split["train"], split["val"], split["test"]
    assert tr.size == 240 and va.size == 80 and te.size == 80
    assert set(np.concatenate([tr, va, te]).tolist()) == set(range(400))
    for name, arr in (("train", tr), ("val", va), ("test", te)):
        want = 6 if name == "train" else 2
        assert all((y[arr] == c).sum() == want for c in range(N_CLASSES))


def split_sha256(split):
    h = hashlib.sha256()
    for key in ("train", "val", "test"):
        h.update(np.ascontiguousarray(split[key]).tobytes())
    return h.hexdigest()


def save_split(split, per_id, seed=SEED):
    SPLIT_NPZ.parent.mkdir(exist_ok=True)
    np.savez(SPLIT_NPZ, train=split["train"], val=split["val"],
             test=split["test"], seed=np.int64(seed))
    payload = {
        "roll_number": ROLL,
        "seed": seed,
        "seed_rule": f"'{ROLL}' -> '{seed}' (numeric characters in order)",
        "sizes": {"train": 240, "val": 80, "test": 80},
        "per_identity": {str(c): per_id[c] for c in range(N_CLASSES)},
        "sha256_split_indices": split_sha256(split),
    }
    SPLIT_JSON.write_text(json.dumps(payload, indent=2))
    return payload


def load_split():
    """Load the saved split. Experiments never re-generate it themselves."""
    if not SPLIT_NPZ.exists():
        sys.exit("Saved split not found. Run: python src/make_split.py")
    with np.load(SPLIT_NPZ) as z:
        split = {"train": z["train"], "val": z["val"], "test": z["test"]}
    meta = json.loads(SPLIT_JSON.read_text()) if SPLIT_JSON.exists() else {}
    if meta:
        assert split_sha256(split) == meta["sha256_split_indices"], \
            "saved split failed its sha256 check - regenerate it"
    return split


# ---------------------------------------------------------------------------
def mask_images(X):
    """Zero the centred 16x16 block (rows and columns 24..39)."""
    Xm = X.reshape(-1, *IMG).copy()
    Xm[:, 24:40, 24:40] = 0.0
    return Xm.reshape(X.shape[0], -1)


# ---------------------------------------------------------------------------
# Small metrics
# ---------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def one_hot(y, k=N_CLASSES):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return out


def vote_fractions(preds_by_tree, k=N_CLASSES):
    """preds_by_tree: (B, n) votes -> (n, k) fraction matrix, rows sum to 1."""
    preds_by_tree = np.asarray(preds_by_tree, dtype=int)
    B = preds_by_tree.shape[0]
    counts = np.zeros((preds_by_tree.shape[1], k))
    for c in range(k):
        counts[:, c] = (preds_by_tree == c).sum(axis=0)
    return counts / B


def vote_margins(votes):
    """Winning vote fraction minus runner-up, per example."""
    s = np.sort(votes, axis=1)
    return s[:, -1] - s[:, -2]


def pairwise_disagreement(preds_by_tree):
    """Mean fraction of examples on which two trees disagree, over all
    unordered tree pairs (None if fewer than 2 trees)."""
    P = np.asarray(preds_by_tree, dtype=int)
    B = P.shape[0]
    if B < 2:
        return None
    iu = np.triu_indices(B, k=1)
    return float((P[:, None, :] != P[None, :, :])[iu].mean())


def tree_leaf_stats(model):
    """Leaves, mean training samples per leaf, fraction of 1-sample leaves."""
    t = model.tree_
    leaf = t.children_left < 0
    n = t.n_node_samples[leaf]
    return {"n_leaves": int(leaf.sum()),
            "mean_samples_per_leaf": float(n.mean()),
            "frac_single_example_leaves": float(np.mean(n == 1))}
