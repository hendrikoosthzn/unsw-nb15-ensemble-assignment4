"""
Random forest built on top of scikit-learn's CART implementation.

The CART base learner (DecisionTreeClassifier) is reused from assignment 2.
Everything that makes it an ensemble is implemented here:
  - bootstrap sampling (standard, or class-balanced as in Chen, Liaw & Breiman 2004)
  - random feature subspace at every split (passed to the tree as max_features)
  - class weighting (none / balanced / balanced_subsample)
  - soft voting (mean of per-tree class probabilities)
  - out-of-bag (OOB) predictions
  - diagnostics: individual tree strength and pairwise disagreement (diversity)

Bootstrap samples are represented as integer multiplicity weights on the full
training set, which is equivalent to fitting on the resampled rows but avoids
copying the data for every tree.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


def _class_weights(y, K, mult=None):
    """'balanced' weights n / (K * n_c), computed on the (possibly resampled) set."""
    counts = np.bincount(y, weights=mult, minlength=K).astype(float)
    w = np.zeros(K)
    nz = counts > 0
    w[nz] = counts.sum() / (nz.sum() * counts[nz])
    return w


def _fit_one(X, y, K, seed, cfg):
    rng = np.random.default_rng(seed)
    n = len(y)

    if cfg["bootstrap"] == "standard":
        mult = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
    elif cfg["bootstrap"] == "balanced":
        # equal number of draws from every class; total size kept at n
        per = n // K
        mult = np.zeros(n)
        for c in range(K):
            idx = np.flatnonzero(y == c)
            mult += np.bincount(rng.choice(idx, per, replace=True), minlength=n)
    elif cfg["bootstrap"] == "none":
        mult = np.ones(n)
    else:
        raise ValueError(cfg["bootstrap"])

    cw = cfg["class_weight"]
    if cw is None:
        sw = mult
    elif cw == "balanced":
        sw = mult * cfg["_global_w"][y]
    elif cw == "balanced_subsample":
        sw = mult * _class_weights(y, K, mult)[y]
    else:
        raise ValueError(cw)

    tree = DecisionTreeClassifier(criterion=cfg["criterion"],
                                  max_depth=cfg["max_depth"],
                                  min_samples_leaf=cfg["min_samples_leaf"],
                                  max_features=cfg["max_features"],
                                  random_state=int(rng.integers(2**31 - 1)))
    tree.fit(X, y, sample_weight=sw)
    return tree, mult == 0          # out-of-bag mask


class RandomForest:
    def __init__(self, n_trees=200, max_features="sqrt", max_depth=None,
                 min_samples_leaf=1, criterion="gini", class_weight=None,
                 bootstrap="standard", n_jobs=-1, seed=42):
        self.cfg = dict(max_features=max_features, max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf, criterion=criterion,
                        class_weight=class_weight, bootstrap=bootstrap)
        self.n_trees, self.n_jobs, self.seed = n_trees, n_jobs, seed

    def fit(self, X, y, oob=False):
        self.K = int(y.max()) + 1
        self.cfg["_global_w"] = _class_weights(y, self.K)
        seeds = np.random.SeedSequence(self.seed).generate_state(self.n_trees)
        out = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_one)(X, y, self.K, int(s), self.cfg) for s in seeds)
        self.trees = [t for t, _ in out]
        if oob:
            P = np.zeros((len(y), self.K))
            cnt = np.zeros(len(y))
            for t, m in out:
                P[m] += self._tree_proba(t, X[m])
                cnt[m] += 1
            ok = cnt > 0
            self.oob_pred_ = np.full(len(y), -1)
            self.oob_pred_[ok] = P[ok].argmax(1)
            self.oob_f1_ = f1_score(y[ok], self.oob_pred_[ok], average="macro")
        return self

    def _tree_proba(self, t, X):
        p = np.zeros((len(X), self.K))
        p[:, t.classes_] = t.predict_proba(X)
        return p

    def tree_probas(self, X):
        """(n_trees, n_samples, K) array of per-tree class probabilities."""
        return np.stack(Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._tree_proba)(t, X) for t in self.trees))

    def predict_proba(self, X, first=None):
        P = self.tree_probas(X[:, :]) if first is None else \
            np.stack([self._tree_proba(t, X) for t in self.trees[:first]])
        return P.mean(0)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def feature_importances(self):
        return np.mean([t.feature_importances_ for t in self.trees], axis=0)


def diversity(tree_P, y, n_pairs=300, seed=0):
    """
    strength    : mean macro-F1 of the individual trees
    disagreement: mean fraction of instances on which two trees predict differently
    """
    preds = tree_P.argmax(2)                       # (T, n)
    T = len(preds)
    strength = np.mean([f1_score(y, p, average="macro") for p in preds])
    rng = np.random.default_rng(seed)
    pairs = [rng.choice(T, 2, replace=False) for _ in range(n_pairs)]
    dis = np.mean([np.mean(preds[i] != preds[j]) for i, j in pairs])
    return strength, dis
