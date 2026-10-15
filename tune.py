"""
Random forest tuning on the 40k tuning subsample (5-fold stratified CV).
Each experiment answers one question fixed before any final evaluation:

  E0  Does the encoding of nominal features matter for the forest?
      (one-hot as in assignment 2, versus integer codes)
  E1  How many trees are needed before performance stops improving?
  E2  How does the size of the random feature subset (max_features) trade
      individual tree strength against ensemble diversity? max_features = all
      features is plain bagging, so E2 also isolates the value of the random
      subspace on top of bagging.
  E3  Do fully grown trees remain best inside an ensemble, given that the
      single tree in assignment 2 was best at depth 20?
  E4  Which imbalance remedy gives the best trade-off between macro-F1 and
      recall on the rare attack classes?

  python3 tune.py [E0 E1 E2 E3 E4]
"""
import sys, time, json, os
import numpy as np
import pandas as pd
from common import (load, build_matrices, ordinal_matrix, split_tune_eval, folds,
                    score, per_class, SEED)
from forest import RandomForest, diversity
from sklearn.metrics import f1_score

OUT = "results"
os.makedirs(OUT, exist_ok=True)

df = load()
Xoh, _, y, _ = build_matrices(df, verbose=False)
Xord, _ = ordinal_matrix(df)
tune, _ = split_tune_eval(y)
Xoh, Xord, y = Xoh[tune], Xord[tune], y[tune]
F = folds(y, 5)
P_OH = Xoh.shape[1]
print(f"tuning set {len(y)} rows, one-hot width {P_OH}, ordinal width {Xord.shape[1]}")

BASE = dict(n_trees=200, max_features="sqrt", max_depth=None, min_samples_leaf=1,
            class_weight=None, bootstrap="standard")


def cv(X, cfg, extra=False):
    rows, cls = [], []
    for k, (tr, te) in enumerate(F):
        t0 = time.time()
        m = RandomForest(**cfg, seed=SEED + k).fit(X[tr], y[tr])
        fit_t = time.time() - t0
        TP = m.tree_probas(X[te]).astype(np.float32)
        yp = TP.mean(0).argmax(1)
        r = score(y[te], yp)
        r.update(fold=k, fit_s=fit_t,
                 depth=np.mean([t.get_depth() for t in m.trees]),
                 leaves=np.mean([t.get_n_leaves() for t in m.trees]))
        if extra:
            r["strength"], r["disagree"] = diversity(TP, y[te])
        rows.append(r)
        c = per_class(y[te], yp); c["fold"] = k; cls.append(c)
    return pd.DataFrame(rows), pd.concat(cls)


def tag(d, **kw):
    for k, v in kw.items():
        d[k] = str(v)
    return d


def report(name, res):
    m = res.mean(numeric_only=True)
    s = res.std(numeric_only=True)
    print(f"  {name:<40} f1m={m.f1_macro:.4f}±{s.f1_macro:.4f} bal={m.bal_acc:.4f} "
          f"acc={m.acc:.4f}" + (f" str={m.strength:.4f} dis={m.disagree:.4f}"
                                if "strength" in res else "") + f" ({m.fit_s:.1f}s/fit)")


def E0():
    print("\nE0 encoding")
    out = []
    for name, X in [("onehot", Xoh), ("ordinal", Xord)]:
        r, _ = cv(X, BASE)
        report(name, r); out.append(tag(r, encoding=name))
    pd.concat(out).to_csv(f"{OUT}/E0_encoding.csv", index=False)


def E1():
    print("\nE1 ensemble size")
    sizes = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]
    out = []
    for mf in ["sqrt", 1.0]:
        for k, (tr, te) in enumerate(F):
            m = RandomForest(**{**BASE, "n_trees": 500, "max_features": mf},
                             seed=SEED + k).fit(Xoh[tr], y[tr])
            TP = m.tree_probas(Xoh[te]).astype(np.float32)
            cum = np.cumsum(TP, axis=0)
            for s in sizes:
                r = score(y[te], cum[s - 1].argmax(1))
                r.update(fold=k, trees=s, max_features=str(mf))
                out.append(r)
        d = pd.DataFrame(out)
        print(d[d.max_features == str(mf)].groupby("trees")[["f1_macro", "acc"]]
              .mean().round(4).T.to_string())
    pd.DataFrame(out).to_csv(f"{OUT}/E1_size.csv", index=False)


def E2():
    print("\nE2 max_features (strength vs diversity)")
    out = []
    for mf in [1, 2, 4, 8, 16, 32, P_OH]:
        r, _ = cv(Xoh, {**BASE, "max_features": mf}, extra=True)
        report(f"max_features={mf}", r); out.append(tag(r, max_features=mf))
    pd.concat(out).to_csv(f"{OUT}/E2_max_features.csv", index=False)


def E3(mf):
    print(f"\nE3 tree complexity (max_features={mf})")
    out = []
    for crit in ["gini", "entropy"]:
        for d in [8, 12, 16, 20, 24, None]:
            for leaf in [1, 3, 10]:
                if crit == "entropy" and leaf != 1:
                    continue
                r, _ = cv(Xoh, {**BASE, "max_features": mf, "max_depth": d,
                                "min_samples_leaf": leaf, "criterion": crit})
                report(f"{crit} depth={d} leaf={leaf}", r)
                out.append(tag(r, criterion=crit, max_depth=d, min_samples_leaf=leaf))
    pd.concat(out).to_csv(f"{OUT}/E3_complexity.csv", index=False)


def E4(mf, depth, leaf, crit):
    print(f"\nE4 imbalance (max_features={mf}, depth={depth}, leaf={leaf}, {crit})")
    out, cls = [], []
    for name, cw, bs in [("none", None, "standard"),
                         ("balanced", "balanced", "standard"),
                         ("balanced_subsample", "balanced_subsample", "standard"),
                         ("balanced_bootstrap", None, "balanced")]:
        r, c = cv(Xoh, {**BASE, "max_features": mf, "max_depth": depth,
                        "min_samples_leaf": leaf, "criterion": crit,
                        "class_weight": cw, "bootstrap": bs})
        report(name, r)
        out.append(tag(r, remedy=name)); cls.append(tag(c, remedy=name))
        g = c.groupby("cls", sort=False)[["precision", "recall"]].mean()
        print("     recall   " + " ".join(f"{k[:6]}={v:.2f}" for k, v in g.recall.items()))
        print("     precision" + " ".join(f"{k[:6]}={v:.2f}" for k, v in g.precision.items()))
    pd.concat(out).to_csv(f"{OUT}/E4_imbalance.csv", index=False)
    pd.concat(cls).to_csv(f"{OUT}/E4_imbalance_perclass.csv", index=False)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["E0", "E1", "E2"]
    cfg = json.load(open("chosen.json")) if os.path.exists("chosen.json") else {}
    for e in todo:
        if e == "E3":
            E3(cfg.get("max_features", "sqrt"))
        elif e == "E4":
            E4(cfg.get("max_features", "sqrt"), cfg.get("max_depth"),
               cfg.get("min_samples_leaf", 1), cfg.get("criterion", "gini"))
        else:
            globals()[e]()
