"""
Final evaluation on the 217,673 rows that were never used for tuning
(neither in assignment 2 nor here). 10-fold stratified CV; every model sees
identical folds so that fold scores are paired.

Models
  cart : assignment 2 classification tree (gini, depth 20, balanced weights)
  knn  : assignment 2 kNN (k=7, distance weights, log1p + min-max, r>0.98 drop)
  rf_* : random forest configurations from chosen.json

  python3 evaluate.py                         # main comparison, writes results/final_*
  python3 evaluate.py posthoc.json posthoc    # forests only, writes results/posthoc_*
"""
import sys
import json, time, os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from common import (load, build_matrices, split_tune_eval, folds, score, per_class, SEED)
from nb15 import scale_knn
from forest import RandomForest

OUT = "results"
K = 10

df = load()
Xt, Xk, y, meta = build_matrices(df, verbose=False)
_, ev = split_tune_eval(y)
Xt, Xk, y = Xt[ev], Xk.iloc[ev].reset_index(drop=True), y[ev]
F = folds(y, K)
CFG = sys.argv[1] if len(sys.argv) > 1 else "chosen.json"
TAG = sys.argv[2] if len(sys.argv) > 2 else "final"
BASELINES = TAG == "final"
cfgs = json.load(open(CFG))["final"]
feat_names = list(pd.get_dummies(
    df.drop(columns=["attack_cat"]).assign(
        **{c: __import__("nb15").collapse_rare(df[c]) for c in ["proto", "state", "service"]}),
    columns=["proto", "state", "service"]).columns)
print(f"evaluation set {len(y)} rows, {K} folds")

rows, cls, cms, imps = [], [], {}, []


def record(name, k, yt, yp, fit_s, pred_s):
    r = score(yt, yp); r.update(model=name, fold=k, fit_s=fit_s, pred_s=pred_s)
    rows.append(r)
    c = per_class(yt, yp); c["model"] = name; c["fold"] = k; cls.append(c)
    cms[name] = cms.get(name, 0) + confusion_matrix(yt, yp, labels=range(10))
    print(f"  {name:<14} fold {k}  f1m={r['f1_macro']:.4f} acc={r['acc']:.4f} "
          f"bal={r['bal_acc']:.4f}  fit {fit_s:.1f}s pred {pred_s:.1f}s", flush=True)


for k, (tr, te) in enumerate(F):
    if BASELINES:
      t0 = time.time()
      m = DecisionTreeClassifier(criterion="gini", max_depth=20, min_samples_leaf=1,
                               class_weight="balanced", random_state=SEED).fit(Xt[tr], y[tr])
      t1 = time.time(); p = m.predict(Xt[te]); t2 = time.time()
      record("cart", k, y[te], p, t1 - t0, t2 - t1)

    for name, cfg in cfgs.items():
        t0 = time.time()
        m = RandomForest(**cfg, seed=SEED + k).fit(Xt[tr], y[tr])
        t1 = time.time(); p = m.predict(Xt[te]); t2 = time.time()
        record(name, k, y[te], p, t1 - t0, t2 - t1)
        imps.append(pd.DataFrame(dict(model=name, fold=k, feature=feat_names,
                                      importance=m.feature_importances())))

    if BASELINES:
      t0 = time.time()
      A, B = scale_knn(Xk.iloc[tr], Xk.iloc[te], meta["num_keep"], "log_minmax")
      m = KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1).fit(A, y[tr])
      t1 = time.time(); p = m.predict(B); t2 = time.time()
      record("knn", k, y[te], p, t1 - t0, t2 - t1)

    pd.DataFrame(rows).to_csv(f"{OUT}/{TAG}_folds.csv", index=False)
    pd.concat(cls).to_csv(f"{OUT}/{TAG}_perclass.csv", index=False)
    pd.concat(imps).to_csv(f"{OUT}/{TAG}_importance.csv", index=False)
    np.savez(f"{OUT}/{TAG}_confusion.npz", **cms)
print("done")
