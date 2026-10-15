"""
UNSW-NB15 assignment 2: kNN vs classification tree.

Usage:
  python nb15.py profile                  # data quality report
  python nb15.py tune  --sample 40000     # hyperparameter search
  python nb15.py final --sample 0         # 5-fold CV with chosen configs (0 = full data)

Final configuration, chosen via ablation on a 40k stratified subsample
(see ablate.py) and re-verified on the full 257673-row dataset:
  tree : criterion=gini, max_depth=20, min_samples_leaf=1,
         class_weight="balanced"
         -> balanced over None: -0.05 accuracy, +0.04 balanced accuracy,
            Backdoor recall 0.086->0.238, Analysis 0.089->0.226,
            Worms 0.148->0.222. Chosen because false negatives on rare
            attack classes are the costlier error for an intrusion
            detection system; macro-F1 was flat (-0.005) so this was a
            genuine precision/recall trade, not a free win.
  knn  : k=7, weights="distance", scaling=log1p then min-max fit on
         training folds only, correlation-drop threshold=0.98
         -> lowering the threshold to 0.95/0.90 to also remove the
            moderately-correlated ct_* cluster was tested and made
            balanced accuracy worse (0.467 -> 0.455 at 0.95), so only
            near-duplicate pairs (r>0.98) are dropped.
"""
import argparse, json, time, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)

DATA = "networkTraffic.csv"
SEED = 42
CLASSES = {0: "Normal", 1: "Reconnaissance", 2: "Backdoor", 3: "DoS", 4: "Exploits",
           5: "Analysis", 6: "Fuzzers", 7: "Worms", 8: "Shellcode", 9: "Generic"}
NOMINAL = ["proto", "state", "service"]


# ----------------------------------------------------------------- loading
def load(path=DATA):
    df = pd.read_csv(path, na_values=["?"], low_memory=False)
    df = df.drop(columns=["id"])                 # unique key, zero predictive value
    df["service"] = df["service"].fillna("none")  # structural missingness, own level
    return df


def collapse_rare(s, min_frac=0.001, other="other"):
    keep = s.value_counts(normalize=True)
    keep = set(keep[keep >= min_frac].index)
    return s.where(s.isin(keep), other)


# ----------------------------------------------------------------- profiling
def profile(df):
    print(f"rows={len(df)}  descriptive features={df.shape[1]-1}")

    print("\n[1] class distribution")
    vc = df.attack_cat.value_counts().sort_index()
    for k, v in vc.items():
        print(f"    {k}  {CLASSES[k]:<15} {v:>7d}  {100*v/len(df):6.3f}%")
    print(f"    imbalance ratio (max/min) = {vc.max()/vc.min():.1f}")

    print("\n[2] missing values ('?')")
    m = df.isna().sum()
    m = m[m > 0]
    print("    none" if m.empty else
          "\n".join(f"    {c:<12} {v:>7d}  {100*v/len(df):5.2f}%" for c, v in m.items()))

    print("\n[3] nominal cardinality")
    for c in NOMINAL:
        print(f"    {c:<10} {df[c].nunique():>4} levels   "
              f"top5={list(df[c].value_counts().head(5).index)}")

    num = df.select_dtypes(include=[np.number]).drop(columns=["attack_cat"])
    print("\n[4] numeric range spread")
    rng = (num.max() - num.min()).sort_values()
    print(f"    smallest range: {rng.index[0]} = {rng.iloc[0]:.4g}")
    print(f"    largest  range: {rng.index[-1]} = {rng.iloc[-1]:.4g}")
    print(f"    ratio = {rng.iloc[-1]/max(rng.iloc[0],1e-12):.3g}")

    print("\n[5] correlated numeric pairs |r| > 0.95")
    C = num.corr().abs().values
    cols = list(num.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if C[i, j] > 0.95:
                print(f"    {cols[i]:<18} {cols[j]:<18} {C[i,j]:.4f}")

    print("\n[6] skewness / outliers (top 10 by |skew|)")
    sk = num.skew().abs().sort_values(ascending=False).head(10)
    for c in sk.index:
        print(f"    {c:<18} skew={num[c].skew():>9.2f}  p99={num[c].quantile(.99):>12.4g}  "
              f"max={num[c].max():>12.4g}")


def redundant(num_df, thresh=0.98):
    """Greedy: drop the later member of any pair above thresh."""
    C = num_df.corr().abs()
    cols = list(num_df.columns)
    drop = set()
    for i in range(len(cols)):
        if cols[i] in drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] not in drop and C.iloc[i, j] > thresh:
                drop.add(cols[j])
    return sorted(drop)


# ----------------------------------------------------------------- encoding
def build_matrices(df, corr_thresh=0.98, rare_frac=0.001, verbose=True):
    """Returns (X_tree, X_knn, y, meta). Both are plain numpy, already encoded."""
    y = df["attack_cat"].to_numpy()
    X = df.drop(columns=["attack_cat"]).copy()

    for c in NOMINAL:
        X[c] = collapse_rare(X[c], rare_frac)

    num_cols = [c for c in X.columns if c not in NOMINAL]

    # --- tree view: one-hot nominals, numerics untouched (no scaling, no transform)
    Xt = pd.get_dummies(X, columns=NOMINAL, dtype=np.float32)

    # --- knn view: drop redundant numerics, log1p heavy tails, then scale
    drop = redundant(X[num_cols], corr_thresh)
    if verbose:
        print(f"  knn drops {len(drop)} redundant numerics: {drop}")
        print(f"  proto collapsed to {X['proto'].nunique()} levels")
    keep = [c for c in num_cols if c not in drop]
    Xk = pd.get_dummies(X[keep + NOMINAL], columns=NOMINAL, dtype=np.float32)

    meta = dict(dropped=drop, n_tree=Xt.shape[1], n_knn=Xk.shape[1],
                num_keep=keep, onehot=[c for c in Xk.columns if c not in keep])
    return Xt.to_numpy(np.float32), Xk, y, meta


def scale_knn(Xtr, Xte, num_keep, mode="log_minmax"):
    """Fit on train only. one-hot columns are left at 0/1 so every feature spans [0,1]."""
    from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
    tr, te = Xtr.copy(), Xte.copy()
    A, B = tr[num_keep].to_numpy(np.float64), te[num_keep].to_numpy(np.float64)

    if mode == "log_minmax":
        A, B = np.log1p(np.clip(A, 0, None)), np.log1p(np.clip(B, 0, None))
        sc = MinMaxScaler().fit(A)
    elif mode == "minmax":
        sc = MinMaxScaler().fit(A)
    elif mode == "standard":
        sc = StandardScaler().fit(A)
    elif mode == "quantile":
        sc = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=1000, random_state=SEED).fit(A)
    else:
        raise ValueError(mode)

    tr[num_keep] = np.clip(sc.transform(A), 0, 1) if "minmax" in mode or mode == "quantile" \
        else sc.transform(A)
    te[num_keep] = np.clip(sc.transform(B), 0, 1) if "minmax" in mode or mode == "quantile" \
        else sc.transform(B)
    return tr.to_numpy(np.float32), te.to_numpy(np.float32)


# ----------------------------------------------------------------- evaluation
def score(yt, yp):
    return dict(acc=accuracy_score(yt, yp),
                bal_acc=balanced_accuracy_score(yt, yp),
                f1_macro=f1_score(yt, yp, average="macro", zero_division=0),
                f1_weighted=f1_score(yt, yp, average="weighted", zero_division=0))


def cv_tree(Xt, y, folds=5, **kw):
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    rows, preds, truth = [], [], []
    for tr, te in skf.split(Xt, y):
        m = DecisionTreeClassifier(random_state=SEED, **kw).fit(Xt[tr], y[tr])
        p = m.predict(Xt[te])
        rows.append(score(y[te], p)); preds.append(p); truth.append(y[te])
    return pd.DataFrame(rows), np.concatenate(truth), np.concatenate(preds)


def cv_knn(Xk, y, num_keep, folds=5, scaling="log_minmax", **kw):
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    rows, preds, truth = [], [], []
    for tr, te in skf.split(Xk, y):
        A, B = scale_knn(Xk.iloc[tr], Xk.iloc[te], num_keep, scaling)
        m = KNeighborsClassifier(n_jobs=-1, **kw).fit(A, y[tr])
        p = m.predict(B)
        rows.append(score(y[te], p)); preds.append(p); truth.append(y[te])
    return pd.DataFrame(rows), np.concatenate(truth), np.concatenate(preds)


def summarise(name, res, yt, yp):
    print(f"\n=== {name} ===")
    print(res.agg(["mean", "std"]).T.rename(columns={"mean": "mean", "std": "sd"}).round(4).to_string())
    print("\nper-class recall (pooled over folds):")
    cm = confusion_matrix(yt, yp)
    rec = cm.diagonal() / cm.sum(1)
    for i, r in enumerate(rec):
        print(f"    {CLASSES[i]:<15} n={cm.sum(1)[i]:>6d}  recall={r:.4f}")
    return cm


# ----------------------------------------------------------------- driver
def subsample(df, n):
    if not n or n >= len(df):
        return df
    keep, _ = train_test_split(df, train_size=n, stratify=df.attack_cat, random_state=SEED)
    return keep.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["profile", "tune", "final"])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    df = load(a.data)
    if a.mode == "profile":
        profile(df); return

    df = subsample(df, a.sample)
    print(f"working set: {len(df)} rows")
    Xt, Xk, y, meta = build_matrices(df)
    print(f"  tree matrix {Xt.shape}, knn matrix {Xk.shape}")
    nk = meta["num_keep"]

    if a.mode == "tune":
        print("\n--- tree grid ---")
        best_t, out = None, []
        for crit in ["gini", "entropy"]:
            for depth in [8, 12, 16, 20, None]:
                for leaf in [1, 5, 20]:
                    t0 = time.time()
                    r, _, _ = cv_tree(Xt, y, a.folds, criterion=crit,
                                      max_depth=depth, min_samples_leaf=leaf)
                    m = r.mean()
                    out.append((crit, depth, leaf, m.acc, m.f1_macro, time.time() - t0))
                    print(f"    {crit:<8} depth={str(depth):<5} leaf={leaf:<3} "
                          f"acc={m.acc:.4f} f1m={m.f1_macro:.4f} ({time.time()-t0:.1f}s)")
        out.sort(key=lambda r: -r[4])
        print(f"  best by macro-F1: {out[0]}")

        print("\n--- knn scaling comparison (k=5) ---")
        for sc in ["minmax", "standard", "log_minmax", "quantile"]:
            t0 = time.time()
            r, _, _ = cv_knn(Xk, y, nk, a.folds, scaling=sc, n_neighbors=5)
            m = r.mean()
            print(f"    {sc:<12} acc={m.acc:.4f} f1m={m.f1_macro:.4f} ({time.time()-t0:.1f}s)")

        print("\n--- knn k / weights / metric grid (log_minmax) ---")
        for k in [1, 3, 5, 7, 11, 15, 25]:
            for w in ["uniform", "distance"]:
                t0 = time.time()
                r, _, _ = cv_knn(Xk, y, nk, a.folds, n_neighbors=k, weights=w)
                m = r.mean()
                print(f"    k={k:<3} {w:<9} acc={m.acc:.4f} f1m={m.f1_macro:.4f} "
                      f"({time.time()-t0:.1f}s)")

    else:  # final
        rt, yt_t, yp_t = cv_tree(Xt, y, a.folds, criterion="gini",
                                 max_depth=20, min_samples_leaf=1,
                                 class_weight="balanced")
        summarise("Classification tree", rt, yt_t, yp_t)
        rk, yt_k, yp_k = cv_knn(Xk, y, nk, a.folds, n_neighbors=7, weights="distance")
        summarise("k-nearest neighbours", rk, yt_k, yp_k)

        from scipy.stats import ttest_rel, wilcoxon
        print("\n=== paired comparison over folds (macro-F1) ===")
        t, p = ttest_rel(rt.f1_macro, rk.f1_macro)
        print(f"    tree {rt.f1_macro.mean():.4f} vs knn {rk.f1_macro.mean():.4f}   "
              f"paired t={t:.3f}, p={p:.4g}")


if __name__ == "__main__":
    main()
