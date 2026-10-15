"""Shared data handling. Pre-processing is identical to assignment 2 (nb15.py)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from nb15 import load, build_matrices, collapse_rare, NOMINAL, CLASSES

SEED = 42
TUNE_N = 40000      # same stratified subsample assignment 2 used for tuning


def split_tune_eval(y):
    """Disjoint index sets: 40k tuning subsample (identical to assignment 2) and
    the remaining 217,673 rows used only for the final evaluation."""
    idx = np.arange(len(y))
    tune, ev = train_test_split(idx, train_size=TUNE_N, stratify=y, random_state=SEED)
    return np.sort(tune), np.sort(ev)


def ordinal_matrix(df):
    """Alternative encoding: nominal levels (after rare collapse) as integer codes."""
    X = df.drop(columns=["attack_cat"]).copy()
    for c in NOMINAL:
        X[c] = collapse_rare(X[c]).astype("category").cat.codes
    return X.to_numpy(np.float32), list(X.columns)


def folds(y, k, seed=SEED):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def score(yt, yp):
    return dict(acc=accuracy_score(yt, yp),
                bal_acc=balanced_accuracy_score(yt, yp),
                f1_macro=f1_score(yt, yp, average="macro", zero_division=0),
                f1_weighted=f1_score(yt, yp, average="weighted", zero_division=0))


def per_class(yt, yp):
    p, r, f, n = precision_recall_fscore_support(yt, yp, labels=range(10), zero_division=0)
    return pd.DataFrame(dict(cls=[CLASSES[i] for i in range(10)], n=n,
                             precision=p, recall=r, f1=f))
