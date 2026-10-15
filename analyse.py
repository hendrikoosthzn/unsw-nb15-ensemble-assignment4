"""
Tables, statistical tests and figures for the report.
  python3 analyse.py            (reads results/, writes report/figures/ and results/summary.txt)
"""
import os, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon

R, FIG = "results", "report/figures"
os.makedirs(FIG, exist_ok=True)
CLS = ["Normal", "Reconnaissance", "Backdoor", "DoS", "Exploits", "Analysis",
       "Fuzzers", "Worms", "Shellcode", "Generic"]
SHORT = ["Norm", "Recon", "Backd", "DoS", "Expl", "Anal", "Fuzz", "Worm", "Shell", "Gen"]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"cart": "CART", "knn": "kNN", "rf": "RF", "rf_bal": "RF-B",
         "rf_full": "RF (no depth limit)", "rf_bal_full": "RF-B (no depth limit)"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.4, "lines.markersize": 4, "pdf.fonttype": 42})
W1 = 3.45     # IEEE column width in inches
out = open(f"{R}/summary.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.write(s + "\n")


def ms(d, col):
    return d.groupby(d.columns[-1])[col].agg(["mean", "std"])


# ------------------------------------------------------------------ tuning
say("== E0 encoding ==")
e0 = pd.read_csv(f"{R}/E0_encoding.csv")
say(e0.groupby("encoding")[["f1_macro", "bal_acc", "acc"]].agg(["mean", "std"]).round(4))

say("\n== E1 size ==")
e1 = pd.read_csv(f"{R}/E1_size.csv")
g1 = e1.groupby(["max_features", "trees"])["f1_macro"].agg(["mean", "std"]).reset_index()
say(g1.pivot(index="trees", columns="max_features", values="mean").round(4))
fig, ax = plt.subplots(figsize=(W1, 2.0))
for i, (mf, lab) in enumerate([("sqrt", "8 features per split (square root)"),
                               ("1.0", "64 features per split (bagging)")]):
    d = g1[g1.max_features == mf]
    ax.plot(d.trees, d["mean"], marker="o", color=COL[i], label=lab)
    ax.fill_between(d.trees, d["mean"] - d["std"], d["mean"] + d["std"], color=COL[i], alpha=.12, lw=0)
ax.set_xscale("log"); ax.set_xlabel("Number of trees"); ax.set_ylabel("Macro F1")
ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(f"{FIG}/size.pdf"); plt.close()

say("\n== E2 max_features ==")
e2 = pd.read_csv(f"{R}/E2_max_features.csv")
g2 = e2.groupby("max_features")[["f1_macro", "bal_acc", "acc", "strength", "disagree", "fit_s"]] \
       .agg(["mean", "std"]).sort_index()
say(g2.round(4))
fig, (a, b) = plt.subplots(1, 2, figsize=(W1, 1.9))
x = g2.index.values
a.plot(x, g2[("f1_macro", "mean")], marker="o", color=COL[0], label="Forest")
a.plot(x, g2[("strength", "mean")], marker="s", color=COL[1], label="Single tree (mean)")
a.set_xscale("log", base=2); a.set_xticks(x); a.set_xticklabels(x)
a.set_xlabel("Features per split"); a.set_ylabel("Macro F1"); a.set_ylim(0.44, 0.57); a.legend(frameon=False, loc="upper left", fontsize=6)
a.set_title("(a) Strength", loc="left")
b.plot(x, g2[("disagree", "mean")], marker="o", color=COL[2])
b.set_xscale("log", base=2); b.set_xticks(x); b.set_xticklabels(x)
b.set_xlabel("Features per split"); b.set_ylabel("Pairwise disagreement")
b.set_title("(b) Diversity", loc="left")
fig.tight_layout(); fig.savefig(f"{FIG}/max_features.pdf"); plt.close()

say("\n== E3 complexity ==")
e3 = pd.read_csv(f"{R}/E3_complexity.csv", dtype={"max_depth": str}, keep_default_na=False)
e3["max_depth"] = e3.max_depth.replace({"None": "none", "": "none"}).str.replace(".0", "", regex=False)
g3 = e3.groupby(["criterion", "max_depth", "min_samples_leaf"])[["f1_macro", "bal_acc", "acc", "leaves", "fit_s"]] \
       .agg(["mean", "std"])
say(g3.round(4))
best = g3[("f1_macro", "mean")].idxmax()
b_m, b_s = g3.loc[best, ("f1_macro", "mean")], g3.loc[best, ("f1_macro", "std")]
say(f"best {best} {b_m:.4f} sd {b_s:.4f} se {b_s/np.sqrt(5):.4f} threshold {b_m - b_s/np.sqrt(5):.4f}")
say("within 1 SE:", list(g3[g3[("f1_macro", "mean")] >= b_m - b_s / np.sqrt(5)].index))
order = ["8", "12", "16", "20", "24", "none"]
fig, ax = plt.subplots(figsize=(W1, 2.0))
for i, (crit, leaf, lab) in enumerate([("gini", 1, "Gini, leaf 1"), ("gini", 3, "Gini, leaf 3"),
                                       ("gini", 10, "Gini, leaf 10"), ("entropy", 1, "Entropy, leaf 1")]):
    v = [g3.loc[(crit, d, leaf), ("f1_macro", "mean")] for d in order]
    ax.plot(range(6), v, marker="o", color=COL[i], label=lab)
ax.set_xticks(range(6)); ax.set_xticklabels(["8", "12", "16", "20", "24", "None"])
ax.set_xlabel("Maximum depth"); ax.set_ylabel("Macro F1"); ax.set_ylim(0.40, 0.54)
ax.legend(frameon=False, loc="lower right", ncol=2); fig.tight_layout(); fig.savefig(f"{FIG}/depth.pdf"); plt.close()

say("\n== E4 imbalance ==")
e4 = pd.read_csv(f"{R}/E4_imbalance.csv")
say(e4.groupby("remedy", sort=False)[["f1_macro", "bal_acc", "acc", "f1_weighted"]].agg(["mean", "std"]).round(4))
e4c = pd.read_csv(f"{R}/E4_imbalance_perclass.csv")
g4 = e4c.groupby(["remedy", "cls"], sort=False)[["recall", "precision", "f1"]].mean()
say(g4.unstack(0).round(3))
rem = [("none", "None"), ("balanced", "Class weights"), ("balanced_subsample", "Per-bootstrap weights"),
       ("balanced_bootstrap", "Balanced bootstrap")]
fig, ax = plt.subplots(figsize=(W1, 2.1))
wd = 0.2
for i, (r, lab) in enumerate(rem):
    v = [g4.loc[(r, c), "recall"] for c in CLS]
    ax.bar(np.arange(10) + (i - 1.5) * wd, v, wd * 0.9, color=COL[i], label=lab)
ax.set_xticks(range(10)); ax.set_xticklabels(SHORT, rotation=45, ha="right")
ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))
fig.tight_layout(); fig.savefig(f"{FIG}/imbalance.pdf"); plt.close()

# ------------------------------------------------------------------ final
say("\n== FINAL ==")
ff = pd.read_csv(f"{R}/final_folds.csv")
pc = pd.read_csv(f"{R}/final_perclass.csv")
cm = dict(np.load(f"{R}/final_confusion.npz"))
if os.path.exists(f"{R}/posthoc_folds.csv"):
    ff = pd.concat([ff, pd.read_csv(f"{R}/posthoc_folds.csv")])
    pc = pd.concat([pc, pd.read_csv(f"{R}/posthoc_perclass.csv")])
    cm.update(dict(np.load(f"{R}/posthoc_confusion.npz")))
models = [m for m in ["cart", "knn", "rf", "rf_bal", "rf_full", "rf_bal_full"] if m in set(ff.model)]
metrics = ["f1_macro", "bal_acc", "acc", "f1_weighted", "fit_s", "pred_s"]
say(ff.groupby("model")[metrics].agg(["mean", "std"]).loc[models].round(4).to_string())

say("\n-- statistical tests (10 paired folds) --")
for met in ["f1_macro", "bal_acc", "acc"]:
    P = ff.pivot(index="fold", columns="model", values=met)
    main = ["cart", "knn", "rf", "rf_bal"]
    st, p = friedmanchisquare(*[P[m] for m in main])
    say(f"{met}: Friedman over {main} chi2={st:.2f} p={p:.2e}  mean ranks "
        + str((-P[main]).rank(axis=1).mean().round(2).to_dict()))
    pairs = list(itertools.combinations(main, 2))
    res = []
    for a_, b_ in pairs:
        w, pv = wilcoxon(P[a_], P[b_])
        res.append((a_, b_, P[a_].mean() - P[b_].mean(), int((P[a_] > P[b_]).sum()), pv))
    # Holm correction over the pairs of this metric
    order_ = np.argsort([r[4] for r in res]); m_ = len(res); adj = [0] * m_; run = 0
    for rank, idx in enumerate(order_):
        run = max(run, min(1, (m_ - rank) * res[idx][4])); adj[idx] = run
    for r, pa in zip(res, adj):
        say(f"   {r[0]:<12} vs {r[1]:<12} diff={r[2]:+.4f} wins={r[3]:>2}/10 p={r[4]:.4f} holm={pa:.4f}")

say("\n-- per-class (mean over folds) --")
gp = pc.groupby(["model", "cls"], sort=False)[["precision", "recall", "f1"]].mean()
for met in ["recall", "precision", "f1"]:
    say(f"[{met}]"); say(gp[met].unstack(0)[models].loc[CLS].round(3).to_string())

fig, ax = plt.subplots(figsize=(W1, 2.1))
show = [m for m in ["cart", "knn", "rf", "rf_bal"] if m in models]
wd = 0.8 / len(show)
for i, m in enumerate(show):
    v = [gp.loc[(m, c), "f1"] for c in CLS]
    ax.bar(np.arange(10) + (i - (len(show) - 1) / 2) * wd, v, wd * 0.9, color=COL[i], label=NAMES[m])
ax.set_xticks(range(10)); ax.set_xticklabels(SHORT, rotation=45, ha="right")
ax.set_ylabel("F1 score"); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
fig.tight_layout(); fig.savefig(f"{FIG}/perclass_f1.pdf"); plt.close()

# confusion matrices (row-normalised) for CART and RF-B
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
for ax, m in zip(axes, ["cart", "rf_bal"]):
    C = cm[m] / cm[m].sum(1, keepdims=True)
    ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if C[i, j] >= 0.005:
                ax.text(j, i, f"{C[i,j]:.2f}"[1:] if C[i, j] < 1 else "1.0", ha="center", va="center",
                        fontsize=5.5, color="white" if C[i, j] > 0.55 else INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(SHORT, rotation=45, ha="right"); ax.set_yticklabels(SHORT)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class"); ax.grid(False)
    ax.set_title(f"({'a' if m=='cart' else 'b'}) {NAMES[m]}", loc="left")
fig.tight_layout(); fig.savefig(f"{FIG}/confusion.pdf"); plt.close()

say("\n-- feature importance (rf_bal, mean decrease in impurity) --")
imp = pd.read_csv(f"{R}/final_importance.csv")
gi = imp[imp.model == "rf_bal"].groupby("feature").importance.mean().sort_values(ascending=False)
say(gi.head(15).round(4).to_string())
top = gi.head(12)[::-1]
fig, ax = plt.subplots(figsize=(W1, 2.4))
ax.barh(range(len(top)), top.values, 0.7, color=COL[0])
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
ax.set_xlabel("Mean decrease in impurity"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/importance.pdf"); plt.close()

# ------------------------------------------------------------------ label conflicts
say("\n== label conflicts (identical feature vectors, different labels) ==")
import sys; sys.path.insert(0, ".")
from common import load, build_matrices, split_tune_eval
from sklearn.metrics import f1_score, accuracy_score
df = load()
feat = list(df.columns.drop("attack_cat"))
g = df.groupby(feat, dropna=False)["attack_cat"]
nun = g.transform("nunique")
maj = g.transform(lambda s: s.value_counts().index[0])
say(f"rows {len(df)}, duplicated feature vectors {df.duplicated(feat, keep=False).sum()}, "
    f"rows in conflicting groups {(nun > 1).sum()}")
say(f"oracle (majority label per vector): acc={accuracy_score(df.attack_cat, maj):.4f} "
    f"f1m={f1_score(df.attack_cat, maj, average='macro'):.4f}")
for c in range(10):
    m = df.attack_cat == c
    say(f"   {CLS[c]:<15} n={m.sum():>6}  in conflict groups {100*(nun[m]>1).mean():5.1f}%  "
        f"oracle recall {(maj[m]==c).mean():.3f}")
out.close()
