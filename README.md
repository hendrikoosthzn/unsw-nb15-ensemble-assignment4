# Assignment 4, Option 1: random forest on UNSW-NB15

Files
- nb15.py      assignment 2 pipeline, unchanged (pre-processing, kNN scaling)
- common.py    tuning/evaluation split, folds, metrics
- forest.py    random forest (bootstrap, feature subspace, class weighting, soft voting, OOB, diversity)
- tune.py      E0-E4 on the 40k tuning subsample (same subsample as assignment 2), 5-fold CV
- evaluate.py  10-fold CV on the other 217,673 rows: CART, kNN and the forest(s) in chosen.json["final"]
- chosen.json  settings carried from one experiment to the next

Run order (put networkTraffic.csv in this folder)
    python3 tune.py E0 E1 E2      # encoding, ensemble size, max_features
    # edit chosen.json: max_features
    python3 tune.py E3            # depth x min_samples_leaf (+ entropy)
    # edit chosen.json: max_depth, min_samples_leaf, criterion
    python3 tune.py E4            # imbalance remedies
    # edit chosen.json["final"]
    python3 evaluate.py

Selection rule: smallest/simplest setting whose mean macro-F1 is within one
standard error (sd / sqrt(5)) of the best.

Post-hoc and report
    python3 evaluate.py posthoc.json posthoc   # forests without depth limit
    python3 analyse.py                         # tables, tests, figures -> report/figures, results/summary.txt
    cd report && latexmk -pdf report.tex
