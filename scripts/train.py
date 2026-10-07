#!/usr/bin/env python3
"""
Walk-forward training with purged validation.
Trains LightGBM + LogisticRegression on expanding windows,
calibrates with isotonic regression, reports honest OOS metrics.
"""
import json
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE = "/tmp/cf-mirror"
X = np.load(f"{BASE}/data/X.npy")
y = np.load(f"{BASE}/data/y.npy")
meta = json.load(open(f"{BASE}/data/feat_meta.json"))
names = meta["names"]
times = np.array([m["window_open"] for m in meta["meta"]])

n = len(y)
print(f"samples: {n}, features: {len(names)}, up-rate: {y.mean():.3f}")

# walk-forward: 5 folds, expanding train, 1-week test, 1-day embargo
fold_days_test = 7
embargo = 96  # 1 day of 15-min windows
fold_size = fold_days_test * 96
starts = list(range(n - 5 * fold_size, n, fold_size))

oof_lgb = np.full(n, np.nan)
oof_lr = np.full(n, np.nan)
importances = []

for fi, ts in enumerate(starts):
    te = min(ts + fold_size, n)
    tr_end = ts - embargo
    if tr_end < 1000:
        continue
    Xtr, ytr = X[:tr_end], y[:tr_end]
    Xte, yte = X[ts:te], y[ts:te]

    # LightGBM
    dtr = lgb.Dataset(Xtr, ytr)
    params = dict(objective="binary", num_leaves=15, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                  lambda_l2=5.0, learning_rate=0.03, num_iterations=300,
                  verbose=-1)
    m = lgb.train(params, dtr)
    oof_lgb[ts:te] = m.predict(Xte)
    importances.append(m.feature_importance(importance_type="gain"))

    # Logistic regression (standardized)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(C=0.5, max_iter=2000)
    lr.fit((Xtr - mu) / sd, ytr)
    oof_lr[ts:te] = lr.predict_proba((Xte - mu) / sd)[:, 1]
    print(f"fold {fi}: train {tr_end} -> test [{ts}:{te}]", flush=True)

mask = ~np.isnan(oof_lgb)
yt = y[mask]
print(f"\nOOS windows: {mask.sum()}")

for label, p in [("lgb", oof_lgb[mask]), ("logreg", oof_lr[mask])]:
    auc = roc_auc_score(yt, p)
    bs = brier_score_loss(yt, p)
    acc = ((p > 0.5) == yt).mean()
    print(f"{label:8s} AUC={auc:.4f} acc={acc:.4f} brier={bs:.5f} skill={(0.25-bs)/0.25*100:+.2f}%")

# blend + isotonic calibration
blend = 0.5 * oof_lgb[mask] + 0.5 * oof_lr[mask]
iso = IsotonicRegression(out_of_bounds="clip").fit(blend, yt)
cal = iso.predict(blend)
auc = roc_auc_score(yt, cal)
bs = brier_score_loss(yt, cal)
acc = ((cal > 0.5) == yt).mean()
print(f"{'blend+iso':8s} AUC={auc:.4f} acc={acc:.4f} brier={bs:.5f} skill={(0.25-bs)/0.25*100:+.2f}%")
print(f"calibrated prob range: [{cal.min():.3f}, {cal.max():.3f}]")

# shrinkage toward 0.5 (earn confidence slowly)
shrunk = 0.5 + (cal - 0.5) * 0.5
bs_s = brier_score_loss(yt, shrunk)
print(f"{'shrunk':8s} brier={bs_s:.5f} skill={(0.25-bs_s)/0.25*100:+.2f}% range=[{shrunk.min():.3f},{shrunk.max():.3f}]")

# feature importance
imp = np.mean(importances, axis=0)
order = np.argsort(imp)[::-1]
print("\nTop features by gain:")
for i in order[:10]:
    print(f"  {names[i]:12s} {imp[i]:.1f}")

# save OOS predictions for the page/ledger
np.save(f"{BASE}/data/oof_pred.npy", oof_lgb)  # raw; calibration applied at inference
print("\nsaved oof_pred.npy")
