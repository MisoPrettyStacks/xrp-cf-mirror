#!/usr/bin/env python3
"""
Finalize: train on ALL data, save model artifacts for live inference.
Includes the leak fix (features from last complete bar before window open).
"""
import json
import os
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X = np.load(f"{BASE}/data/X.npy")
y = np.load(f"{BASE}/data/y.npy")
meta = json.load(open(f"{BASE}/data/feat_meta.json"))
names = meta["names"]

# final models on all data
dtr = lgb.Dataset(X, y)
params = dict(objective="binary", num_leaves=15, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=5.0, learning_rate=0.03, num_iterations=300, verbose=-1)
gbm = lgb.train(params, dtr)
gbm.save_model(f"{BASE}/models/lgbm.txt")

mu, sd = X.mean(0), X.std(0) + 1e-9
lr = LogisticRegression(C=0.5, max_iter=2000)
lr.fit((X - mu) / sd, y)

# isotonic calibrator fit on walk-forward OOF blend (honest)
oof = np.load(f"{BASE}/data/oof_pred.npy")
mask = ~np.isnan(oof)
# recompute logreg OOF quickly for the blend
from sklearn.model_selection import TimeSeriesSplit
lr_oof = np.full(len(y), np.nan)
fold_size = 7 * 96
starts = list(range(len(y) - 5 * fold_size, len(y), fold_size))
for ts in starts:
    te = min(ts + fold_size, len(y))
    tr_end = ts - 96
    if tr_end < 1000:
        continue
    m_, s_ = X[:tr_end].mean(0), X[:tr_end].std(0) + 1e-9
    m2 = LogisticRegression(C=0.5, max_iter=2000)
    m2.fit((X[:tr_end] - m_) / s_, y[:tr_end])
    lr_oof[ts:te] = m2.predict_proba((X[ts:te] - m_) / s_)[:, 1]
blend = 0.5 * oof[mask] + 0.5 * lr_oof[mask]
iso = IsotonicRegression(out_of_bounds="clip").fit(blend, y[mask])

import pickle
artifacts = {
    "names": names, "mu": mu, "sd": sd,
    "lr_coef": lr.coef_, "lr_intercept": lr.intercept_,
    "iso_x": iso.X_thresholds_, "iso_y": iso.y_thresholds_,
    # shrink factor: humility guard
    "shrink": 0.5,
}
with open(f"{BASE}/models/artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

# sanity: in-sample AUC (optimistic) vs OOS
p_gbm = gbm.predict(X)
p_lr = lr.predict_proba((X - mu) / sd)[:, 1]
b = 0.5 * p_gbm + 0.5 * p_lr
c = iso.predict(np.clip(b, 0, 1))
s = 0.5 + (c - 0.5) * 0.5
print(f"final in-sample AUC: {roc_auc_score(y, s):.4f} (optimistic)")
print(f"final prob range: [{s.min():.3f}, {s.max():.3f}]")
print("saved models/lgbm.txt + models/artifacts.pkl")
