#!/usr/bin/env python3
"""
Live inference: fetch recent 5-min candles, compute features,
run the trained ensemble, emit forecast JSON.
"""
import json
import math
import pickle
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import lightgbm as lgb

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Mozilla/5.0 (research project)"}


def get_candles():
    # pull 4 days of 5-min bars in 1-day chunks (Coinbase caps at 300/request)
    from urllib.parse import quote
    end = datetime.now(timezone.utc).replace(microsecond=0)
    bars = []
    for d in range(4):
        de = end - timedelta(days=d)
        ds = de - timedelta(days=1)
        url = ("https://api.exchange.coinbase.com/products/XRP-USD/candles"
               f"?granularity=300&start={quote(ds.isoformat())}&end={quote(de.isoformat())}")
        req = urllib.request.Request(url, headers=UA)
        bars.extend(json.load(urllib.request.urlopen(req, timeout=30)))
    out = []
    for b in sorted(bars):
        out.append({"t": datetime.fromtimestamp(b[0], timezone.utc),
                    "o": b[3], "h": b[2], "l": b[1], "c": b[4], "v": b[5]})
    return out


def main():
    candles = get_candles()
    # need enough history for features (4h RSI etc.)
    # fetch more if needed
    if len(candles) < 400:
        print(json.dumps({"error": "not enough candles"}))
        return

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "features", f"{BASE}/scripts/features.py")
    featmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(featmod)
    feats = featmod.build_features(candles)
    # use last COMPLETE bar (not the forming one)
    f = feats[-2]
    names = json.load(open(f"{BASE}/data/feat_meta.json"))["names"]
    x = np.array([[f[k] for k in names]])

    gbm = lgb.Booster(model_file=f"{BASE}/models/lgbm.txt")
    art = pickle.load(open(f"{BASE}/models/artifacts.pkl", "rb"))
    p_gbm = gbm.predict(x)[0]
    z = (x - art["mu"]) / art["sd"]
    p_lr = 1 / (1 + np.exp(-(z @ art["lr_coef"].T + art["lr_intercept"])))[0][0]
    blend = 0.5 * p_gbm + 0.5 * p_lr
    # isotonic via interpolation
    cal = float(np.interp(blend, art["iso_x"], art["iso_y"]))
    p = 0.5 + (cal - 0.5) * art["shrink"]

    now = datetime.now(timezone.utc)
    # current 15-min window
    q = (now.minute // 15) * 15
    wopen = now.replace(minute=q, second=0, microsecond=0)
    wclose = wopen + timedelta(minutes=15)

    out = {
        "asof": now.isoformat(),
        "window_open": wopen.isoformat(),
        "window_close": wclose.isoformat(),
        "p_up": round(p, 4),
        "p_down": round(1 - p, 4),
        "components": {"lgbm": round(float(p_gbm), 4), "logreg": round(float(p_lr), 4)},
        "model": "lgbm+logreg blend, isotonic-calibrated, shrunk (OOS AUC 0.55)",
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
