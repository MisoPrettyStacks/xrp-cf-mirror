#!/usr/bin/env python3
"""
Feature engineering for 15-min XRP direction.
Input: 5-min OHLCV candles. Output: feature matrix aligned to
Kalshi window opens (features known at window open; label = outcome).
"""
import csv
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np


def load_candles(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append({
                "t": datetime.fromisoformat(r["time"]),
                "o": float(r["open"]), "h": float(r["high"]),
                "l": float(r["low"]), "c": float(r["close"]),
                "v": float(r["volume"]),
            })
    out.sort(key=lambda x: x["t"])
    return out


def ema(vals, span):
    a = 2 / (span + 1)
    e = vals[0]
    out = [e]
    for v in vals[1:]:
        e = a * v + (1 - a) * e
        out.append(e)
    return out


def rsi(closes, n=14):
    out = [np.nan] * len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    ag = sum(gains[:n]) / n; al = sum(losses[:n]) / n
    for i in range(n, len(closes)):
        ag = (ag * (n - 1) + gains[i - 1]) / n
        al = (al * (n - 1) + losses[i - 1]) / n
        out[i] = 100 - 100 / (1 + ag / al) if al > 0 else 100.0
    return out


def boll_pctb(closes, n=20, k=2):
    out = [np.nan] * len(closes)
    for i in range(n - 1, len(closes)):
        w = closes[i - n + 1:i + 1]
        m, s = np.mean(w), np.std(w)
        out[i] = (closes[i] - (m - k * s)) / (2 * k * s) if s > 0 else 0.5
    return out


def build_features(candles):
    n = len(candles)
    closes = np.array([c["c"] for c in candles])
    highs = np.array([c["h"] for c in candles])
    lows = np.array([c["l"] for c in candles])
    vols = np.array([c["v"] for c in candles])
    opens = np.array([c["o"] for c in candles])

    lr = np.zeros(n); lr[1:] = np.log(closes[1:] / closes[:-1])

    # EWMA vol (lambda=0.94) on 5-min returns
    var = np.zeros(n)
    var[0] = np.mean(lr[1:289] ** 2) if n > 289 else np.mean(lr[1:] ** 2)
    for i in range(1, n):
        var[i] = 0.94 * var[i - 1] + 0.06 * lr[i] ** 2
    sig = np.sqrt(var)

    # Parkinson vol (20-bar)
    pk = np.zeros(n)
    for i in range(20, n):
        pk[i] = math.sqrt(sum((math.log(highs[j] / lows[j]) ** 2)
                              for j in range(i - 19, i + 1)) / (20 * 4 * math.log(2)))

    # volume z-score vs 1-day median
    vol_z = np.zeros(n)
    for i in range(288, n):
        med = np.median(vols[i - 287:i + 1])
        vol_z[i] = math.log(vols[i] / med) if med > 0 and vols[i] > 0 else 0

    rsi14 = rsi(closes.tolist())
    pctb = boll_pctb(closes.tolist())

    # 4h resample (48 bars) for regime
    c4 = closes[::48]
    rsi4 = rsi(c4.tolist())
    pctb4 = boll_pctb(c4.tolist())
    rsi4_full = np.full(n, np.nan); pctb4_full = np.full(n, np.nan)
    for j, v in enumerate(rsi4):
        rsi4_full[j * 48:(j + 1) * 48] = v
    for j, v in enumerate(pctb4):
        pctb4_full[j * 48:(j + 1) * 48] = v

    feats = []
    for i in range(n):
        t = candles[i]["t"]
        f = {}
        # vol-scaled returns
        for k in (1, 3, 12, 36, 144, 288):
            if i >= k and sig[i] > 0:
                f[f"z{k}"] = math.log(closes[i] / closes[i - k]) / (sig[i] * math.sqrt(k))
            else:
                f[f"z{k}"] = 0.0
        f["sign1"] = float(np.sign(lr[i])) if i > 0 else 0.0
        f["rsi"] = (rsi14[i] - 50) / 50 if not np.isnan(rsi14[i]) else 0.0
        f["pctb"] = pctb[i] - 0.5 if not np.isnan(pctb[i]) else 0.0
        f["rsi4"] = (rsi4_full[i] - 50) / 50 if not np.isnan(rsi4_full[i]) else 0.0
        f["pctb4"] = pctb4_full[i] - 0.5 if not np.isnan(pctb4_full[i]) else 0.0
        f["park"] = pk[i] / (sig[i] + 1e-12) if sig[i] > 0 else 1.0
        f["volz"] = vol_z[i]
        rng = highs[i] - lows[i]
        body = abs(closes[i] - opens[i])
        f["body_ratio"] = body / rng if rng > 0 else 0.0
        f["wick_up"] = (highs[i] - max(opens[i], closes[i])) / rng if rng > 0 else 0.0
        f["wick_dn"] = (min(opens[i], closes[i]) - lows[i]) / rng if rng > 0 else 0.0
        # large-move reversal interaction: fade only big moves
        z3 = f["z3"]
        f["fade_big"] = -np.sign(z3) * max(0.0, abs(z3) - 1.5)
        # clock
        f["qh"] = (t.minute // 15) / 3.0 - 0.5
        f["hod"] = t.hour / 23.0 - 0.5
        f["dow"] = t.weekday() / 6.0 - 0.5
        f["weekend"] = 1.0 if t.weekday() >= 5 else 0.0
        feats.append(f)
    return feats


def align_to_windows(feats, candles, kh_path):
    """Join features (at window open) with Kalshi outcomes."""
    tmap = {c["t"]: i for i, c in enumerate(candles)}
    X, y, meta = [], [], []
    with open(kh_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["result"] not in ("yes", "no"):
                continue
            wo = datetime.fromisoformat(r["window_open"])
            # Coinbase candle timestamps are the START of each 5-min bucket.
            # The last fully-complete bar at window open is the one starting
            # 5 min earlier; using the bar AT wo would leak the window's
            # first 5 minutes into the features.
            key = wo - timedelta(minutes=5)
            if key not in tmap:
                continue
            i = tmap[key]
            if i < 300:  # warmup
                continue
            X.append(feats[i])
            y.append(1 if r["result"] == "yes" else 0)
            meta.append({"window_open": r["window_open"], "strike": r["strike"]})
    return X, np.array(y), meta


if __name__ == "__main__":
    import sys, os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candles = load_candles(f"{base}/data/candles_5m.csv")
    print(f"candles: {len(candles)}")
    feats = build_features(candles)
    X, y, meta = align_to_windows(feats, candles, f"{base}/data/kalshi_history.jsonl")
    print(f"aligned windows: {len(X)}, up-rate: {y.mean():.3f}")
    names = sorted(X[0].keys())
    print(f"features ({len(names)}): {names}")
    np.save(f"{base}/data/X.npy", np.array([[d[k] for k in names] for d in X]))
    np.save(f"{base}/data/y.npy", y)
    json.dump({"names": names, "meta": meta}, open(f"{base}/data/feat_meta.json", "w"))
    print("saved X.npy, y.npy, feat_meta.json")
