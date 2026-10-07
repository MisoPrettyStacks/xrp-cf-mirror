#!/usr/bin/env python3
"""
Score the live ledger against Kalshi settlements.
Writes stats.json with rolling accuracy, AUC, Brier skill.
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Mozilla/5.0 (research project)"}


def main():
    # load ledger (forecasts made by the runner)
    ledger = []
    lp = os.path.join(BASE, "ledger.jsonl")
    if os.path.exists(lp):
        with open(lp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ledger.append(json.loads(line))
    # dedupe by window (runner may forecast same window multiple times; keep latest)
    by_win = {}
    for r in ledger:
        by_win[r["window_open"]] = r
    ledger = sorted(by_win.values(), key=lambda r: r["window_open"])

    # load truth (Kalshi history from main branch)
    truth = {}
    url = ("https://raw.githubusercontent.com/MisoPrettyStacks/"
           "xrp-cf-mirror/main/data/kalshi_history.jsonl")
    req = urllib.request.Request(url, headers=UA)
    for line in urllib.request.urlopen(req, timeout=60):
        line = line.decode("utf-8").strip()
        if line:
            r = json.loads(line)
            if r["result"] in ("yes", "no"):
                truth[r["window_open"]] = 1 if r["result"] == "yes" else 0

    # match
    ys, ps, wins = [], [], []
    for r in ledger:
        if r["window_open"] in truth:
            ys.append(truth[r["window_open"]])
            ps.append(r["p_up"])
            wins.append(r["window_open"])

    import numpy as np
    ys = np.array(ys); ps = np.array(ps)
    stats = {"scored": len(ys), "updated": datetime.now(timezone.utc).isoformat()}

    def metrics(y, p):
        from sklearn.metrics import roc_auc_score, brier_score_loss
        acc = float(((p > 0.5) == y).mean())
        auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else 0.5
        bs = float(brier_score_loss(y, p))
        return {"n": len(y), "accuracy": round(acc, 4),
                "auc": round(auc, 4), "brier": round(bs, 5),
                "brier_skill_pct": round((0.25 - bs) / 0.25 * 100, 2)}

    if len(ys) >= 20:
        stats["last_100"] = metrics(ys[-100:], ps[-100:])
    if len(ys) >= 100:
        stats["last_500"] = metrics(ys[-500:], ps[-500:])
    if len(ys) >= 20:
        stats["all_time"] = metrics(ys, ps)
        # calibration bins
        bins = []
        for lo in range(40, 60, 2):
            m = (ps >= lo/100) & (ps < (lo+2)/100)
            if m.sum() >= 5:
                bins.append({"prob": (lo+1)/100,
                             "actual": round(float(ys[m].mean()), 3),
                             "n": int(m.sum())})
        stats["calibration"] = bins

    with open(os.path.join(BASE, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1)[:800])


if __name__ == "__main__":
    main()
