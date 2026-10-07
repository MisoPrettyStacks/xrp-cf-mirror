#!/usr/bin/env python3
"""
The key experiment: does fading the prior 15-minute move predict the
Kalshi outcome? Features from Coinbase 5-min candles (known at window
open), labels from the Kalshi truth series.
"""
import csv
import json
import math
from datetime import datetime, timezone

# load candles
candles = []
with open("/tmp/cf-mirror/data/candles_5m.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        candles.append((datetime.fromisoformat(r["time"]),
                        float(r["open"]), float(r["high"]),
                        float(r["low"]), float(r["close"]),
                        float(r["volume"])))
candles.sort()
print(f"candles: {len(candles)}, {candles[0][0]} -> {candles[-1][0]}")
cmap = {c[0]: c for c in candles}

# EWMA volatility (lambda=0.94) over 5-min log returns
rets = []
for i in range(1, len(candles)):
    rets.append(math.log(candles[i][4] / candles[i-1][4]))
var = sum(r*r for r in rets[:288]) / 288
vol = {candles[288][0]: math.sqrt(var)}
for i in range(289, len(candles)):
    var = 0.94 * var + 0.06 * rets[i-1] ** 2
    vol[candles[i][0]] = math.sqrt(var)

# load Kalshi windows
recs = []
with open("/tmp/cf-mirror/data/kalshi_history.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if r["result"] in ("yes", "no"):
                recs.append(r)

from datetime import timedelta
tested = 0
fade_wins = 0
mom_wins = 0
rows = []
for r in recs:
    wo = datetime.fromisoformat(r["window_open"])
    # need the 3 five-min bars ending at window open
    bars = [wo - timedelta(minutes=15), wo - timedelta(minutes=10), wo - timedelta(minutes=5)]
    if not all(b in cmap for b in bars) or wo not in vol:
        continue
    p0 = cmap[bars[0]][1]  # open of first bar
    p1 = cmap[wo - timedelta(minutes=5)][4]  # close of last bar
    lr = math.log(p1 / p0)
    z = lr / (vol[wo] * math.sqrt(3))  # vol-scaled 15-min move
    y = 1 if r["result"] == "yes" else 0
    # fade: predict down if z > 0
    fade_pred = 0 if z > 0 else 1
    mom_pred = 1 if z > 0 else 0
    fade_wins += (fade_pred == y)
    mom_wins += (mom_pred == y)
    tested += 1
    rows.append((abs(z), fade_pred == y))

print(f"\ntested windows: {tested}")
print(f"FADE prior 15m move: {fade_wins/tested:.4f}")
print(f"FOLLOW prior 15m move: {mom_wins/tested:.4f}")
print(f"(naive always-up: 0.514)")

# by |z| magnitude: does stronger prior move -> stronger reversal?
rows.sort()
for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)]:
    sub = [w for z, w in rows if lo <= z < hi]
    if sub:
        print(f"  |z| in [{lo},{hi}): fade hit {sum(sub)/len(sub):.4f} (n={len(sub)})")
