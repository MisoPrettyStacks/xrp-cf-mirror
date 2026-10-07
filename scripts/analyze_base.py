#!/usr/bin/env python3
"""First analysis of the Kalshi truth series: base rates, streaks,
strike momentum, and time-of-day patterns. No candles needed."""
import json
from collections import defaultdict
from datetime import datetime, timezone

recs = []
with open("/tmp/cf-mirror/data/kalshi_history.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            if r["result"] in ("yes", "no") and r["strike"]:
                recs.append(r)
recs.sort(key=lambda r: r["window_open"])
print(f"windows: {len(recs)}")

ys = [1 if r["result"] == "yes" else 0 for r in recs]
print(f"base up-rate: {sum(ys)/len(ys):.3f}")

# streaks
max_up = max_dn = cur_up = cur_dn = 0
for y in ys:
    if y:
        cur_up += 1; cur_dn = 0; max_up = max(max_up, cur_up)
    else:
        cur_dn += 1; cur_up = 0; max_dn = max(max_dn, cur_dn)
print(f"longest up streak: {max_up}, longest down streak: {max_dn}")

# strike momentum: does the strike direction predict the outcome?
# strike_up = strike rose vs previous window's strike
prev_strike = None
mom_up = []; mom_dn = []
for r in recs:
    if prev_strike and r["strike"]:
        if r["strike"] > prev_strike:
            mom_up.append(1 if r["result"] == "yes" else 0)
        elif r["strike"] < prev_strike:
            mom_dn.append(1 if r["result"] == "yes" else 0)
    if r["strike"]:
        prev_strike = r["strike"]
print(f"after strike rose:  up-rate={sum(mom_up)/len(mom_up):.3f} (n={len(mom_up)})  <- momentum test")
print(f"after strike fell:  up-rate={sum(mom_dn)/len(mom_dn):.3f} (n={len(mom_dn)})")

# reversal test: fade the strike move
rev_win = sum(1 for i, r in enumerate(recs[1:], 1)
              if recs[i-1]["strike"] and r["strike"]
              and ((r["strike"] > recs[i-1]["strike"] and r["result"] == "no")
                   or (r["strike"] < recs[i-1]["strike"] and r["result"] == "yes")))
n_rev = sum(1 for i, r in enumerate(recs[1:], 1)
            if recs[i-1]["strike"] and r["strike"] and r["strike"] != recs[i-1]["strike"])
print(f"fade-the-strike-move hit rate: {rev_win/n_rev:.3f} (n={n_rev})")

# time of day (UTC hour)
by_hour = defaultdict(list)
for r in recs:
    h = datetime.fromisoformat(r["window_open"]).hour
    by_hour[h].append(1 if r["result"] == "yes" else 0)
print("\nup-rate by UTC hour (n>=100):")
for h in sorted(by_hour):
    v = by_hour[h]
    if len(v) >= 100:
        print(f"  {h:02d}h: {sum(v)/len(v):.3f} (n={len(v)})")
