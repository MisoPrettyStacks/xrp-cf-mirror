#!/usr/bin/env python3
"""
Free exchange candle history (Coinbase public API, no key).
Pulls 5-min XRP-USD candles back to the start of the Kalshi history
so every settled window has matching features. Resume-safe.
"""
import csv
import json
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "candles_5m.csv")
UA = {"User-Agent": "Mozilla/5.0 (research project)"}
PRODUCT = "XRP-USD"
GRAN = 300  # 5 minutes


def get(url, retries=5):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if a < retries - 1:
                time.sleep(2 ** a * 2)
                continue
            raise


def main():
    os.makedirs(DATA, exist_ok=True)
    # find oldest Kalshi window to know how far back to go
    oldest = None
    kh = os.path.join(DATA, "kalshi_history.jsonl")
    if os.path.exists(kh):
        with open(kh, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)["window_open"]
                    if oldest is None or t < oldest:
                        oldest = t
    if not oldest:
        print("no kalshi history found")
        sys.exit(1)
    # start 3 days before oldest window (warmup for features)
    from datetime import datetime, timedelta, timezone
    start = datetime.fromisoformat(oldest) - timedelta(days=3)
    end = datetime.now(timezone.utc)

    # resume: find latest candle we have
    have_until = None
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if rows:
                have_until = datetime.fromisoformat(rows[-1]["time"])

    cur_end = end
    cur_start = have_until + timedelta(minutes=5) if have_until else start
    print(f"[candles] from {cur_start.isoformat()} to {cur_end.isoformat()}")

    mode = "a" if os.path.exists(OUT) else "w"
    f = open(OUT, mode, newline="", encoding="utf-8")
    w = csv.writer(f)
    if mode == "w":
        w.writerow(["time", "open", "high", "low", "close", "volume"])

    total = 0
    chunk = timedelta(hours=24)  # 288 candles per request
    s = cur_start
    while s < cur_end:
        e = min(s + chunk, cur_end)
        url = (f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles"
               f"?granularity={GRAN}&start={s.isoformat()}&end={e.isoformat()}")
        try:
            bars = get(url)
        except Exception as ex:
            print(f"  failed {s.isoformat()}: {ex}")
            break
        for b in sorted(bars):
            # [time, low, high, open, close, volume]
            w.writerow([datetime.fromtimestamp(b[0], timezone.utc).isoformat(),
                        b[3], b[2], b[1], b[4], b[5]])
            total += 1
        f.flush()
        s = e
        time.sleep(0.6)
    f.close()
    print(f"[candles] wrote {total} new bars -> {OUT}")


if __name__ == "__main__":
    main()
