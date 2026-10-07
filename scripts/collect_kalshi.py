#!/usr/bin/env python3
"""
Kalshi KXXRP15M settlement history collector (free, public API, no key).
=========================================================================
Pulls every settled 15-minute XRP window: the CF-derived strike
(floor_strike), the outcome (yes/no), and window times. This is the
"truth" series the predictor is measured against.

Usage:
  python3 scripts/collect_kalshi.py                 # full backfill
  python3 scripts/collect_kalshi.py --incremental    # only new windows

Output: data/kalshi_history.jsonl (one JSON object per window)
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "kalshi_history.jsonl")
SERIES = "KXXRP15M"
UA = {"User-Agent": "Mozilla/5.0 (research project)"}


def get(url, retries=5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 2 ** attempt * 3
                print(f"  [rate-limited, waiting {wait}s]", flush=True)
                time.sleep(wait)
                continue
            raise


def load_existing():
    seen = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seen.add(json.loads(line)["ticker"])
                    except (KeyError, ValueError):
                        pass
    return seen


def main():
    incremental = "--incremental" in sys.argv
    os.makedirs(DATA, exist_ok=True)
    # always resume-safe: never lose already-collected windows
    seen = load_existing()
    mode = "a"

    url = (f"https://api.elections.kalshi.com/trade-api/v2/markets"
           f"?series_ticker={SERIES}&status=settled&limit=100")
    total, new, pages = 0, 0, 0
    with open(OUT, mode, encoding="utf-8") as f:
        cursor = None
        while True:
            u = url + (f"&cursor={cursor}" if cursor else "")
            d = get(u)
            markets = d.get("markets", [])
            if not markets:
                break
            pages += 1
            for m in markets:
                total += 1
                if m["ticker"] in seen:
                    continue
                rec = {
                    "ticker": m["ticker"],
                    "window_open": m.get("open_time"),
                    "window_close": m.get("close_time"),
                    "strike": m.get("floor_strike"),
                    "result": m.get("result"),  # "yes" = closed >= strike
                    "status": m.get("status"),
                }
                f.write(json.dumps(rec) + "\n")
                seen.add(m["ticker"])
                new += 1
            cursor = d.get("cursor")
            if not cursor:
                break
            time.sleep(1.2)
            if incremental and new == 0 and pages >= 3:
                # caught up to already-seen territory
                break

    print(f"[collect] pages={pages} scanned={total} new={new} -> {OUT}")


if __name__ == "__main__":
    main()
