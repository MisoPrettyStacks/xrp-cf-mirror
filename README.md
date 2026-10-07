# xrp-cf-mirror

XRP 15-minute direction forecasting research, measured against CF Benchmarks
settlement truth (via Kalshi's free public market history).

- `scripts/collect_kalshi.py` — pulls every settled KXXRP15M window (strike + outcome) from Kalshi's free public API
- `data/kalshi_history.jsonl` — the truth series: 6,427 windows back to 2026-07-31

Free and ethical: only public APIs, no keys, no scraping.
