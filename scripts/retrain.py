#!/usr/bin/env python3
"""
Weekly self-improvement: refresh data, retrain challenger, promote only if
it beats the champion on walk-forward out-of-sample Brier skill.
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAMPION = os.path.join(BASE, "models", "champion.json")


def run(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=BASE)
    if r.returncode != 0:
        sys.exit(f"FAILED: {cmd}")


def main():
    # 1. refresh data
    run("python3 scripts/collect_kalshi.py --incremental")
    run("python3 scripts/collect_candles.py")
    run("python3 scripts/features.py")

    # 2. train challenger (walk-forward OOS) and capture metrics
    out = subprocess.run("python3 scripts/train.py", shell=True, cwd=BASE,
                         capture_output=True, text=True)
    print(out.stdout[-1500:])
    if out.returncode != 0:
        print(out.stderr[-500:])
        sys.exit("train failed")

    # parse challenger Brier skill from train output
    chal_skill = None
    for line in out.stdout.splitlines():
        if line.startswith("blend+iso"):
            # brier=0.24741 skill=+1.04%
            for tok in line.split():
                if tok.startswith("skill="):
                    chal_skill = float(tok.split("=")[1].rstrip("%"))
    if chal_skill is None:
        sys.exit("could not parse challenger skill")

    # 3. compare with champion
    champ_skill = -999
    if os.path.exists(CHAMPION):
        champ_skill = json.load(open(CHAMPION)).get("brier_skill_pct", -999)
    print(f"challenger skill: {chal_skill:+.2f}% | champion skill: {champ_skill:+.2f}%")

    if chal_skill > champ_skill + 0.05:  # meaningful margin, not noise
        print("PROMOTING challenger")
        run("python3 scripts/finalize.py")
        json.dump({"brier_skill_pct": chal_skill,
                   "promoted": __import__("datetime").datetime.now(
                       __import__("datetime").timezone.utc).isoformat()},
                  open(CHAMPION, "w"), indent=1)
    else:
        print("challenger not better; champion stands")


if __name__ == "__main__":
    main()
