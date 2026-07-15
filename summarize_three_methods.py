#!/usr/bin/env python3
"""Summarize success rate / SPL / NE / LE for three logoplanner checkpoints
on two shared internscenes_home scenes."""
import csv
from pathlib import Path

ROOT = Path("/home/asus/Research/Nav/NavDP")

METHODS = {
    "Official":   "startgoal_logoplanner_internscenes_home_OFFICIAL",
    "Retrain":    "startgoal_logoplanner_internscenes_home_retrain",
    "Critic2":    "startgoal_logoplanner_internscenes_home_CRIT2",
    "Critic2_ng0":"startgoal_logoplanner_internscenes_home_critic2_ng0",
}

SCENES = [
    "MVUCSQAKTKJ5EAABAAAAABA8_usd",
    "MVUCSQAKTKJ5EAABAAAAABY8_usd",
]


def load(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def agg(rows):
    n = len(rows)
    if n == 0:
        return None
    sr  = sum(r["success"] for r in rows) / n
    spl = sum(r["spl"]     for r in rows) / n
    ne  = sum(r["ne"]      for r in rows) / n
    le  = sum(r["le"]      for r in rows) / n
    return n, sr, spl, ne, le


def fmt(label, stats):
    n, sr, spl, ne, le = stats
    return f"  {label:>6s}  N={n:3d}  SR={sr*100:5.1f}%  SPL={spl*100:5.1f}%  NE_mean={ne:5.2f}  LE_mean={le:5.2f}"


for method, subdir in METHODS.items():
    print(f"=== {method} ===")
    all_rows = []
    for s in SCENES:
        csv_path = ROOT / subdir / s / "metric.csv"
        if not csv_path.exists():
            print(f"  {s[-3:]:>6s}  (missing: {csv_path})")
            continue
        rows = load(csv_path)
        all_rows.extend(rows)
        print(fmt(s[-7:-4], agg(rows)))
    if all_rows:
        print(fmt("COMB", agg(all_rows)))
    print()
