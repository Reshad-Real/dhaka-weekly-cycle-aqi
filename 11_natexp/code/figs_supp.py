"""Supplementary figures.

  figS1_station_hourly.png  Friday PM2.5 by clock hour at each Dhaka-region station
                            (out/hourly_profiles_station.csv), 95% intervals
  figS2_eid_timing.png      PM2.5 anomaly by day relative to Eid, city sites against sites on
                            the exit highways (out/eid_relative_day.csv)
Usage: python -I figs_supp.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e9e8e4"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

    S = pd.read_csv(os.path.join(OUT, "hourly_profiles_station.csv"))
    S = S[(S.region == "Dhaka") & (S.tracer == "PM25") & (S.day == "Fri")].copy()
    f = lambda v: 100 * (np.exp(v) - 1)
    S["pct"], S["lo"], S["hi"] = f(S.b), f(S.b - 1.96 * S.se), f(S.b + 1.96 * S.se)
    sts = sorted(S.station.unique(), key=lambda x: (x != "US Embassy", x))
    n = len(sts)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(11, 2.4 * rows), sharex=True, sharey=True)
    ax = np.atleast_2d(ax)
    for i, st in enumerate(sts):
        a = ax[i // cols, i % cols]
        q = S[S.station == st].sort_values("hour")
        a.axvspan(12, 23.5, color=GRID, lw=0)
        a.fill_between(q.hour, q.lo, q.hi, color=BLUE, alpha=0.18, lw=0)
        a.plot(q.hour, q.pct, color=BLUE, lw=1.8)
        a.axhline(0, color=INK2, lw=0.7)
        a.set_title(st, loc="left", fontsize=8.5)
        a.set_ylim(-30, 25)
        a.set_xticks(range(0, 24, 6))
        a.spines[["top", "right"]].set_visible(False)
    for j in range(n, rows * cols):
        ax[j // cols, j % cols].axis("off")
    for a in ax[-1]:
        a.set_xlabel("Local hour")
    for a in ax[:, 0]:
        a.set_ylabel("Friday vs Sun-Wed (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figS1_station_hourly.png"), dpi=200)

    R = pd.read_csv(os.path.join(OUT, "eid_relative_day.csv"))
    R = R[R.tracer == "PM25"]
    groups = (("(a) City sites", [("US Embassy", BLUE), ("BARC", ORANGE), ("Darussalam", AQUA)]),
              ("(b) Sites on exit highways", [("Narsingdi", BLUE), ("Savar", ORANGE)]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for a, (title, members) in zip(ax, groups):
        a.axvspan(-1, 3, color=GRID, lw=0, label="Eid window (-1..+3)")
        for st, col in members:
            q = R[R.station == st].sort_values("rel")
            if q.empty:
                continue
            lo = 100 * (np.exp(np.log1p(q.pct / 100) - 1.96 * q.se) - 1)
            hi = 100 * (np.exp(np.log1p(q.pct / 100) + 1.96 * q.se) - 1)
            a.fill_between(q.rel, lo, hi, color=col, alpha=0.12, lw=0)
            a.plot(q.rel, q.pct, color=col, lw=2, marker="o", ms=3, label=st)
        a.axhline(0, color=INK2, lw=0.7)
        a.axvline(0, color=INK2, lw=0.7, ls=":")
        a.set_title(title, loc="left", fontsize=9)
        a.set_xlabel("Days relative to Eid")
        a.set_xticks(range(-5, 9))
        a.legend(frameon=False, fontsize=7, loc="lower left")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=GRID, lw=0.6)
    ax[0].set_ylabel("PM2.5 anomaly (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figS2_eid_timing.png"), dpi=200)
    print("wrote figS1_station_hourly.png, figS2_eid_timing.png")


if __name__ == "__main__":
    main()
