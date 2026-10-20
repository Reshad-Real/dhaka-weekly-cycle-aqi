"""Supplementary figures.

  figS1_station_hourly.png  Friday PM2.5 by clock hour at each Dhaka-region station
                            (out/hourly_profiles_station.csv), 95% intervals
  figS2_eid_timing.png      pooled PM2.5 anomaly by day relative to Eid, city sites against
                            sites on the exit highways, event-bootstrap intervals
                            (out/eid_trajectory.csv)
  figS3_lockdown_trajectory.png  weekly residual PM2.5, 2020 against other years
                            (out/lockdown_trajectory.csv)
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

    # Eid: pooled event-time trajectories with event-bootstrap intervals (events_checks.py)
    T = pd.read_csv(os.path.join(OUT, "eid_trajectory.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for a, (grp, title, col) in zip(ax, (("city", "(a) City sites (Embassy, BARC, Darussalam)", BLUE),
                                         ("corridor", "(b) Exit-highway sites (Narsingdi, Savar)", ORANGE))):
        q = T[T.group == grp].sort_values("rel")
        a.axvspan(-1, 3, color=GRID, lw=0, label="main Eid window (-1..+3)")
        a.fill_between(q.rel, q.lo, q.hi, color=col, alpha=0.18, lw=0, label="95% interval (events resampled)")
        a.plot(q.rel, q.pct, color=col, lw=2, marker="o", ms=3, label="pooled anomaly")
        a.axhline(0, color=INK2, lw=0.7)
        a.axvline(0, color=INK2, lw=0.7, ls=":")
        a.set_title(title, loc="left", fontsize=9)
        a.set_xlabel("Days relative to Eid")
        a.set_xticks(range(-10, 13, 2))
        a.legend(frameon=False, fontsize=7, loc="lower left")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=GRID, lw=0.6)
    ax[0].set_ylabel("PM2.5 vs days -30..-11 and +13..+30 (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figS2_eid_timing.png"), dpi=200)

    # Lockdown: weekly residuals of 2020 against the same weeks of the other years
    L = pd.read_csv(os.path.join(OUT, "lockdown_trajectory.csv"))
    fig, a = plt.subplots(figsize=(7.5, 3.4))
    # the placebo years of events_checks.py: 2021 had its own lockdown, 2025 is incomplete
    other = L[L.year.isin([2019, 2022, 2023, 2024])]
    env = other.groupby("week").pct.agg(["min", "max", "mean"])
    a.fill_between(env.index, env["min"], env["max"], color=GRID, lw=0, label="other years, range")
    a.plot(env.index, env["mean"], color=INK2, lw=1.2, ls="--", label="other years, mean")
    q = L[L.year == 2020].sort_values("week")
    a.plot(q.week, q.pct, color=BLUE, lw=2, marker="o", ms=3, label="2020")
    w0 = pd.Timestamp("2020-03-26").isocalendar()[1]
    w1 = pd.Timestamp("2020-05-30").isocalendar()[1]
    a.axvspan(w0, w1, color=BLUE, alpha=0.06, lw=0, label="lockdown (26 Mar - 30 May 2020)")
    a.axhline(0, color=INK2, lw=0.7)
    a.set_xlabel("ISO week (February - July)")
    a.set_ylabel("PM2.5 residual vs model (%)")
    a.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figS3_lockdown_trajectory.png"), dpi=200)
    print("wrote figS1_station_hourly.png, figS2_eid_timing.png, figS3_lockdown_trajectory.png")


if __name__ == "__main__":
    main()
