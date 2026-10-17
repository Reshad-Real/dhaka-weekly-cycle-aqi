"""Fig. 5 draft: hour profiles of the three activity shocks, and what each buys in AQI.

(a) Pooled Dhaka-region percent change in PM2.5 by local hour for Friday (from
    out/robust_hourly.csv, with the working-day placebo band), Eid and the lockdown (from
    out/events_pooled.csv), 95% intervals shaded.
(b) Daily AQI change per shock in index points: pooled over Dhaka-region stations with
    95% interval, and the US Embassy alone (out/aqi_by_shock.csv).
Usage: python -I fig5_events.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
# validated categorical slots 1-3 (dataviz reference palette; all-pairs safe)
COL = {"Friday": "#2a78d6", "Eid": "#eb6834", "Lockdown": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e9e8e4"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2})
    H = pd.read_csv(os.path.join(OUT, "robust_hourly.csv"))
    fri = H[(H.region == "Dhaka") & (H.day == "Fri")].sort_values("hour")
    E = pd.read_csv(os.path.join(OUT, "events_pooled.csv"))
    E = E[E.unit.astype(str).str.isdigit()].copy()
    E["hour"] = E.unit.astype(int)
    A = pd.read_csv(os.path.join(OUT, "aqi_by_shock.csv")).set_index("event")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
    a = ax[0]
    a.fill_between(fri.hour, fri.placebo_min, fri.placebo_max, color=GRID, lw=0, zorder=0,
                   label="working-day placebos")
    series = [("Friday", fri)] + [(ev, E[E.event == ev].sort_values("hour")) for ev in ("Eid", "Lockdown")]
    for name, q in series:
        a.fill_between(q.hour, q.lo, q.hi, color=COL[name], alpha=0.15, lw=0)
        a.plot(q.hour, q.pct, color=COL[name], lw=2, label=name)
        a.text(23.4, q.pct.iloc[-1], name, color=INK, fontsize=7.5, va="center")
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlim(0, 26.5)
    a.set_xticks(range(0, 24, 3))
    a.set_xlabel("Local hour")
    a.set_ylabel("PM2.5 change, pooled Dhaka region (%)")
    a.set_title("(a) When each shock reaches PM2.5", loc="left", fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.6)

    b = ax[1]
    order = ["Friday", "Saturday", "Eid", "Lockdown"]
    A = A.reindex(order)
    y = np.arange(len(order))
    b.errorbar(A.pooled_aqi_pts, y, xerr=[A.pooled_aqi_pts - A.lo, A.hi - A.pooled_aqi_pts],
               fmt="o", color="#2a78d6", ms=6, capsize=2, label="pooled, Dhaka region")
    b.scatter(A.embassy_aqi_pts, y + 0.22, marker="o", s=28, facecolors="none",
              edgecolors="#2a78d6", label="US Embassy")
    for yi, v, lo in zip(y, A.pooled_aqi_pts, A.lo):
        b.text(lo - 1.0, yi, "%.0f" % v, ha="right", va="center", fontsize=7.5, color=INK)
    b.axvline(0, color=INK2, lw=0.8)
    b.set_yticks(y)
    b.set_yticklabels(order)
    b.invert_yaxis()
    b.set_xlabel("Change in daily AQI (index points)")
    b.set_title("(b) What each shock buys in the index", loc="left", fontsize=9)
    b.legend(frameon=False, fontsize=7, loc="upper left")
    b.set_xlim(min(A.lo.min(), A.embassy_aqi_pts.min()) - 8, 4)
    b.spines[["top", "right"]].set_visible(False)
    b.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_events.png"), dpi=220)
    print("wrote fig5_events.png")


if __name__ == "__main__":
    main()
