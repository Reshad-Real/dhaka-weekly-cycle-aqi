"""Fig. 2 (which shocks reduce activity, and when) and Fig. 3 (PM2.5 through the weekend).

Fig. 2  (a) Google mobility change for the lockdown, each Eid and strike days: transit and
            workplaces.
        (b) Surface NOx and CO across Thursday-Friday-Saturday, hour by hour, pooled over
            the national stations, against the same clock hour Sunday-Wednesday.
        (c) The NOx/CO ratio over the same 72 hours (diesel share signature).
Fig. 3  PM2.5 across the same 72 hours: (a) Dhaka region, (b) national stations, with the
        range of the four working-day placebos at each hour and 95% intervals.

Inputs: out/mobility_shocks.csv, out/hourly_profiles.csv.  Usage: python -I figs_2_3.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e9e8e4", "#dcdad4"


def seq(P, region, tracer, cols=("pct", "lo", "hi", "placebo_min", "placebo_max")):
    q = P[(P.region == region) & (P.tracer == tracer) & P.day.isin(["Thu", "Fri", "Sat"])].copy()
    q["x"] = q.day.map({"Thu": 0, "Fri": 24, "Sat": 48}) + q.hour
    return q.sort_values("x"), int(q.k.max()) if len(q) else 0


def day_axis(a):
    for x in (24, 48):
        a.axvline(x, color=INK2, lw=0.6, ls=":")
    a.set_xlim(0, 72)
    a.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72])
    a.set_xticklabels(["0", "6", "12", "18", "0", "6", "12", "18", "0", "6", "12", "18", "24"])
    ymax = a.get_ylim()[1]
    for x, lab in ((12, "Thursday"), (36, "Friday (rest day)"), (60, "Saturday")):
        a.text(x, ymax, lab, ha="center", va="bottom", fontsize=7.5, color=INK)
    a.axhline(0, color=INK2, lw=0.8)
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.6)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2})
    P = pd.read_csv(os.path.join(OUT, "hourly_profiles.csv"))
    M = pd.read_csv(os.path.join(OUT, "mobility_shocks.csv"))

    # ------------------------------------------------------------------ Fig. 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [0.8, 1.3, 1.3]})
    a = ax[0]
    lock = M[M.shock.str.startswith("lockdown")].iloc[0]
    eids = M[M.shock.str.startswith("Eid")]
    strike = M[M.shock.str.startswith("strike")].iloc[0]
    cats = ["Lockdown", "Eid (mean of %d)" % len(eids), "Strike days"]
    vals = {"transit": [lock.transit, eids.transit.mean(), strike.transit],
            "workplaces": [lock.workplaces, eids.workplaces.mean(), strike.workplaces]}
    y = np.arange(3)
    for off, (k, col) in zip((-0.18, 0.18), (("transit", BLUE), ("workplaces", ORANGE))):
        a.barh(y + off, vals[k], height=0.34, color=col, label=k, edgecolor="white", linewidth=1)
    for off, k in ((-0.18, "transit"), (0.18, "workplaces")):
        a.scatter(eids[k], np.full(len(eids), 1 + off), s=10, color=INK, zorder=3,
                  label="single Eid" if k == "transit" else None)
    a.set_yticks(y)
    a.set_yticklabels(cats)
    a.invert_yaxis()
    a.axvline(0, color=INK2, lw=0.8)
    a.set_xlabel("Google mobility change (%)")
    a.set_title("(a) Activity measured by mobility", loc="left", fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="lower left")
    a.spines[["top", "right"]].set_visible(False)

    a = ax[1]
    for tr, col in (("NOX", BLUE), ("CO", ORANGE)):
        q, k = seq(P, "national", tr)
        a.fill_between(q.x, q.lo, q.hi, color=col, alpha=0.15, lw=0)
        a.plot(q.x, q.pct, color=col, lw=2, label="%s (k=%d)" % ("NOx" if tr == "NOX" else "CO", k))
    a.set_ylim(-16, 12)
    a.set_ylabel("vs same hour Sun-Wed (%)")
    a.set_title("(b) Surface combustion tracers, national stations", loc="left", fontsize=9, pad=14)
    day_axis(a)
    a.legend(frameon=False, fontsize=7, loc="lower left")
    a.set_xlabel("Local hour")

    a = ax[2]
    # National stations only: the single Dhaka station where both analyzers pass has hourly
    # intervals about 50 percentage points wide and cannot resolve the signature.
    q, k = seq(P, "national", "NOX_CO")
    a.fill_between(q.x, q.placebo_min, q.placebo_max, color=BAND, lw=0, label="working-day placebos")
    a.fill_between(q.x, q.lo, q.hi, color=BLUE, alpha=0.15, lw=0, label="95% interval")
    a.plot(q.x, q.pct, color=BLUE, lw=2, label="national stations (k=%d)" % k)
    a.set_ylim(-16, 16)
    a.set_ylabel("NOx/CO ratio vs same hour Sun-Wed (%)")
    a.set_title("(c) NOx per unit CO, same site and hour", loc="left", fontsize=9, pad=14)
    day_axis(a)
    a.legend(frameon=False, fontsize=7, loc="lower left")
    a.set_xlabel("Local hour")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_activity.png"), dpi=220)

    # ------------------------------------------------------------------ Fig. 3
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for a, reg, title in ((ax[0], "Dhaka", "(a) Dhaka region"), (ax[1], "national", "(b) Nine stations in other cities")):
        q, k = seq(P, reg, "PM25")
        a.fill_between(q.x, q.placebo_min, q.placebo_max, color=BAND, lw=0, label="working-day placebos")
        a.fill_between(q.x, q.lo, q.hi, color=BLUE, alpha=0.18, lw=0, label="95% interval")
        a.plot(q.x, q.pct, color=BLUE, lw=2, label="PM2.5 (k=%d stations)" % k)
        a.set_ylim(-16, 10)
        day_axis(a)
        a.set_title(title, loc="left", fontsize=9, pad=14)
        a.set_xlabel("Local hour")
        a.legend(frameon=False, fontsize=7, loc="lower left")
    ax[0].set_ylabel("PM2.5 vs same hour Sun-Wed (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_weekend_pm25.png"), dpi=220)
    print("wrote fig2_activity.png, fig3_weekend_pm25.png")


if __name__ == "__main__":
    main()
