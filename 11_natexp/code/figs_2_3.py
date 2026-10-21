"""Fig. 2 (which shocks reduce activity, and when) and Fig. 3 (PM2.5 through the weekend).

Fig. 2  (a) Google mobility change for the lockdown, each Eid and strike days: transit and
            workplaces.
        (b) Surface NOx and CO across Thursday-Friday-Saturday, hour by hour, pooled over
            the national stations, against the same clock hour Sunday-Wednesday.
        (c) The NOx/CO ratio over the same 72 hours (diesel share signature).
Fig. 3  PM2.5 across the same 72 hours: (a) Dhaka region, (b) national stations, with the
        range of the four working-day placebos at each hour.
Intervals (2b, 2c, 3): pointwise 95% intervals (shaded) and the simultaneous 95% band for the
whole 72-hour curve (dashed), both from the joint-date bootstrap of hourly_boot.py.

Both are drawn at the printed width (figstyle.py).
Inputs: out/mobility_shocks.csv, out/hourly_profiles.csv, out/hourly_boot.csv.
Usage: python -I figs_2_3.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402

OUT = os.path.join(HERE, "..", "out")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BAND = fs.INK, fs.INK2, fs.GRID, "#dcdad4"
LW, LW_BAND = 1.5, 0.6


def seq(P, region, tracer, cols=("pct", "lo", "hi", "placebo_min", "placebo_max")):
    q = P[(P.region == region) & (P.tracer == tracer) & P.day.isin(["Thu", "Fri", "Sat"])].copy()
    q["x"] = q.day.map({"Thu": 0, "Fri": 24, "Sat": 48}) + q.hour
    return q.sort_values("x"), int(q.k.max()) if len(q) else 0


def day_axis(a, days=True):
    for x in (24, 48):
        a.axvline(x, color=INK2, lw=0.6, ls=":")
    a.set_xlim(0, 72)
    a.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72])
    a.set_xticklabels(["0", "6", "12", "18", "0", "6", "12", "18", "0", "6", "12", "18", "24"])
    ymax = a.get_ylim()[1]
    if days:
        for x, lab in ((12, "Thursday"), (36, "Friday (rest day)"), (60, "Saturday")):
            a.text(x, ymax, lab, ha="center", va="bottom", fontsize=fs.SMALL, color=INK)
    a.axhline(0, color=INK2, lw=0.8)
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.6)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.apply(plt)
    P = pd.read_csv(os.path.join(OUT, "hourly_profiles.csv"))
    # dependence-aware intervals replace the independent-station ones (Thu, Fri, Sat)
    H = pd.read_csv(os.path.join(OUT, "hourly_boot.csv"))
    P = P.drop(columns=["lo", "hi"]).merge(
        H[["region", "tracer", "day", "hour", "lo", "hi", "sim_lo", "sim_hi"]],
        on=["region", "tracer", "day", "hour"], how="left")
    M = pd.read_csv(os.path.join(OUT, "mobility_shocks.csv"))

    # ------------------------------------------------------------------ Fig. 2
    fig = plt.figure(figsize=(fs.WIDTH, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.9, 1.6], height_ratios=[1, 1], left=0.135, right=0.985,
                          top=0.935, bottom=0.165, wspace=0.45, hspace=0.55)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
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
        a.scatter(eids[k], np.full(len(eids), 1 + off), s=7, color=INK, zorder=3,
                  label="single Eid" if k == "transit" else None)
    a.set_yticks(y)
    a.set_yticklabels([c.replace(" (", "\n(") for c in cats])
    a.invert_yaxis()
    a.axvline(0, color=INK2, lw=0.8)
    a.set_xlabel("Google mobility change (%)")
    a.set_title("(a) Mobility", loc="left", pad=14)
    a.legend(frameon=False, loc="lower left")
    a.spines[["top", "right"]].set_visible(False)

    a = ax[1]
    for tr, col in (("NOX", BLUE), ("CO", ORANGE)):
        q, k = seq(P, "national", tr)
        a.fill_between(q.x, q.lo, q.hi, color=col, alpha=0.15, lw=0)
        a.plot(q.x, q.sim_lo, color=col, lw=LW_BAND, ls="--")
        a.plot(q.x, q.sim_hi, color=col, lw=LW_BAND, ls="--")
        a.plot(q.x, q.pct, color=col, lw=LW, label="%s (k=%d)" % ("NOx" if tr == "NOX" else "CO", k))
    a.set_ylim(-16, 12)
    a.set_ylabel("vs same hour Sun-Wed (%)")
    a.set_title("(b) Surface NOx and CO, national stations", loc="left", pad=14)
    day_axis(a)
    a.legend(frameon=False, loc="lower left")
    a.set_xlabel("Local hour")

    a = ax[2]
    # the national stations (the two Dhaka stations with both analyzers are in the text)
    q, k = seq(P, "national", "NOX_CO")
    a.fill_between(q.x, q.placebo_min, q.placebo_max, color=BAND, lw=0, label="working-day placebos")
    a.fill_between(q.x, q.lo, q.hi, color=BLUE, alpha=0.15, lw=0, label="pointwise 95%")
    a.plot(q.x, q.sim_lo, color=BLUE, lw=LW_BAND, ls="--", label="simultaneous 95%")
    a.plot(q.x, q.sim_hi, color=BLUE, lw=LW_BAND, ls="--")
    a.plot(q.x, q.pct, color=BLUE, lw=LW, label="national stations (k=%d)" % k)
    a.set_ylim(-16, 16)
    a.set_ylabel("NOx/CO ratio vs\nsame hour Sun-Wed (%)")
    a.set_title("(c) NOx per unit CO, same site and hour", loc="left", pad=14)
    day_axis(a)
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, columnspacing=3)
    a.set_xlabel("Local hour")
    fig.savefig(os.path.join(OUT, "fig2_activity.png"))

    # ------------------------------------------------------------------ Fig. 3
    fig, ax = plt.subplots(2, 1, figsize=(fs.WIDTH, 5.4), sharex=True, sharey=True)
    for a, reg, title in ((ax[0], "Dhaka", "(a) Dhaka region"), (ax[1], "national", "(b) Nine stations in other cities")):
        q, k = seq(P, reg, "PM25")
        a.fill_between(q.x, q.placebo_min, q.placebo_max, color=BAND, lw=0, label="working-day placebos")
        a.fill_between(q.x, q.lo, q.hi, color=BLUE, alpha=0.18, lw=0, label="pointwise 95%")
        a.plot(q.x, q.sim_lo, color=BLUE, lw=LW_BAND, ls="--", label="simultaneous 95%")
        a.plot(q.x, q.sim_hi, color=BLUE, lw=LW_BAND, ls="--")
        a.plot(q.x, q.pct, color=BLUE, lw=LW, label="PM2.5 (k=%d stations)" % k)
        a.set_ylim(-16, 10)
        day_axis(a)
        a.set_title(title, loc="left", pad=14)
        a.set_ylabel("PM2.5 vs same hour\nSun-Wed (%)")
    ax[1].set_xlabel("Local hour")
    h, lab = ax[0].get_legend_handles_labels()
    lab = [x.replace("k=7 stations", "k = 7 or 9 stations") for x in lab]
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=1.2, rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "fig3_weekend_pm25.png"))
    print("wrote fig2_activity.png, fig3_weekend_pm25.png")


if __name__ == "__main__":
    main()
