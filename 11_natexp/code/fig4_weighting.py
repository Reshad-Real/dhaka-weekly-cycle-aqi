"""Fig. 4: how the hourly responses combine into the daily mean.

(a) Median ERA5 boundary-layer height by local hour and season (out/blh_by_hour.csv).
(b) Share of the daily PM2.5 mean contributed by each hour, averaged over sites after
    normalizing each to its own daily mean (out/pm_share_by_hour.csv).
(c) The Friday change in daily PM2.5 built up in steps from the block contrasts, Dhaka region
    and national, on the common sample (out/decompose.csv, log scale, all seasons): the mean
    of the afternoon and evening contrasts; that reduction spread over the whole day with
    nothing at night or in the morning; with the night and morning contrasts, equal block
    weights; with the blocks weighted by their counterfactual concentrations, as the daily
    mean does.  Points with 95% joint-date bootstrap intervals.
Drawn at the printed width (figstyle.py).
Usage: python -I fig4_weighting.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402

OUT = os.path.join(HERE, "..", "out")
SORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
# validated categorical slots 1-4 (dataviz reference palette, light mode)
SCOL = {"Winter": "#2a78d6", "Pre-monsoon": "#eb6834", "Monsoon": "#1baf7a",
        "Post-monsoon": "#eda100"}
# panel (c) shows regions, not seasons: neutral inks, so no season colour is reused
RCOL = {"Dhaka": "#0b0b0b", "national": "#8a8984"}
INK, INK2, GRID = fs.INK, fs.INK2, fs.GRID
STEPS = [("ae", "Afternoon and evening average"),
         ("partial", "Spread over 24 h, no night or morning change"),
         ("equal", "+ night and morning contrasts, equal weights"),
         ("conc", "+ concentration weights (the daily mean)")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs.apply(plt)
    W1 = pd.read_csv(os.path.join(OUT, "blh_by_hour.csv"))
    W2 = pd.read_csv(os.path.join(OUT, "pm_share_by_hour.csv"))
    D = pd.read_csv(os.path.join(OUT, "decompose.csv"))
    D = D[(D.season == "all") & (D.scale == "log")].set_index("region")

    fig = plt.figure(figsize=(fs.WIDTH, 5.7))
    gs = fig.add_gridspec(1, 2, left=0.1, right=0.985, top=0.91, bottom=0.555, wspace=0.32)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for a in ax:
        for x0, x1 in ((12, 17.99), (18, 23.99)):
            a.axvspan(x0, x1, color=GRID, lw=0, zorder=0)
    for s_ in SORDER:
        q = W1[W1.season == s_]
        ax[0].plot(q.hour, q.blh_median, color=SCOL[s_], lw=1.5, label=s_)
        q = W2[W2.season == s_]
        ax[1].plot(q.hour, 100 * q.share, color=SCOL[s_], lw=1.5)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    for a in ax:
        a.set_xlim(0, 23)
        a.set_xticks(range(0, 24, 3))
        a.set_xlabel("Local hour")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=GRID, lw=0.6)
    ax[0].set_ylabel("ERA5 boundary-layer\nheight, median (m)")
    ax[0].set_title("(a) Mixing depth", loc="left")
    ax[1].axhline(100 / 24, color=INK2, lw=0.8, ls=":")
    ax[1].set_ylabel("Share of the daily\nPM2.5 mean (%)")
    ax[1].set_title("(b) Where the daily mean comes from", loc="left")
    ax[1].text(15, ax[1].get_ylim()[1], "hours Fridays\nreduce", ha="center", va="top", color=INK2)

    # (c) the daily change built up in steps
    # its own grid, so the step labels have room on the left
    c = fig.add_subplot(fig.add_gridspec(1, 1, left=0.43, right=0.985, top=0.375, bottom=0.08)[0, 0])
    y = np.arange(len(STEPS))
    for reg, dy in (("Dhaka", -0.14), ("national", 0.14)):
        r = D.loc[reg]
        v = np.array([r[k + "_pct"] for k, _ in STEPS])
        lo = np.array([r[k + "_lo"] for k, _ in STEPS])
        hi = np.array([r[k + "_hi"] for k, _ in STEPS])
        c.plot(v, y + dy, color=RCOL[reg], lw=0.7, ls=":", zorder=1)
        c.errorbar(v, y + dy, xerr=[v - lo, hi - v], fmt="o", color=RCOL[reg], ms=4.5, capsize=2,
                   lw=1.1, zorder=2, mfc=RCOL[reg] if reg == "Dhaka" else "white",
                   label="Dhaka region" if reg == "Dhaka" else "national")
        if reg == "Dhaka":
            for yi, vi, li in zip(y + dy, v, lo):
                c.text(li - 0.25, yi, "%.1f" % vi, ha="right", va="center", color=INK)
    c.axvline(0, color=INK2, lw=0.8)
    c.set_yticks(y)
    c.set_yticklabels([lab for _, lab in STEPS])
    c.invert_yaxis()
    c.set_xlim(min(D.loc[:, [k + "_lo" for k, _ in STEPS]].min()) - 1.8, 3)
    c.set_xlabel("Friday change in daily PM2.5 (%)")
    # panel title aligned with (a), at the left of the step labels
    c.text(0.1, 0.405, "(c) From the hours to the daily mean", transform=fig.transFigure,
           fontsize=plt.rcParams["axes.titlesize"], ha="left", va="bottom")
    c.legend(frameon=False, loc="upper right")
    c.spines[["top", "right"]].set_visible(False)
    c.grid(axis="x", color=GRID, lw=0.6)
    fig.savefig(os.path.join(OUT, "fig4_weighting.png"))
    print("wrote fig4_weighting.png")


if __name__ == "__main__":
    main()
