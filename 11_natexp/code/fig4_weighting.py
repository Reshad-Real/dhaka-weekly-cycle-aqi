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
Usage: python -I fig4_weighting.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
SORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
# validated categorical slots 1-4 (dataviz reference palette, light mode)
SCOL = {"Winter": "#2a78d6", "Pre-monsoon": "#eb6834", "Monsoon": "#1baf7a",
        "Post-monsoon": "#eda100"}
# panel (c) shows regions, not seasons: neutral inks, so no season colour is reused
RCOL = {"Dhaka": "#0b0b0b", "national": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e9e8e4"
STEPS = [("ae", "afternoon and evening\naverage"),
         ("partial", "spread over 24 h,\nno night or morning change"),
         ("equal", "+ night and morning,\nequal weights"),
         ("conc", "+ concentration weights\n(the daily mean)")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    W1 = pd.read_csv(os.path.join(OUT, "blh_by_hour.csv"))
    W2 = pd.read_csv(os.path.join(OUT, "pm_share_by_hour.csv"))
    D = pd.read_csv(os.path.join(OUT, "decompose.csv"))
    D = D[(D.season == "all") & (D.scale == "log")].set_index("region")

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for a in ax[:2]:
        for x0, x1 in ((12, 17.99), (18, 23.99)):
            a.axvspan(x0, x1, color=GRID, lw=0, zorder=0)
    ends = {}
    for s_ in SORDER:
        q = W1[W1.season == s_]
        ax[0].plot(q.hour, q.blh_median, color=SCOL[s_], lw=2, label=s_)
        ends[s_] = q.blh_median.iloc[-1]
        q = W2[W2.season == s_]
        ax[1].plot(q.hour, 100 * q.share, color=SCOL[s_], lw=2)
    # direct labels at the line ends, nudged apart so none collide
    placed = []
    for s_, yv in sorted(ends.items(), key=lambda kv: kv[1]):
        yv = max([yv] + [p + 75 for p in placed])
        placed.append(yv)
        ax[0].text(23.4, yv, s_, color=INK, fontsize=7, va="center")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.33, 1.02))
    ax[0].set_xlim(0, 27)
    ax[0].set_xticks(range(0, 24, 3))
    ax[0].set_xlabel("Local hour")
    ax[0].set_ylabel("ERA5 boundary-layer height, median (m)")
    ax[0].set_title("(a) Mixing depth", loc="left", fontsize=9)
    ax[1].axhline(100 / 24, color=INK2, lw=0.8, ls=":")
    ax[1].set_xticks(range(0, 24, 3))
    ax[1].set_xlabel("Local hour")
    ax[1].set_ylabel("Share of the daily PM2.5 mean (%)")
    ax[1].set_title("(b) Where the daily mean comes from", loc="left", fontsize=9)
    ax[1].text(15, ax[1].get_ylim()[1] * 0.98, "hours Fridays\nreduce", ha="center", va="top",
               fontsize=7, color=INK2)
    for a in ax[:2]:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=GRID, lw=0.6)

    # (c) the daily change built up in steps
    c = ax[2]
    y = np.arange(len(STEPS))
    for reg, dy in (("Dhaka", -0.13), ("national", 0.13)):
        r = D.loc[reg]
        v = np.array([r[k + "_pct"] for k, _ in STEPS])
        lo = np.array([r[k + "_lo"] for k, _ in STEPS])
        hi = np.array([r[k + "_hi"] for k, _ in STEPS])
        c.plot(v, y + dy, color=RCOL[reg], lw=0.8, ls=":", zorder=1)
        c.errorbar(v, y + dy, xerr=[v - lo, hi - v], fmt="o", color=RCOL[reg], ms=5, capsize=2,
                   lw=1.2, zorder=2, mfc=RCOL[reg] if reg == "Dhaka" else "white",
                   label="Dhaka region" if reg == "Dhaka" else "national")
        if reg == "Dhaka":
            for yi, vi, li in zip(y + dy, v, lo):
                c.text(li - 0.25, yi, "%.1f" % vi, ha="right", va="center", fontsize=7, color=INK)
    c.axvline(0, color=INK2, lw=0.8)
    c.set_yticks(y)
    c.set_yticklabels([lab for _, lab in STEPS], fontsize=7)
    c.invert_yaxis()
    c.set_xlim(min(D.loc[:, [k + "_lo" for k, _ in STEPS]].min()) - 2.2, 6)
    c.set_xlabel("Friday change in daily PM2.5 (%)")
    c.set_title("(c) From the hours to the daily mean", loc="left", fontsize=9)
    c.legend(frameon=False, fontsize=7, loc="upper right")
    c.spines[["top", "right"]].set_visible(False)
    c.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig4_weighting.png"), dpi=220)
    print("wrote fig4_weighting.png")


if __name__ == "__main__":
    main()
