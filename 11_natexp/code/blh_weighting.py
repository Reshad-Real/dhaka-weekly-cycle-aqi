"""Why a daytime cut barely moves the daily mean: the boundary-layer weighting
(OUTLINE.md, next steps 2).

  W1  ERA5 boundary-layer height and ventilation coefficient by local hour and season.
  W2  PM2.5 by local hour and season, as a share of the daily mean: the US Embassy monitor
      and the screened Dhaka CAMS stations (each station normalized to its own daily mean
      before averaging, so no single site dominates).
  W3  Weighting identity: the daily Friday effect predicted as the share-weighted sum of
      the pooled block effects, against the observed daily effect, for dry (Nov-Apr) and
      wet (May-Oct) seasons and for the full year.  Block effects come from
      out/robust_pooled.csv (variants dry, wet, main).
  W4  Ventilation dependence: within each diurnal block, the pooled Friday effect by
      tercile of that block's own ventilation coefficient.

Writes out/blh_by_hour.csv, out/pm_share_by_hour.csv, out/weighting_check.csv,
out/friday_by_block_vc.csv and out/fig4_weighting.png.
Usage: python -I blh_weighting.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_shocks import BLOCKS, embassy_hourly  # noqa: E402
from cams_pooled import DHAKA, dl_pool  # noqa: E402
import robustness as rb  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
          5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
          10: "Post-monsoon", 11: "Post-monsoon"}
SORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
# validated categorical slots 1-4 (dataviz reference palette, light mode)
SCOL = {"Winter": "#2a78d6", "Pre-monsoon": "#eb6834", "Monsoon": "#1baf7a",
        "Post-monsoon": "#eda100"}
DRY = [11, 12, 1, 2, 3, 4]
BLK_OF = {h: b for b, hs in BLOCKS.items() for h in hs}


def main(root):
    # ---------------------------------------------------------------- W1
    m = pd.read_csv(os.path.join(root, "02_data", "raw", "meteo_dhaka_hourly_filled.csv"),
                    parse_dates=["time"])
    m = m[(m.time >= "2019-01-01") & (m.time < "2026-01-01")]
    m["season"] = m.time.dt.month.map(SEASON)
    m["hour"] = m.time.dt.hour
    m["vc"] = np.maximum(m.wind_speed_10m / 3.6, 0.5) * np.maximum(m.boundary_layer_height, 30)
    W1 = m.groupby(["season", "hour"]).agg(blh_median=("boundary_layer_height", "median"),
                                           blh_q25=("boundary_layer_height", lambda x: x.quantile(.25)),
                                           blh_q75=("boundary_layer_height", lambda x: x.quantile(.75)),
                                           vc_median=("vc", "median")).reset_index()
    W1.to_csv(os.path.join(OUT, "blh_by_hour.csv"), index=False)
    blk_blh = m.assign(block=m.hour.map(BLK_OF)).groupby(["season", "block"]).boundary_layer_height.median()
    print("median ERA5 boundary-layer height (m) by block")
    print(blk_blh.unstack()[list(BLOCKS)].reindex(SORDER).round(0).to_string())

    # ---------------------------------------------------------------- W2
    val = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    ok_pm = set(val[(val.daily_r_embassy >= 0.6) & (val.best_lag_h.abs() <= 1)].station)
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[cams.station.isin(ok_pm) & (cams.time >= "2019-01-01")][["station", "time", "PM25"]]
    emb = embassy_hourly(root)
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2026-01-01")]
    h = pd.concat([cams, emb], ignore_index=True).dropna(subset=["PM25"])
    h = h[h.PM25 > 0]
    h["season"] = h.time.dt.month.map(SEASON)
    h["hour"] = h.time.dt.hour
    prof = h.groupby(["station", "season", "hour"]).PM25.mean().reset_index()
    prof["share"] = prof.PM25 / prof.groupby(["station", "season"]).PM25.transform("sum")
    W2 = prof.groupby(["season", "hour"]).share.agg(["mean", "min", "max"]).reset_index()
    W2.columns = ["season", "hour", "share", "share_min", "share_max"]
    W2.to_csv(os.path.join(OUT, "pm_share_by_hour.csv"), index=False)
    prof["block"] = prof.hour.map(BLK_OF)
    bshare = prof.groupby(["station", "season", "block"]).share.sum().groupby(["season", "block"]).mean()
    print("\nshare of the daily PM2.5 mean by block, mean over %d sites" % prof.station.nunique())
    print((100 * bshare.unstack()[list(BLOCKS)].reindex(SORDER)).round(1).to_string())
    # dry / wet / all block weights
    prof["sgroup"] = np.where(prof.season.isin(["Winter", "Post-monsoon"]) |
                              (prof.season == "Pre-monsoon"), "dry", "wet")
    # use months for an exact dry/wet split
    h["sgroup"] = np.where(h.time.dt.month.isin(DRY), "dry", "wet")
    pg = h.groupby(["station", "sgroup", "hour"]).PM25.mean().reset_index()
    pa = h.groupby(["station", "hour"]).PM25.mean().reset_index().assign(sgroup="main")
    pg = pd.concat([pg, pa])
    pg["share"] = pg.PM25 / pg.groupby(["station", "sgroup"]).PM25.transform("sum")
    pg["block"] = pg.hour.map(BLK_OF)
    wts = pg.groupby(["station", "sgroup", "block"]).share.sum().groupby(["sgroup", "block"]).mean()

    # ---------------------------------------------------------------- W3
    R = pd.read_csv(os.path.join(OUT, "robust_pooled.csv"))
    rows = []
    for var, sg in (("main", "main"), ("dry", "dry"), ("wet", "wet")):
        for reg in ("Dhaka", "national"):
            q = R[(R.variant == var) & (R.region == reg) & (R.tracer == "PM25") & (R.day == "Fri")]
            if q.empty:
                continue
            q = q.set_index("block")
            pred = sum(wts.loc[(sg, b)] * q.loc[b, "pct"] for b in BLOCKS)
            rows.append(dict(variant=var, region=reg, predicted_daily_pct=pred,
                             observed_daily_pct=q.loc["daily", "pct"],
                             observed_lo=q.loc["daily", "lo"], observed_hi=q.loc["daily", "hi"],
                             **{"w_" + b.split()[0]: wts.loc[(sg, b)] for b in BLOCKS},
                             **{"eff_" + b.split()[0]: q.loc[b, "pct"] for b in BLOCKS}))
    W3 = pd.DataFrame(rows)
    W3.to_csv(os.path.join(OUT, "weighting_check.csv"), index=False)
    print("\nweighting identity: predicted vs observed daily Friday effect (%)")
    print(W3.round(2).to_string(index=False))

    # ---------------------------------------------------------------- W4
    frames = rb.load(root)[0]          # the main-screened frames
    rows = []
    for (st, reg, y, blk), s in frames.items():
        if y != "PM25" or blk == "daily":
            continue
        s = s.copy()
        s["vct"] = pd.qcut(s["vc"], 3, labels=["low", "mid", "high"])
        for lev in ("low", "mid", "high"):
            r = rb.fit(s[s.vct == lev], 4)
            if r:
                rows.append(dict(station=st, region=reg, block=blk, vc_tercile=lev, b=r[0], se=r[1],
                                 vc_mean=s.loc[s.vct == lev, "vc"].mean()))
    V = pd.DataFrame(rows)
    out = []
    for (reg, blk, lev), g in V.groupby(["region", "block", "vc_tercile"]):
        b, se, tau2, i2, k = dl_pool(g.b, g.se)
        out.append(dict(region=reg, block=blk, vc_tercile=lev, pct=100 * (np.exp(b) - 1),
                        lo=100 * (np.exp(b - 1.96 * se) - 1), hi=100 * (np.exp(b + 1.96 * se) - 1),
                        k=k, vc_mean=g.vc_mean.mean()))
    W4 = pd.DataFrame(out)
    W4.to_csv(os.path.join(OUT, "friday_by_block_vc.csv"), index=False)
    print("\nFriday PM2.5 effect by same-block ventilation tercile (pooled %)")
    print(W4.pivot_table(index=["region", "block"], columns="vc_tercile", values="pct")[
        ["low", "mid", "high"]].round(1).to_string())

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for a in ax[:2]:
        for x0, x1 in ((12, 17.99), (18, 23.99)):
            a.axvspan(x0, x1, color="#e9e8e4", lw=0, zorder=0)
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
        ax[0].text(23.4, yv, s_, color="#0b0b0b", fontsize=7, va="center")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.38, 1.02))
    ax[0].set_xlim(0, 27)
    ax[0].set_xticks(range(0, 24, 3))
    ax[0].set_xlabel("Local hour")
    ax[0].set_ylabel("ERA5 boundary-layer height, median (m)")
    ax[0].set_title("(a) Mixing depth", loc="left", fontsize=9)
    ax[1].axhline(100 / 24, color="#52514e", lw=0.8, ls=":")
    ax[1].set_xticks(range(0, 24, 3))
    ax[1].set_xlabel("Local hour")
    ax[1].set_ylabel("Share of the daily PM2.5 mean (%)")
    ax[1].set_title("(b) Where the daily mean comes from", loc="left", fontsize=9)
    ax[1].text(15, ax[1].get_ylim()[1] * 0.98, "hours Fridays\nreduce", ha="center", va="top",
               fontsize=7, color="#52514e")
    for a in ax[:2]:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e9e8e4", lw=0.6)
    # (c) predicted vs observed daily effect
    q = W3[W3.region == "Dhaka"].set_index("variant").reindex(["main", "dry", "wet"])
    y = np.arange(len(q))
    ax[2].errorbar(q.observed_daily_pct, y, xerr=[q.observed_daily_pct - q.observed_lo,
                                                  q.observed_hi - q.observed_daily_pct],
                   fmt="o", color="#2a78d6", ms=6, capsize=2, label="observed daily")
    ax[2].scatter(q.predicted_daily_pct, y + 0.18, marker="D", s=30, color="#eb6834",
                  label="predicted from blocks")
    ax[2].axvline(0, color="#52514e", lw=0.8)
    ax[2].set_yticks(y)
    ax[2].set_yticklabels(["all year", "dry (Nov-Apr)", "wet (May-Oct)"])
    ax[2].invert_yaxis()
    ax[2].set_xlabel("Friday effect on daily PM2.5 (%)")
    ax[2].set_title("(c) Accounting check", loc="left", fontsize=9)
    ax[2].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[2].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig4_weighting.png"), dpi=220)
    print("\nwrote fig4_weighting.png")


if __name__ == "__main__":
    main(sys.argv[1])
