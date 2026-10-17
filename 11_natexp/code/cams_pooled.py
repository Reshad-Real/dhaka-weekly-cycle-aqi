"""The central test: is the Friday deficit by diurnal block real once placebo days and
between-site heterogeneity are accounted for?

For every station that passes screening (cams_validate.py) and for the US Embassy monitor,
each day of the week is in turn treated as the "rest day" and compared with the full
working days (Sunday to Wednesday, minus itself when it is one of them), block by block,
with year-month fixed effects and same-block ERA5 meteorology.  Station estimates are pooled
by DerSimonian-Laird random effects, separately for the Dhaka region and for the divisional
cities (a national replication, since Friday is a national rest day).  The four working-day
placebos give the empirical null each Friday and Saturday estimate is read against.

Writes 11_natexp/out/pooled_blocks.csv and 11_natexp/out/fig_pooled_blocks.png.
Usage: python -I cams_pooled.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_shocks import (BLOCKS, MET, calendars, embassy_hourly, frame, met_daily)  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
DHAKA = {"US Embassy", "BARC", "Darussalam", "DoE", "Savar", "Gazipur",
         "Narayanganj", "Narsingdi"}
DAYS = {6: "Sun", 0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat"}
WORK = [6, 0, 1, 2]


def dl_pool(b, se):
    b, se = np.asarray(b, float), np.asarray(se, float)
    w = 1 / se ** 2
    bf = np.sum(w * b) / np.sum(w)
    q = np.sum(w * (b - bf) ** 2)
    k = len(b)
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if k > 1 else 0.0
    ws = 1 / (se ** 2 + tau2)
    br = np.sum(ws * b) / np.sum(ws)
    ser = np.sqrt(1 / np.sum(ws))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return br, ser, tau2, i2, k


def station_estimates(d, y):
    out = []
    s = d[(~d.event) & d[y].notna() & (d[y] > 0)].copy()
    if len(s) < 300:
        return out
    s["ly"] = np.log(s[y])
    for dw, name in DAYS.items():
        comp = [w for w in WORK if w != dw]
        ss = s[s.dow.isin(comp + [dw])].copy()
        ss["treat"] = (ss.dow == dw).astype(int)
        if ss.treat.sum() < 40:
            continue
        f = smf.ols("ly ~ treat + C(ym) + " + MET, data=ss).fit(
            cov_type="HAC", cov_kwds={"maxlags": 7})
        out.append(dict(day=name, b=f.params["treat"], se=f.bse["treat"], n=len(ss)))
    return out


def main(root):
    metd, metb = met_daily(root)
    hart, eid, eid_days = calendars(root)
    val = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    ok_pm = set(val[(val.daily_r_embassy >= 0.6) & (val.best_lag_h.abs() <= 1)].station)
    gas_ok = {g: set(val[val[g + "_rho_tropomi"] >= 0.2].station) for g in ("NOX", "CO")}
    print("PM2.5 screen passed:", sorted(ok_pm))
    print("NOx screen passed:", sorted(gas_ok["NOX"]), "| CO screen passed:", sorted(gas_ok["CO"]))

    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[(cams.time >= "2019-01-01") & (cams.time < "2025-01-01")]
    emb = embassy_hourly(root)
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2026-01-01")]
    allh = pd.concat([cams, emb], ignore_index=True)

    rows = []
    for st, h in allh.groupby("station"):
        region = "Dhaka" if st in DHAKA else "national"
        if region == "Dhaka" and st != "US Embassy" and st not in ok_pm:
            continue
        db = frame(h, metd, hart, eid_days, how="blocks", metb=metb)
        dd = frame(h, metd, hart, eid_days)
        for y in ("PM25", "NOX", "CO"):
            if y != "PM25" and region == "Dhaka" and st not in gas_ok.get(y, set()):
                continue
            for blk in list(BLOCKS) + ["daily"]:
                src = dd if blk == "daily" else db[db.block == blk]
                if y not in src:
                    continue
                for e in station_estimates(src, y):
                    rows.append(dict(station=st, region=region, tracer=y, block=blk, **e))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "station_dow_blocks.csv"), index=False)

    pooled = []
    for (reg, y, blk, day), g in S.groupby(["region", "tracer", "block", "day"]):
        b, se, tau2, i2, k = dl_pool(g.b, g.se)
        pooled.append(dict(region=reg, tracer=y, block=blk, day=day, pct=100 * (np.exp(b) - 1),
                           lo=100 * (np.exp(b - 1.96 * se) - 1), hi=100 * (np.exp(b + 1.96 * se) - 1),
                           z=b / se, tau2=tau2, I2=i2, k=k))
    P = pd.DataFrame(pooled)
    # empirical null: the spread of the four working-day placebos in the same cell
    nul = P[P.day.isin(["Sun", "Mon", "Tue", "Wed"])].groupby(["region", "tracer", "block"]).agg(
        placebo_min=("pct", "min"), placebo_max=("pct", "max"), placebo_absmax=("pct", lambda x: np.abs(x).max()))
    P = P.join(nul, on=["region", "tracer", "block"])
    P["outside_placebo"] = (P.pct < P.placebo_min) | (P.pct > P.placebo_max)
    P.to_csv(os.path.join(OUT, "pooled_blocks.csv"), index=False)
    pd.set_option("display.width", 250)
    order = ["night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23", "daily"]
    for reg in ("Dhaka", "national"):
        for y in ("PM25", "NOX", "CO"):
            q = P[(P.region == reg) & (P.tracer == y)]
            if q.empty:
                continue
            t = q.pivot(index="day", columns="block", values="pct").reindex(
                index=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], columns=order)
            kk = q.groupby("block").k.max().reindex(order)
            print("\n%s %s pooled percent vs working days (k stations: %s)" % (reg, y, kk.to_dict()))
            print(t.round(1).to_string())
            z = q.pivot(index="day", columns="block", values="z").reindex(
                index=["Thu", "Fri", "Sat"], columns=order)
            print("z:", z.round(2).to_string().replace("\n", "\n   "))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    xs = np.arange(4)
    for ax, y in zip(axes, ("PM25", "NOX", "CO")):
        for reg, col, off in (("Dhaka", "#c1121f", -0.12), ("national", "#1b4965", 0.12)):
            q = P[(P.region == reg) & (P.tracer == y) & (P.block != "daily")].set_index(["day", "block"])
            if q.empty:
                continue
            try:
                f = q.xs("Fri", level="day").reindex(order[:4])
            except KeyError:
                continue
            ax.fill_between(xs + off, f.placebo_min, f.placebo_max, color=col, alpha=0.12, lw=0)
            ax.errorbar(xs + off, f.pct, yerr=[f.pct - f.lo, f.hi - f.pct], fmt="o", color=col,
                        label="%s (k=%d)" % (reg, int(f.k.max())), capsize=2)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels(["night", "morning", "afternoon", "evening"], fontsize=8)
        ax.set_title({"PM25": "PM2.5", "NOX": "NOx", "CO": "CO"}[y], fontsize=10)
        ax.set_ylabel("Friday vs working days (%)", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Pooled Friday effect by diurnal block; shading = range of working-day placebos",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pooled_blocks.png"), dpi=200)


if __name__ == "__main__":
    main(sys.argv[1])
