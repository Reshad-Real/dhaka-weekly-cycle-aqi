"""Hour profiles of the two whole-city shocks, and the AQI change every shock buys
(OUTLINE.md, next steps 3).

Eid      each Eid window (-1..+3 days) against its own surrounding days (-15..+18), with
         event-window and day-of-week fixed effects and same-period ERA5 meteorology;
         lockdown period and strike days excluded.
Lockdown 2020-03-26..05-30 against the same calendar window in every other year the
         station reports, with year and day-of-week fixed effects and meteorology; Eid
         days outside 2020 excluded.  PM2.5 only: CAMS gas analyzers change level between
         years (see cams_validate.py), so cross-year gas comparisons are not used.
Weekly   Friday and Saturday vs Sunday-Wednesday (the main design), for the AQI rows.

Effects are estimated per station for every hour, every diurnal block and the daily
mean, then pooled across Dhaka-region stations (DerSimonian-Laird).  AQI effects are in
index points, from the daily mean PM2.5 through the US EPA 2024 / Bangladesh breakpoints.

Writes out/events_hourly.csv, out/events_blocks.csv, out/aqi_by_shock.csv.
Usage: python -I events_hourly.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_shocks import BLOCKS, COVID, calendars, embassy_hourly, met_daily  # noqa: E402
from cams_pooled import DHAKA, dl_pool  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
METC = ["lvc", "lvc2", "T", "rh", "lpr", "wd_s", "wd_c"]
BP = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
      (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]


def aqi_pm25(c, cap=True):
    """Daily PM2.5 sub-index on the 2024 breakpoints, concentration truncated to 0.1, not
    rounded.  Above 325.4 the index is set to 500 (cap=True, the main convention) or the top
    segment is extended linearly (cap=False, a sensitivity check)."""
    c = np.floor(np.asarray(c, float) * 10) / 10
    out = np.full_like(c, np.nan)
    for lo, hi, ilo, ihi in BP:
        m = (c >= lo) & (c <= hi + 0.0999)
        out[m] = ilo + (ihi - ilo) / (hi - lo) * (np.minimum(c[m], hi) - lo)
    top = c > 325.4
    if cap:
        out[top] = 500
    else:
        lo, hi, ilo, ihi = BP[-1]
        out[top] = ilo + (ihi - ilo) / (hi - lo) * (c[top] - lo)
    return out


def ols_fe(y, treat, X, fe, cov_lags=7):
    """OLS of y on treat + X with one set of fixed effects absorbed by demeaning."""
    codes, uniq = pd.factorize(fe)
    ng = len(uniq)
    cnt = np.bincount(codes, minlength=ng)

    def dm(a):
        return a - (np.bincount(codes, a, ng) / cnt)[codes]
    Z = np.column_stack([treat] + [X[:, j] for j in range(X.shape[1])])
    yd = dm(y.astype(float))
    Zd = np.column_stack([dm(Z[:, j].astype(float)) for j in range(Z.shape[1])])
    r = sm.OLS(yd, Zd).fit(cov_type="HAC", cov_kwds={"maxlags": cov_lags})
    n, k = Zd.shape
    corr = np.sqrt((n - k) / max(n - k - ng, 1))
    return float(r.params[0]), float(r.bse[0] * corr), n


def add_met(s):
    s = s.copy()
    s["lvc2"] = s.lvc ** 2
    s["lpr"] = np.log1p(s.pr)
    return s


def design_frames(st_hourly, metd, metb, hart, eid, eid_days):
    """Yield (unit, frame) where unit is an hour (0-23), a block name, or 'daily'."""
    h = st_hourly.copy()
    h["date"] = h["time"].dt.normalize()
    h["hour"] = h["time"].dt.hour
    h = h.drop_duplicates(["date", "hour"])
    g = h.groupby("date").PM25
    daily = g.mean().where(g.count() >= 18).to_frame("PM25")
    units = {"daily": daily.join(metd, how="inner")}
    for blk, hs in BLOCKS.items():
        x = h[h.hour.isin(list(hs))].groupby("date").PM25
        x = x.mean().where(x.count() >= 4).to_frame("PM25")
        units[blk] = x.join(metb.xs(blk, level="block"), how="inner")
    for hh in range(24):
        blk = [b for b, hs in BLOCKS.items() if hh in hs][0]
        x = h[h.hour == hh].set_index("date")[["PM25"]]
        units[hh] = x.join(metb.xs(blk, level="block"), how="inner")
    for u, s in units.items():
        s = add_met(s[(s.PM25 > 0)].dropna(subset=["PM25", "lvc", "T", "rh", "pr", "wd_s", "wd_c"]))
        s["dow"] = s.index.dayofweek
        s["year"] = s.index.year
        s["ym"] = s.index.strftime("%Y-%m")
        s["covid"] = ((s.index >= COVID[0]) & (s.index <= COVID[1])).astype(int)
        s["eid"] = s.index.isin(list(eid_days)).astype(int)
        s["hart"] = s.index.isin(list(hart)).astype(int)
        yield u, s.sort_index()


def eid_effect(s, eid, y):
    s = s[(s.covid == 0) & (s.hart == 0)].copy()
    s["win"] = -1
    for i, t in enumerate(eid):
        m = (s.index >= t - pd.Timedelta(days=15)) & (s.index <= t + pd.Timedelta(days=18))
        s.loc[m, "win"] = i
    s = s[s.win >= 0]
    if s.eid.sum() < 10 or len(s) < 60:
        return None
    X = np.column_stack([s[METC].values] + [(s.dow == d).astype(float).values for d in range(1, 7)])
    return ols_fe(y(s), s.eid.values.astype(float), X, s.win.values)


def lockdown_effect(s, y):
    """Lockdown against what year, season and meteorology predict.

    Within a March-May window the lockdown indicator is identical to the 2020 year dummy,
    so a same-window design with year effects cannot identify it (an earlier version did
    exactly that and the pseudo-inverse split the effect arbitrarily).  Here the whole
    record is used: year effects are identified from the rest of each year, the seasonal
    shape from calendar-month effects shared across years, so the lockdown is the 2020
    March-May deviation from both.  Rows from 2020 outside the window anchor 2020's level;
    a station without them cannot be estimated and is skipped.
    """
    w = s[(s.hart == 0) & ((s.eid == 0) | (s.covid == 1))]
    in20 = w.index.year == 2020
    if w.covid.sum() < 15 or (in20 & (w.covid == 0)).sum() < 60 or (~in20).sum() < 300:
        return None
    X = np.column_stack([w[METC].values]
                        + [(w.dow == d).astype(float).values for d in range(1, 7)]
                        + [(w.index.month == mo).astype(float) for mo in range(2, 13)])
    return ols_fe(y(w), w.covid.values.astype(float), X, w.year.values)


def weekly_effect(s, dw, y):
    s = s[(s.covid == 0) & (s.eid == 0) & (s.hart == 0) & s.dow.isin([6, 0, 1, 2, dw])]
    if (s.dow == dw).sum() < 30:
        return None
    return ols_fe(y(s), (s.dow == dw).values.astype(float), s[METC].values, s.ym.values)


def main(root):
    metd, metb = met_daily(root)
    hart, eid, eid_days = calendars(root)
    val = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    ok_pm = set(val[(val.daily_r_embassy >= 0.6) & (val.best_lag_h.abs() <= 1)].station)
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[cams.station.isin(ok_pm) & (cams.time >= "2016-01-01")]
    emb = embassy_hourly(root)
    allh = pd.concat([cams[["station", "time", "PM25"]], emb], ignore_index=True)
    allh = allh[allh.time < "2026-01-01"]

    logy = lambda s: np.log(s.PM25.values)
    aqiy = lambda s: aqi_pm25(s.PM25.values)
    rows, aqi_rows = [], []
    for st, h in allh.groupby("station"):
        if st not in DHAKA:
            continue
        for u, s in design_frames(h, metd, metb, hart, eid, eid_days):
            for ev, fn in (("Eid", lambda s, y: eid_effect(s, eid, y)),
                           ("Lockdown", lockdown_effect)):
                r = fn(s, logy)
                if r:
                    rows.append(dict(station=st, unit=str(u), event=ev, b=r[0], se=r[1], n=r[2]))
            if u == "daily":
                for ev, fn in (("Eid", lambda s, y: eid_effect(s, eid, y)),
                               ("Lockdown", lockdown_effect),
                               ("Friday", lambda s, y: weekly_effect(s, 4, y)),
                               ("Saturday", lambda s, y: weekly_effect(s, 5, y))):
                    r = fn(s, aqiy)
                    rl = fn(s, logy)
                    if r and rl:
                        base = s[(s.covid == 0) & (s.eid == 0)].PM25
                        aqi_rows.append(dict(station=st, event=ev, aqi_pts=r[0], aqi_se=r[1],
                                             pm_pct=100 * (np.exp(rl[0]) - 1), n=r[2],
                                             mean_aqi=float(np.nanmean(aqi_pm25(base.values)))))
        print("done", st)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(OUT, "events_station_units.csv"), index=False)
    pooled = []
    for (ev, u), g in E.groupby(["event", "unit"]):
        b, se, tau2, i2, k = dl_pool(g.b, g.se)
        pooled.append(dict(event=ev, unit=u, pct=100 * (np.exp(b) - 1), lo=100 * (np.exp(b - 1.96 * se) - 1),
                           hi=100 * (np.exp(b + 1.96 * se) - 1), z=b / se, I2=i2, k=k,
                           stations=";".join(sorted(g.station))))
    P = pd.DataFrame(pooled)
    P.to_csv(os.path.join(OUT, "events_pooled.csv"), index=False)
    pd.set_option("display.width", 250)
    hrs = P[P.unit.str.isdigit()].copy()
    hrs["hour"] = hrs.unit.astype(int)
    print("\nPooled Dhaka hourly percent change")
    print(hrs.pivot(index="hour", columns="event", values="pct").round(1).T.to_string())
    print("\nBlocks and daily")
    print(P[~P.unit.str.isdigit()][["event", "unit", "pct", "lo", "hi", "z", "I2", "k"]].round(2).to_string(index=False))
    print("\nStation-level lockdown and Eid, daily percent")
    sd = E[E.unit == "daily"].copy()
    sd["pct"] = 100 * (np.exp(sd.b) - 1)
    print(sd.pivot(index="station", columns="event", values="pct").round(1).to_string())

    A = pd.DataFrame(aqi_rows)
    A.to_csv(os.path.join(OUT, "aqi_by_shock_station.csv"), index=False)
    ap = []
    for ev, g in A.groupby("event"):
        b, se, tau2, i2, k = dl_pool(g.aqi_pts, g.aqi_se)
        emb_row = g[g.station == "US Embassy"]
        ap.append(dict(event=ev, pooled_aqi_pts=b, lo=b - 1.96 * se, hi=b + 1.96 * se, k=k,
                       embassy_aqi_pts=float(emb_row.aqi_pts.iloc[0]) if len(emb_row) else np.nan,
                       embassy_se=float(emb_row.aqi_se.iloc[0]) if len(emb_row) else np.nan,
                       mean_aqi=g.mean_aqi.mean()))
    AP = pd.DataFrame(ap)
    AP.to_csv(os.path.join(OUT, "aqi_by_shock.csv"), index=False)
    print("\nAQI change per shock (index points)")
    print(AP.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
