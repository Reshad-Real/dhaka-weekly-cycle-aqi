"""Natural experiments at every Dhaka-region CAMS station, with same-site traffic tracers.

For each station with enough data, and for the US Embassy monitor alongside:

  weekly  Friday and Saturday against Sunday-Wednesday, daily means, year-month fixed effects
          and ERA5 meteorology, event days excluded, Newey-West (lag 7) errors.
  ratio   the Friday PM2.5 response divided by the Friday NOx (and CO) response at the same
          station.  The size of the Friday traffic cut cancels; what remains is how strongly
          PM2.5 tracks the surface traffic tracer relative to how strongly NOx does.
  blocks  the same Friday contrast on four diurnal blocks.
  events  lockdown (2020-03-26..05-30, against the same window in other years) and Eid
          (-1..+3 days, against the surrounding 4 weeks), with the same meteorology.

Writes 11_natexp/out/cams_weekly.csv, cams_blocks.csv, cams_events.csv.
Usage: python -I cams_shocks.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
MET = "lvc + I(lvc**2) + T + rh + np.log1p(pr) + wd_s + wd_c"
TRACERS = ("PM25", "PM10", "NOX", "NO2", "CO")
BLOCKS = {"night 00-05": range(0, 6), "morning 06-11": range(6, 12),
          "afternoon 12-17": range(12, 18), "evening 18-23": range(18, 24)}
COVID = (pd.Timestamp("2020-03-26"), pd.Timestamp("2020-05-30"))


def met_daily(root):
    m = pd.read_csv(os.path.join(root, "02_data", "raw", "meteo_dhaka_hourly_filled.csv"),
                    parse_dates=["time"])
    m["vc"] = np.maximum(m["wind_speed_10m"] / 3.6, 0.5) * np.maximum(m["boundary_layer_height"], 30)
    m["date"] = m["time"].dt.normalize()
    m["block"] = m["time"].dt.hour.map({h: b for b, hs in BLOCKS.items() for h in hs})
    agg = dict(vc=("vc", "mean"), T=("temperature_2m", "mean"), rh=("relative_humidity_2m", "mean"),
               pr=("precipitation", "sum"),
               wd_s=("wind_direction_10m", lambda x: np.sin(np.deg2rad(x)).mean()),
               wd_c=("wind_direction_10m", lambda x: np.cos(np.deg2rad(x)).mean()))
    d = m.groupby("date").agg(**agg)
    d["lvc"] = np.log(d["vc"])
    b = m.groupby(["date", "block"]).agg(**agg)
    b["lvc"] = np.log(b["vc"])
    return d, b


def calendars(root):
    ev = os.path.join(root, "Traffic", "data_availability", "02_event_calendars")
    hart = set(pd.read_csv(os.path.join(ev, "hartal_event_days_DERIVED.csv"), parse_dates=["date"])["date"])
    eid = pd.read_csv(os.path.join(ev, "eid_dates.csv"), parse_dates=["date"])["date"].tolist()
    eid_days = {t + pd.Timedelta(days=k) for t in eid for k in range(-1, 4)}
    return hart, eid, eid_days


def embassy_hourly(root):
    import glob
    fs = sorted(glob.glob(os.path.join(root, "02_data", "raw", "ground", "Dhaka_PM2.5_*_YTD.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d.rename(columns={"Date (LT)": "time", "Raw Conc.": "PM25", "QC Name": "qc"})
    d["time"] = pd.to_datetime(d["time"], format="%Y-%m-%d %I:%M %p")
    d = d[(d.qc == "Valid") & (d.PM25 > 0) & (d.PM25 < 1500)]
    d["station"] = "US Embassy"
    return d[["station", "time", "PM25"]]


def frame(h, metd, hart, eid_days, how="daily", metb=None):
    h = h.copy()
    h["date"] = h["time"].dt.normalize()
    if how == "daily":
        g = h.groupby("date")
        cols = [c for c in TRACERS if c in h]
        d = g[cols].mean()
        n = g[cols].count()
        d = d.where(n >= 18)
        d = d.join(metd, how="inner")
    else:
        h["block"] = h["time"].dt.hour.map({hh: b for b, hs in BLOCKS.items() for hh in hs})
        cols = [c for c in TRACERS if c in h]
        g = h.groupby(["date", "block"])
        d = g[cols].mean().where(g[cols].count() >= 4)
        d = d.join(metb, how="inner").reset_index().set_index("date")
    d["dow"] = d.index.dayofweek
    d["ym"] = d.index.strftime("%Y-%m")
    d["year"] = d.index.year
    d["event"] = (d.index.isin(list(hart)) | d.index.isin(list(eid_days)) |
                  ((d.index >= COVID[0]) & (d.index <= COVID[1])))
    d["eid"] = d.index.isin(list(eid_days)).astype(int)
    d["covid"] = ((d.index >= COVID[0]) & (d.index <= COVID[1])).astype(int)
    d["hart"] = d.index.isin(list(hart)).astype(int)
    d["fri"] = (d.dow == 4).astype(int)
    d["sat"] = (d.dow == 5).astype(int)
    d["thu"] = (d.dow == 3).astype(int)
    return d


def pct(b):
    return 100.0 * (np.exp(b) - 1.0)


def weekly(d, y):
    s = d[(~d.event) & d[y].notna() & (d[y] > 0)].copy()
    if len(s) < 400:
        return None
    s["ly"] = np.log(s[y])
    f = smf.ols("ly ~ fri + sat + thu + C(ym) + " + MET, data=s).fit(
        cov_type="HAC", cov_kwds={"maxlags": 7})
    return dict(n=len(s), fri=f.params["fri"], fri_se=f.bse["fri"], fri_p=f.pvalues["fri"],
                sat=f.params["sat"], sat_se=f.bse["sat"], sat_p=f.pvalues["sat"])


def events(d, y):
    out = {}
    s = d[d[y].notna() & (d[y] > 0) & (d.hart == 0)].copy()
    s["ly"] = np.log(s[y])
    # Lockdown against year, calendar-month and meteorology effects estimated on the whole
    # record.  (A same-window design with year effects is not identified: inside the window
    # the lockdown dummy equals the 2020 year dummy.  An earlier version made that mistake.)
    w = s[(s.eid == 0) | (s.covid == 1)].copy()
    w["month"] = w.index.month
    in20 = w.index.year == 2020
    if w.covid.sum() >= 20 and (in20 & (w.covid == 0)).sum() >= 60:
        f = smf.ols("ly ~ covid + C(year) + C(month) + C(dow) + " + MET, data=w).fit(
            cov_type="HAC", cov_kwds={"maxlags": 7})
        out["lockdown"] = (f.params["covid"], f.bse["covid"], int(w.covid.sum()))
    # Eid vs surrounding four weeks, outside the lockdown
    e = s[s.covid == 0]
    if e.eid.sum() >= 15:
        f = smf.ols("ly ~ eid + C(ym) + " + MET, data=e).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        out["eid"] = (f.params["eid"], f.bse["eid"], int(e.eid.sum()))
    return out


def main(root):
    metd, metb = met_daily(root)
    hart, eid, eid_days = calendars(root)
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[(cams.time >= "2019-01-01") & (cams.time < "2025-01-01")]
    emb = embassy_hourly(root)
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2025-01-01")]
    allh = pd.concat([cams, emb], ignore_index=True)
    stations = [s for s in allh.station.unique()
                if allh.loc[allh.station == s, "PM25"].notna().sum() > 0.25 * 6 * 8766]
    wk, bl, ev = [], [], []
    for st in sorted(stations):
        h = allh[allh.station == st]
        d = frame(h, metd, hart, eid_days)
        for y in TRACERS:
            if y not in d or d[y].notna().sum() < 400:
                continue
            r = weekly(d, y)
            if r:
                wk.append(dict(station=st, tracer=y, **r))
            for k, (b, se, n) in events(d, y).items():
                ev.append(dict(station=st, tracer=y, event=k, pct=pct(b), lo=pct(b - 1.96 * se),
                               hi=pct(b + 1.96 * se), n_event_days=n))
        db = frame(h, metd, hart, eid_days, how="blocks", metb=metb)
        for blk in BLOCKS:
            sb = db[db.block == blk]
            for y in ("PM25", "NOX", "CO"):
                if y not in sb or sb[y].notna().sum() < 400:
                    continue
                r = weekly(sb, y)
                if r:
                    bl.append(dict(station=st, block=blk, tracer=y, **r))
    W = pd.DataFrame(wk)
    W["fri_pct"], W["fri_lo"], W["fri_hi"] = pct(W.fri), pct(W.fri - 1.96 * W.fri_se), pct(W.fri + 1.96 * W.fri_se)
    W["sat_pct"] = pct(W.sat)
    W.to_csv(os.path.join(OUT, "cams_weekly.csv"), index=False)
    B = pd.DataFrame(bl)
    B["fri_pct"] = pct(B.fri)
    B.to_csv(os.path.join(OUT, "cams_blocks.csv"), index=False)
    E = pd.DataFrame(ev)
    E.to_csv(os.path.join(OUT, "cams_events.csv"), index=False)
    pd.set_option("display.width", 250)
    print("WEEKLY: Friday and Saturday vs Sun-Wed, percent")
    print(W.pivot_table(index="station", columns="tracer", values="fri_pct").round(1).to_string())
    print("\nFriday p-values")
    print(W.pivot_table(index="station", columns="tracer", values="fri_p").round(3).to_string())
    print("\nSaturday percent")
    print(W.pivot_table(index="station", columns="tracer", values="sat_pct").round(1).to_string())
    # same-site ratio test
    rows = []
    for st, g in W.groupby("station"):
        g = g.set_index("tracer")
        for tr in ("NOX", "CO"):
            if "PM25" in g.index and tr in g.index:
                rows.append(dict(station=st, tracer=tr, pm_pct=g.loc["PM25", "fri_pct"],
                                 pm_lo=g.loc["PM25", "fri_lo"], pm_hi=g.loc["PM25", "fri_hi"],
                                 tr_pct=g.loc[tr, "fri_pct"], tr_p=g.loc[tr, "fri_p"],
                                 ratio=g.loc["PM25", "fri_pct"] / g.loc[tr, "fri_pct"]))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "cams_ratio_test.csv"), index=False)
    print("\nSAME-SITE RATIO TEST (Friday PM2.5 % / Friday tracer %)")
    print(R.round(3).to_string(index=False))
    print("\nBLOCKS: Friday percent by diurnal block")
    print(B.pivot_table(index=["station", "tracer"], columns="block", values="fri_pct").round(1).to_string())
    print("\nEVENTS")
    print(E.pivot_table(index=["station", "tracer"], columns="event", values="pct").round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
