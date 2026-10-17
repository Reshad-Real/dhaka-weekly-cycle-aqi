"""Why does PM2.5 rise at Narsingdi over Eid when it falls at the Dhaka-city sites?

Two candidate explanations are checked:
  (a) a real local increase, e.g. the Eid exodus raising traffic on the Dhaka-Sylhet
      highway past Narsingdi; this predicts a rise in NOx and CO as well, recurring across
      events and concentrated before Eid (outbound) and after it (return);
  (b) a data artefact: one or two anomalous events, or hours the QC did not catch.

Design (well-posed, unlike a per-event fit): for each station and outcome, all Eid windows
(-15..+18 days, outside the lockdown, strike days dropped) are stacked and log(daily mean)
is regressed on window fixed effects, day-of-week effects and ERA5 daily meteorology, with
either (i) one indicator per event for days -1..+3, or (ii) one indicator per relative day
-5..+8 shared across events.  Raw daily means around each Eid are printed for Narsingdi
next to the Embassy and Gazipur.
Writes out/eid_by_event.csv and out/eid_relative_day.csv.
Usage: python -I narsingdi_eid.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_shocks import COVID, calendars, embassy_hourly, met_daily  # noqa: E402
from events_hourly import METC  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
SITES = ["Narsingdi", "Gazipur", "Savar", "Narayanganj", "Darussalam", "BARC", "DoE", "US Embassy"]


def stacked(d, metd, eid, hart):
    rows = []
    for i, t in enumerate(eid):
        if COVID[0] <= t <= COVID[1]:
            continue
        w = d.loc[t - pd.Timedelta(days=15): t + pd.Timedelta(days=18)]
        w = w[~w.index.isin(list(hart))]
        if len(w) < 15:
            continue
        w = w.to_frame("ly").join(metd, how="inner")
        w["win"], w["eid_date"], w["rel"] = i, t.date(), (w.index - t).days
        rows.append(w)
    if not rows:
        return None
    s = pd.concat(rows)
    s["lvc2"] = s.lvc ** 2
    s["lpr"] = np.log1p(s.pr)
    return s.dropna(subset=["ly"] + METC)


def fit(s, cols):
    X = pd.concat([s[METC], pd.get_dummies(s.index.dayofweek, prefix="d", drop_first=True).set_index(s.index),
                   pd.get_dummies(s.win, prefix="w", drop_first=False).set_index(s.index), cols], axis=1).astype(float)
    r = sm.OLS(s.ly.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    names = list(X.columns)
    return {c: (r.params[names.index(c)], r.bse[names.index(c)]) for c in cols.columns}


def main(root):
    metd, _ = met_daily(root)
    hart, eid, _ = calendars(root)
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    emb = embassy_hourly(root)
    allh = pd.concat([cams[cams.station.isin(SITES)], emb], ignore_index=True)
    allh["date"] = allh.time.dt.normalize()
    ev, rel, daily = [], [], {}
    for st, h in allh.groupby("station"):
        for tracer in ("PM25", "NOX", "CO"):
            if tracer not in h or h[tracer].notna().sum() < 1000:
                continue
            g = h.groupby("date")[tracer]
            d = g.mean().where(g.count() >= 18).dropna()
            d = d[d > 0]
            daily[(st, tracer)] = d
            s = stacked(np.log(d), metd, eid, hart)
            if s is None or s.win.nunique() < 2:
                continue
            core = (s.rel >= -1) & (s.rel <= 3)
            ind = pd.DataFrame({"e_%s" % e: (core & (s.eid_date == e)).astype(float)
                                for e in sorted(s.eid_date.unique())}, index=s.index)
            ind = ind.loc[:, ind.sum() >= 2]
            for c, (b, se) in fit(s, ind).items():
                ev.append(dict(station=st, tracer=tracer, eid=c[2:], pct=100 * (np.exp(b) - 1),
                               lo=100 * (np.exp(b - 1.96 * se) - 1), hi=100 * (np.exp(b + 1.96 * se) - 1)))
            rd = pd.DataFrame({"r%+d" % k: (s.rel == k).astype(float) for k in range(-5, 9)}, index=s.index)
            rd = rd.loc[:, rd.sum() >= 2]
            for c, (b, se) in fit(s, rd).items():
                rel.append(dict(station=st, tracer=tracer, rel=int(c[1:]), pct=100 * (np.exp(b) - 1),
                                se=se))
    E, R = pd.DataFrame(ev), pd.DataFrame(rel)
    E.to_csv(os.path.join(OUT, "eid_by_event.csv"), index=False)
    R.to_csv(os.path.join(OUT, "eid_relative_day.csv"), index=False)
    pd.set_option("display.width", 250)
    print("Eid core window (-1..+3) by event, percent")
    print(E.pivot_table(index=["station", "tracer"], columns="eid", values="pct").round(0).to_string())
    print("\nAnomaly by day relative to Eid, percent (shared across events)")
    print(R.pivot_table(index=["station", "tracer"], columns="rel", values="pct").round(0).to_string())
    print("\nRaw daily means around each Eid (Narsingdi PM2.5 | NOx | CO | Embassy PM2.5 | Gazipur PM2.5)")
    for t in eid:
        if t.year < 2022:
            continue
        idx = pd.date_range(t - pd.Timedelta(days=4), t + pd.Timedelta(days=5))
        get = lambda k: daily.get(k, pd.Series(dtype=float)).reindex(idx).round(0).tolist()
        print(t.date())
        for k in (("Narsingdi", "PM25"), ("Narsingdi", "NOX"), ("Narsingdi", "CO"),
                  ("US Embassy", "PM25"), ("Gazipur", "PM25")):
            print("   %-22s %s" % ("%s %s" % k, get(k)))


if __name__ == "__main__":
    main(sys.argv[1])
