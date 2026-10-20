"""Sensitivity of the PM2.5 sub-index results to the cap at 500.

Daily means above 325.4 ug m-3, beyond the top of the 2024 breakpoint table, are set to 500
in every sub-index result.  Here the Friday, Saturday, Eid and lockdown effects on the daily
sub-index at the Dhaka-region stations are re-estimated with the top segment of the table
(301-500 over 225.5-325.4 ug m-3) extended linearly instead, and the number of station-days
above 325.4 in each analysis sample is counted.  Designs as in robustness.py (weekly),
events_checks.py (Eid, lockdown); pooled by DerSimonian-Laird with Newey-West station errors
(a sensitivity check on the point estimates; the main intervals are dependence-aware).

Writes out/aqi_cap.csv.
Usage: python -I aqi_cap.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robustness as rb  # noqa: E402
from cams_pooled import DHAKA, dl_pool  # noqa: E402
from cams_shocks import calendars, embassy_hourly, met_daily  # noqa: E402
from events_checks import eid_fit, eid_stack, lock_fit, lock_frame  # noqa: E402
from events_hourly import aqi_pm25, design_frames  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")


def daily_frames(root):
    metd, metb = met_daily(root)
    hart, eid, eid_days = calendars(root)
    val = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    ok_pm = set(val[(val.daily_r_embassy >= 0.6) & (val.best_lag_h.abs() <= 1)].station)
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[cams.station.isin(ok_pm) & (cams.time >= "2016-01-01")]
    emb = embassy_hourly(root)
    allh = pd.concat([cams[["station", "time", "PM25"]], emb], ignore_index=True)
    allh = allh[allh.time < "2026-01-01"]
    daily = {}
    for st, h in allh.groupby("station"):
        if st not in DHAKA or st == "Savar":
            continue
        for u, s in design_frames(h, metd, metb, hart, eid, eid_days):
            if u == "daily":
                daily[st] = s
                break
    return daily, eid


def capped(c):
    return int((np.floor(np.asarray(c, float) * 10) / 10 > 325.4).sum())


def main(root):
    daily, eid = daily_frames(root)
    rows = []
    for cap in (True, False):
        y = lambda w, cap=cap: aqi_pm25(w.PM25.values, cap=cap)  # noqa: E731
        fits = {"Friday": {}, "Saturday": {}, "Eid": {}, "Lockdown": {}}
        ncap = {k: 0 for k in fits}
        nday = {k: 0 for k in fits}
        for st, s in daily.items():
            wk = s[(s.index >= "2019-01-01") & (s.covid == 0) & (s.eid == 0) & (s.hart == 0)].copy()
            wk["ly"] = y(wk)
            for name, dw in (("Friday", 4), ("Saturday", 5)):
                r = rb.fit(wk.dropna(subset=["ly"] + rb.METC), dw)
                if r:
                    fits[name][st] = r
                    sub = wk[wk.dow.isin(rb.WORK + [dw])]
                    ncap[name] += capped(sub.PM25)
                    nday[name] += len(sub)
            r = eid_fit(s, eid, y)
            if r:
                fits["Eid"][st] = r
                W = eid_stack(s, eid)
                ncap["Eid"] += capped(W.PM25)
                nday["Eid"] += len(W)
            r = lock_fit(s, y)
            if r:
                fits["Lockdown"][st] = r
                W = lock_frame(s)
                ncap["Lockdown"] += capped(W.PM25)
                nday["Lockdown"] += len(W)
        for name, f in fits.items():
            b, se, _, _, k = dl_pool(pd.Series([v[0] for v in f.values()]), pd.Series([v[1] for v in f.values()]))
            rows.append(dict(shock=name, version="capped" if cap else "linear top segment", k=k, b=b,
                             lo=b - 1.96 * se, hi=b + 1.96 * se, station_days=nday[name],
                             days_above_325=ncap[name]))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "aqi_cap.csv"), index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
