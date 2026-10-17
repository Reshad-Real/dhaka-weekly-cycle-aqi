"""What activity reductions of the observed kinds would buy in the daily AQI (OUTLINE.md
next step 1).

Each scenario rescales the observed hourly PM2.5 record by an hour-of-day multiplier
taken from the pooled Dhaka-region estimates, recomputes daily means (>= 18 valid hours)
and the daily AQI (US EPA 2024 / Bangladesh breakpoints), and reports the change in the
mean index, in annual mean PM2.5 and in days per year at or above 150, 200 and 300.

  rest_day         every day behaves like the rest day: Friday 06-23 h and the following
                   night (Saturday 00-05 h), so the pre-weekend Thursday night carried by
                   calendar Friday 00-05 is not imposed on every day
  rest_day_lo/hi   the same with every hour at its pooled 95% bound (a conservative
                   envelope; hours are not jointly at their bounds in reality)
  cut_allday       the pooled Friday afternoon effect applied to all 24 hours: what a
                   round-the-clock measure of the same hourly strength would buy
  cut_day_only     the same cut on 12-23 h only (the hours the working week reaches)
  cut_night_only   the same cut on 00-11 h only (the hours it does not)
  eid_like         the pooled Eid hourly profile on every day (context: whole-city collapse)
  lockdown_like    the pooled lockdown hourly profile on every day (context)

Applied to the US Embassy record (2019-2025) and, for replication, to each screened Dhaka
CAMS station (2019-2024).  Writes out/aqi_scenarios.csv and out/aqi_scenarios_stations.csv.
Usage: python -I aqi_scenarios.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_shocks import embassy_hourly  # noqa: E402
from events_hourly import aqi_pm25  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
WHO = {"AQG": 5, "IT4": 10, "IT3": 15, "IT2": 25, "IT1": 35}


def multipliers():
    P = pd.read_csv(os.path.join(OUT, "hourly_profiles.csv"))
    P = P[(P.region == "Dhaka") & (P.tracer == "PM25")]
    fri = P[P.day == "Fri"].set_index("hour")
    sat = P[P.day == "Sat"].set_index("hour")
    rest = {k: np.r_[sat.loc[0:5, k].values, fri.loc[6:23, k].values] for k in ("pct", "lo", "hi")}
    R = pd.read_csv(os.path.join(OUT, "robust_bootstrap.csv"))
    aft = float(R[(R.region == "Dhaka") & (R.tracer == "PM25") & (R.block == "afternoon 12-17")].pct.iloc[0])
    E = pd.read_csv(os.path.join(OUT, "events_pooled.csv"))
    E = E[E.unit.astype(str).str.isdigit()].copy()
    E["hour"] = E.unit.astype(int)
    eid = E[E.event == "Eid"].set_index("hour").pct.reindex(range(24)).values
    lock = E[E.event == "Lockdown"].set_index("hour").pct.reindex(range(24)).values
    hrs = np.arange(24)
    M = {"baseline": np.zeros(24),
         "rest_day": rest["pct"], "rest_day_lo": rest["lo"], "rest_day_hi": rest["hi"],
         "cut_allday": np.full(24, aft),
         "cut_day_only": np.where(hrs >= 12, aft, 0.0),
         "cut_night_only": np.where(hrs < 12, aft, 0.0),
         "eid_like": eid, "lockdown_like": lock}
    return {k: 1 + v / 100.0 for k, v in M.items()}, aft


def evaluate(h, mult):
    """h: hourly frame with time, PM25.  Returns metrics for one multiplier vector."""
    x = h.copy()
    x["pm"] = x.PM25 * mult[x.time.dt.hour.values]
    x["date"] = x.time.dt.normalize()
    g = x.groupby("date").pm
    d = g.mean()[g.count() >= 18]
    a = aqi_pm25(d.values)
    years = len(d) / 365.25
    return dict(n_days=len(d), mean_pm=float(d.mean()), mean_aqi=float(np.nanmean(a)),
                days150=float((a >= 150).sum() / years), days200=float((a >= 200).sum() / years),
                days300=float((a >= 300).sum() / years))


def main(root):
    M, aft = multipliers()
    emb = embassy_hourly(root)
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2026-01-01")].drop_duplicates("time")
    rows = []
    base = evaluate(emb, M["baseline"])
    for k, m in M.items():
        r = evaluate(emb, m)
        rows.append(dict(scenario=k, **r, d_mean_aqi=r["mean_aqi"] - base["mean_aqi"],
                         d_pm_pct=100 * (r["mean_pm"] / base["mean_pm"] - 1),
                         **{"d_" + t: r[t] - base[t] for t in ("days150", "days200", "days300")}))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "aqi_scenarios.csv"), index=False)
    pd.set_option("display.width", 250)
    print("US Embassy 2019-2025; afternoon cut used = %.2f%%" % aft)
    print(T[["scenario", "mean_pm", "d_pm_pct", "mean_aqi", "d_mean_aqi", "days150", "d_days150",
             "days200", "d_days200", "days300", "d_days300"]].round(2).to_string(index=False))
    print("\nWHO ladder (annual mean PM2.5, ug/m3):", WHO)

    # replication on the screened Dhaka CAMS stations
    val = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    ok = set(val[(val.daily_r_embassy >= 0.6) & (val.best_lag_h.abs() <= 1)].station)
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[cams.station.isin(ok) & (cams.time >= "2019-01-01") & (cams.time < "2025-01-01")]
    srows = []
    for st, h in cams.groupby("station"):
        h = h[["time", "PM25"]].dropna().drop_duplicates("time")
        h = h[h.PM25 > 0]
        b = evaluate(h, M["baseline"])
        for k in ("rest_day", "cut_allday", "cut_day_only", "cut_night_only"):
            r = evaluate(h, M[k])
            srows.append(dict(station=st, scenario=k, base_aqi=b["mean_aqi"],
                              d_mean_aqi=r["mean_aqi"] - b["mean_aqi"],
                              d_days150=r["days150"] - b["days150"], d_days200=r["days200"] - b["days200"]))
    S = pd.DataFrame(srows)
    S.to_csv(os.path.join(OUT, "aqi_scenarios_stations.csv"), index=False)
    print("\nDhaka CAMS stations, change in mean AQI by scenario")
    print(S.pivot(index="station", columns="scenario", values="d_mean_aqi").round(2).to_string())
    print("\nchange in days/yr >= 150")
    print(S.pivot(index="station", columns="scenario", values="d_days150").round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
