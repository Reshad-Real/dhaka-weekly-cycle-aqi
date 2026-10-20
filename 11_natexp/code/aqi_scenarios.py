"""Illustrative scenarios: what concentration reductions shaped like the observed weekly
pattern would mean for the daily PM2.5 AQI sub-index.

These rescale observed concentrations.  They are not forecasts of a policy: a sustained
measure could shift emissions in time, change chemistry or transport, or change behaviour,
and a concentration cut is not an emissions cut.

Baseline (a working-day counterfactual).  The weekly effects are estimated against Sunday to
Wednesday, so they are applied to a record in which every day is a Sunday-Wednesday day:
Thursday, Friday and Saturday hours of the US Embassy record are divided by their own pooled
Dhaka-region block effects.  Eid windows, lockdown days and strike days are left out, as in
the weekly design.

  rest_day        every day takes the rest-day block profile: the Friday morning, afternoon
                  and evening blocks, and the night that follows Friday (the Saturday night
                  block, 00-05 h).  Friday's own night block is Thursday night, whose
                  pre-weekend activity is not part of a rest day.
  rest_day_calendar  sensitivity: the four blocks of calendar Friday
  cut_allday      the pooled Friday afternoon effect as a concentration cut at all hours
  cut_day_only    the same cut at 12-23 h only
  cut_night_only  the same cut at 00-11 h only
Intervals come from the joint-date bootstrap draws of the block effects (robustness.py), so
the effects move together as they do in the data; each draw also rebuilds the baseline.

Index: daily mean of >= 18 valid hours, truncated to 0.1 ug m-3, mapped to the PM2.5
sub-index on the US EPA 2024 breakpoints (unrounded; values above 325.4 set to 500).  Days
in a category: sub-index above 150 (Unhealthy or worse), 200 (Very Unhealthy or worse) and
300 (Hazardous), per year of valid days.

Writes out/aqi_scenarios.csv and out/aqi_scenarios_stations.csv.
Usage: python -I aqi_scenarios.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_shocks import BLOCKS, COVID, calendars, embassy_hourly  # noqa: E402
from events_hourly import aqi_pm25  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
WHO = {"AQG": 5, "IT4": 10, "IT3": 15, "IT2": 25, "IT1": 35}
BL = list(BLOCKS)
BLOCK_OF_HOUR = np.array([[i for i, b in enumerate(BL) if h in BLOCKS[b]][0] for h in range(24)])
SCEN = ["rest_day", "rest_day_calendar", "cut_allday", "cut_day_only", "cut_night_only"]


def effects(table):
    """{(day, block index): beta} for Dhaka-region PM2.5 from a bootstrap table or draw."""
    t = table[(table.region == "Dhaka") & (table.tracer == "PM25") & table.block.isin(BL)]
    return {(r.day, BL.index(r.block)): r.b for r in t.itertuples()}


def hourly_mult(eff):
    """Per-hour multipliers for the baseline conversion and for each scenario."""
    blk = BLOCK_OF_HOUR
    conv = {d: np.exp(np.array([eff[(d, b)] for b in blk])) for d in ("Thu", "Fri", "Sat")}
    rest = np.exp(np.array([eff[("Sat", 0)] if b == 0 else eff[("Fri", b)] for b in blk]))
    cal = np.exp(np.array([eff[("Fri", b)] for b in blk]))
    cut = np.exp(eff[("Fri", 2)])
    hrs = np.arange(24)
    return conv, {"rest_day": rest, "rest_day_calendar": cal, "cut_allday": np.full(24, cut),
                  "cut_day_only": np.where(hrs >= 12, cut, 1.0), "cut_night_only": np.where(hrs < 12, cut, 1.0)}


def prepare(h, excl):
    x = h[["time", "PM25"]].dropna().drop_duplicates("time")
    x = x[x.PM25 > 0].copy()
    x["date"] = x.time.dt.normalize()
    x = x[~x.date.isin(excl)]
    x["hour"] = x.time.dt.hour.values
    x["dow"] = x.time.dt.dayofweek.values
    codes, dates = pd.factorize(x.date)
    return x.PM25.values, x.hour.values, x.dow.values, codes, len(dates)


def metrics(pm, codes, nd):
    cnt = np.bincount(codes, minlength=nd)
    d = np.bincount(codes, pm, nd) / np.maximum(cnt, 1)
    d = d[cnt >= 18]
    a = aqi_pm25(d)
    years = len(d) / 365.25
    return dict(n_days=len(d), mean_pm=float(d.mean()), mean_aqi=float(np.nanmean(a)),
                days150=float((a > 150).sum() / years), days200=float((a > 200).sum() / years),
                days300=float((a > 300).sum() / years))


def run(prep, eff):
    pm, hour, dow, codes, nd = prep
    conv, scen = hourly_mult(eff)
    base = pm.copy()
    for d, k in (("Thu", 3), ("Fri", 4), ("Sat", 5)):
        m = dow == k
        base[m] = base[m] / conv[d][hour[m]]
    out = {"baseline": metrics(base, codes, nd)}
    for s in SCEN:
        out[s] = metrics(base * scen[s][hour], codes, nd)
    return out


def main(root):
    hart, eid, eid_days = calendars(root)
    excl = set(hart) | set(eid_days) | set(pd.date_range(COVID[0], COVID[1]))
    pts = effects(pd.read_csv(os.path.join(OUT, "robust_bootstrap_days.csv")).rename(columns={"b_point": "b"}))
    draws = pd.read_csv(os.path.join(OUT, "robust_bootstrap_draws.csv.gz"))
    draws = draws[(draws.region == "Dhaka") & (draws.tracer == "PM25")]
    emb = embassy_hourly(root)
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2026-01-01")]
    P = prepare(emb, excl)
    point = run(P, pts)
    sims = {s: [] for s in SCEN}
    for it, g in draws.groupby("it"):
        e = effects(g)
        if len(e) < 12:
            continue
        r = run(P, e)
        for s in SCEN:
            sims[s].append({k: r[s][k] - r["baseline"][k] for k in ("mean_aqi", "days150", "days200", "days300")}
                           | {"pm_pct": 100 * (r[s]["mean_pm"] / r["baseline"]["mean_pm"] - 1)})
    rows = []
    b = point["baseline"]
    rows.append(dict(scenario="baseline", **b))
    for s in SCEN:
        r = point[s]
        row = dict(scenario=s, **r, d_mean_aqi=r["mean_aqi"] - b["mean_aqi"],
                   d_pm_pct=100 * (r["mean_pm"] / b["mean_pm"] - 1),
                   **{"d_" + t: r[t] - b[t] for t in ("days150", "days200", "days300")})
        D = pd.DataFrame(sims[s])
        for k, col in (("mean_aqi", "d_mean_aqi"), ("days150", "d_days150"), ("days200", "d_days200"),
                       ("days300", "d_days300"), ("pm_pct", "d_pm_pct")):
            row[col + "_lo"] = float(np.percentile(D[k], 2.5))
            row[col + "_hi"] = float(np.percentile(D[k], 97.5))
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "aqi_scenarios.csv"), index=False)
    pd.set_option("display.width", 250)
    print("US Embassy, working-day counterfactual baseline; %d bootstrap draws" % len(sims["rest_day"]))
    print(T.round(2).to_string(index=False))

    # replication on the screened Dhaka CAMS stations, point effects
    val = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    ok = set(val[(val.daily_r_embassy >= 0.6) & (val.best_lag_h.abs() <= 1)].station)
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[cams.station.isin(ok) & (cams.time >= "2019-01-01") & (cams.time < "2025-01-01")]
    srows = []
    for st, h in cams.groupby("station"):
        r = run(prepare(h, excl), pts)
        for s in SCEN:
            srows.append(dict(station=st, scenario=s, base_aqi=r["baseline"]["mean_aqi"],
                              d_mean_aqi=r[s]["mean_aqi"] - r["baseline"]["mean_aqi"],
                              d_days150=r[s]["days150"] - r["baseline"]["days150"],
                              d_days200=r[s]["days200"] - r["baseline"]["days200"]))
    S = pd.DataFrame(srows)
    S.to_csv(os.path.join(OUT, "aqi_scenarios_stations.csv"), index=False)
    print(S.pivot(index="station", columns="scenario", values="d_mean_aqi").round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
