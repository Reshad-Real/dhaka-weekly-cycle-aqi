"""Activity and PM2.5 responses to Eid, the lockdown and strike days in the present
designs, replacing the previous analysis's TROPOMI values quoted in Table 2.

  Eid       each window (-1..+3) against its own -15..+18 days; window and day-of-week
            fixed effects; ERA5 daily meteorology (events_hourly.eid_effect)
  Lockdown  against year and calendar-month effects from the whole record, day-of-week
            effects and meteorology (events_hourly.lockdown_effect)
  Strikes   strike days against all other non-event days; year-month and day-of-week
            fixed effects and meteorology
  Friday, Saturday  the weekly design (events_hourly.weekly_effect), for completeness

Outcomes: TROPOMI tropospheric NO2 (domain mean, daily); daily-mean PM2.5 pooled over the
screened Dhaka-region stations; daily-mean NOx and CO pooled over gas-valid stations
(Dhaka and national).  Writes out/shock_activity.csv.
Usage: python -I tropomi_events.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_shocks import COVID, calendars, embassy_hourly, met_daily  # noqa: E402
from cams_pooled import DHAKA, dl_pool  # noqa: E402
from events_hourly import METC, eid_effect, lockdown_effect, ols_fe, weekly_effect  # noqa: E402
from screens import gas_allowed, screens  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")


def frame(series, metd, hart, eid_days):
    s = series.to_frame("y").join(metd, how="inner")
    s = s[s.y > 0].copy()
    s["lvc2"] = s.lvc ** 2
    s["lpr"] = np.log1p(s.pr)
    s = s.dropna(subset=["y"] + METC)
    s["dow"] = s.index.dayofweek
    s["year"] = s.index.year
    s["ym"] = s.index.strftime("%Y-%m")
    s["covid"] = ((s.index >= COVID[0]) & (s.index <= COVID[1])).astype(int)
    s["eid"] = s.index.isin(list(eid_days)).astype(int)
    s["hart"] = s.index.isin(list(hart)).astype(int)
    return s.sort_index()


def strike_effect(s, y):
    w = s[(s.covid == 0) & (s.eid == 0)]
    if w.hart.sum() < 10:
        return None
    X = np.column_stack([w[METC].values] + [(w.dow == d).astype(float).values for d in range(1, 7)])
    return ols_fe(y(w), w.hart.values.astype(float), X, w.ym.values)


def main(root):
    metd, _ = met_daily(root)
    hart, eid, eid_days = calendars(root)
    ly = lambda s: np.log(s.y.values)
    designs = {"Friday": lambda s: weekly_effect(s, 4, ly),
               "Saturday": lambda s: weekly_effect(s, 5, ly),
               "Eid": lambda s: eid_effect(s, eid, ly),
               "Lockdown": lambda s: lockdown_effect(s, ly),
               "Strike days": lambda s: strike_effect(s, ly)}
    rows = []

    tr = pd.read_csv(os.path.join(root, "02_data", "raw", "tropomi_no2_dhaka_daily.csv"), parse_dates=["date"])
    tr = tr.set_index("date")["tropospheric_NO2_column_number_density_mean"] * 1e6
    tr = tr.loc["2019-01-01":"2025-12-31"]
    s = frame(tr, metd, hart, eid_days)
    for name, fn in designs.items():
        r = fn(s)
        if r:
            rows.append(dict(shock=name, outcome="TROPOMI NO2", region="domain", k=1, b=r[0], se=r[1], n=r[2],
                             n_event=int({"Eid": s.eid, "Lockdown": s.covid, "Strike days": s.hart}.get(
                                 name, (s.dow == (4 if name == "Friday" else 5)).astype(int)).sum())))

    ok_pm, gas_ok, flat = screens()
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[cams.time >= "2016-01-01"]
    emb = embassy_hourly(root)
    allh = pd.concat([cams, emb], ignore_index=True)
    allh = allh[allh.time < "2026-01-01"]
    allh["date"] = allh.time.dt.normalize()
    station_rows = []
    for st, h in allh.groupby("station"):
        reg = "Dhaka" if st in DHAKA else "national"
        for tracer in ("PM25", "NOX", "CO"):
            if tracer not in h:
                continue
            if tracer == "PM25" and reg == "Dhaka" and st != "US Embassy" and st not in ok_pm:
                continue
            if tracer in ("NOX", "CO") and not gas_allowed(st, tracer, reg, gas_ok, flat):
                continue
            g = h.groupby("date")[tracer]
            d = g.mean().where(g.count() >= 18).dropna()
            if len(d) < 300:
                continue
            s_all = frame(d, metd, hart, eid_days)
            for name, fn in designs.items():
                if tracer != "PM25" and name == "Lockdown":
                    continue        # gas analyzers are not comparable across years
                if reg == "national" and name == "Lockdown":
                    # The lockdown is read against each station's own 2020 level outside the
                    # window, so a level shift between years biases it.  Dhaka-region PM2.5
                    # stations are screened against the Embassy reference; the national ones
                    # have no reference, and the two with 2020 data (both in Chattogram) show a
                    # shift: Agrabad's January-February mean halves from 2019 to 2020.
                    continue
                # the lockdown needs earlier years to anchor the seasonal shape; every other
                # design uses 2019 onwards, the same record as the main tables
                s = s_all if name == "Lockdown" else s_all[s_all.index >= "2019-01-01"]
                if len(s) < 200:
                    continue
                r = fn(s)
                if r:
                    station_rows.append(dict(station=st, region=reg, tracer=tracer, shock=name,
                                             b=r[0], se=r[1], n=r[2]))
    S = pd.DataFrame(station_rows)
    S.to_csv(os.path.join(OUT, "shock_activity_station.csv"), index=False)
    for (reg, tracer, name), g in S.groupby(["region", "tracer", "shock"]):
        b, se, tau2, i2, k = dl_pool(g.b, g.se)
        rows.append(dict(shock=name, outcome=tracer, region=reg, k=k, b=b, se=se, n=int(g.n.sum()),
                         I2=i2, stations=";".join(sorted(g.station))))
    T = pd.DataFrame(rows)
    f = lambda v: 100 * (np.exp(v) - 1)
    T["pct"], T["lo"], T["hi"] = f(T.b), f(T.b - 1.96 * T.se), f(T.b + 1.96 * T.se)
    T["z"] = T.b / T.se
    T.to_csv(os.path.join(OUT, "shock_activity.csv"), index=False)
    pd.set_option("display.width", 220)
    print(T[["shock", "outcome", "region", "k", "pct", "lo", "hi", "z", "n"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
