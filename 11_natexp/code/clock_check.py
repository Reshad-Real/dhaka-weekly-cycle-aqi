"""Station clock check (the basis for the Savar exclusion).

A timestamp offset moves every instrument at a station together, sunlight included; a
transported or locally different signal moves one pollutant only. For each station this
writes, from 2019 onwards:
  sr_centroid_h   the solar-radiation-weighted mean hour of the day (local, hour-beginning)
  lag_<tracer>    for PM2.5, NOx and CO in the Dhaka region, the hourly lag (-4..+4 h) at
                  which the within-day log anomalies correlate best with Darussalam, and
                  that correlation (positive lag: the station follows Darussalam)
Output: out/clock_check.csv
Usage: python -I clock_check.py <FYDP_D root>
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
DHAKA = ["BARC", "Darussalam", "DoE", "Gazipur", "Narayanganj", "Narsingdi", "Savar"]
REF = "Darussalam"
LAGS = range(-4, 5)
MIN_SR_HOURS = 5000


def anomalies(d, st, var):
    x = d.loc[d.station == st].set_index("time")[var]
    x = np.log(x.where(x > 0))
    return x - x.groupby(x.index.normalize()).transform("mean")


def best_lag(a, b):
    r = {}
    for L in LAGS:
        x = pd.concat([a, b.shift(L, freq="h")], axis=1, sort=True).dropna()
        if len(x) > 1000:
            r[L] = x.corr().iloc[0, 1]
    if not r:
        return np.nan, np.nan
    L = max(r, key=r.get)
    return L, r[L]


def main():
    d = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"),
                    usecols=["station", "time", "PM25", "NOX", "CO", "Solar Rad", "SR"])
    d["time"] = pd.to_datetime(d.time)
    d = d[d.time >= "2019-01-01"]
    d["sr"] = d["Solar Rad"].fillna(d["SR"]).clip(lower=0)
    rows = []
    for st in sorted(d.station.unique()):
        g = d[d.station == st]
        n = int(g.sr.notna().sum())
        prof = g.groupby(g.time.dt.hour).sr.mean()
        row = dict(station=st, sr_hours=n,
                   sr_centroid_h=(prof * prof.index).sum() / prof.sum() if n >= MIN_SR_HOURS else np.nan)
        if st in DHAKA and st != REF:
            for y in ("PM25", "NOX", "CO"):
                row["lag_" + y], row["r_" + y] = best_lag(anomalies(d, st, y), anomalies(d, REF, y))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "clock_check.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
