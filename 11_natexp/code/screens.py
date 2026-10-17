"""Station screening shared by every analysis, so all scripts use the same rules.

  PM2.5 (Dhaka region): daily log correlation with the US Embassy >= 0.6 and best hourly
         lag within one hour (cams_validate.py).  Stations outside the Dhaka region have
         no reference monitor and are not screened on PM2.5.
  Gases (Dhaka region): deseasonalized daily Spearman correlation with TROPOMI NO2 >= 0.2
         (cams_validate.py).
  Gases (all regions): not flat-lined.  An analyzer whose daily means barely move (median
         30-day coefficient of variation < 0.15) cannot register a weekly or holiday
         change and is excluded.  Computed here from the QC'd hourly table and cached in
         out/cams_flatline_check.csv.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
CV_MIN = 0.15


def flatline_table(force=False):
    path = os.path.join(OUT, "cams_flatline_check.csv")
    if os.path.exists(path) and not force:
        return pd.read_csv(path)
    c = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    c = c[c.time >= "2019-01-01"]
    rows = []
    for st, h in c.groupby("station"):
        h = h.assign(date=h.time.dt.normalize())
        for g in ("NOX", "CO", "PM25"):
            x = h[["date", g]].dropna()
            if len(x) < 2000:
                continue
            d = x.groupby("date")[g].mean()
            cv = (d.rolling(30).std() / d.rolling(30).mean()).median()
            rows.append(dict(station=st, tracer=g, hours=len(x), daily_cv30=round(float(cv), 3),
                             median=round(float(x[g].median()), 2)))
    t = pd.DataFrame(rows)
    t.to_csv(path, index=False)
    return t


def screens():
    val = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    ok_pm = set(val[(val.daily_r_embassy >= 0.6) & (val.best_lag_h.abs() <= 1)].station)
    gas_ok = {g: set(val[val[g + "_rho_tropomi"] >= 0.2].station) for g in ("NOX", "CO")}
    fl = flatline_table()
    flat = {(r.station, r.tracer) for r in fl.itertuples() if r.tracer != "PM25" and r.daily_cv30 < CV_MIN}
    for g in gas_ok:
        gas_ok[g] = {s for s in gas_ok[g] if (s, g) not in flat}
    return ok_pm, gas_ok, flat


def gas_allowed(station, tracer, region, gas_ok, flat):
    """True if this station's analyzer for `tracer` may be used."""
    if (station, tracer) in flat:
        return False
    if region == "Dhaka":
        return station in gas_ok.get(tracer, set())
    return True


if __name__ == "__main__":
    t = flatline_table(force=True)
    ok_pm, gas_ok, flat = screens()
    print("flat-lined analyzers excluded:", sorted(flat))
    print("Dhaka gas screen passed:", {k: sorted(v) for k, v in gas_ok.items()})
