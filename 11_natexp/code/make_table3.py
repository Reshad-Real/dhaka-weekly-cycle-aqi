"""Table 3: pooled effects by diurnal block and shock, every interval dependence-aware.

  Friday, Saturday  joint-date moving-block bootstrap (out/robust_bootstrap_days.csv): the same
                    calendar blocks are resampled for every station, so shared dates and
                    weather are kept; the AQI sub-index column comes from the same draws
  Eid               event bootstrap over the 11 usable events (out/events_dependence.csv;
                    the AQI from out/eid_checks.csv)
  Lockdown          model interval widened by the spread of the same design in placebo
                    windows of other years (out/events_dependence.csv, out/lockdown_checks.csv)
Writes out/table3.csv and out/table3.md.  Usage: python -I make_table3.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
UNITS = ["night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23", "daily"]
NAMES = {"PM25": "PM2.5", "NOX": "NOx", "CO": "CO"}


def cell(p, lo, hi):
    f = lambda v: ("%+.1f" % v).replace("-", "−")
    return "%s [%s, %s]" % (f(p), f(lo), f(hi))


def main():
    rows = []
    B = pd.read_csv(os.path.join(OUT, "robust_bootstrap_days.csv"))
    for day, shock in (("Fri", "Friday"), ("Sat", "Saturday")):
        a = B[(B.day == day) & (B.tracer == "AQI")]
        for (reg, tr), g in B[(B.day == day) & B.tracer.isin(list(NAMES))].groupby(["region", "tracer"]):
            g = g.set_index("block")
            aq = a[a.region == reg]
            aqi = cell(aq.b_point.iloc[0], aq.lo_b.iloc[0], aq.hi_b.iloc[0]) if tr == "PM25" and len(aq) else ""
            rows.append(dict(shock=shock, region=reg, tracer=NAMES[tr], interval="joint-date bootstrap",
                             **{u: cell(g.loc[u, "pct"], g.loc[u, "lo_pct"], g.loc[u, "hi_pct"]) if u in g.index
                                else "" for u in UNITS}, **{"daily AQI (points)": aqi}))
    D = pd.read_csv(os.path.join(OUT, "events_dependence.csv"))
    E = pd.read_csv(os.path.join(OUT, "eid_checks.csv"))
    L = pd.read_csv(os.path.join(OUT, "lockdown_checks.csv"))
    ea = E[(E.analysis == "event_boot") & (E.outcome == "AQI")].iloc[0]
    la = L[(L.analysis == "main_with_placebo_spread") & (L.outcome == "AQI")].iloc[0]
    for ev, interval, arow in (("Eid", "event bootstrap", ea), ("Lockdown", "model + placebo spread", la)):
        g = D[D.event == ev].set_index("unit")
        rows.append(dict(shock=ev, region="Dhaka", tracer="PM2.5", interval="%s (k=%d)" % (interval, g.k.max()),
                         **{u: cell(g.loc[u, "b_pct"], g.loc[u, "lo_pct"], g.loc[u, "hi_pct"]) if u in g.index
                            else "" for u in UNITS},
                         **{"daily AQI (points)": cell(arow.b, arow.lo, arow.hi)}))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "table3.csv"), index=False)
    with open(os.path.join(OUT, "table3.md"), "w", encoding="utf-8") as f:
        try:
            f.write(T.to_markdown(index=False))
        except ImportError:          # tabulate not installed
            f.write(T.to_string(index=False))
    # the console may not be UTF-8 (cp1252 when redirected on Windows)
    print(T.to_string(index=False).replace("−", "-"))


if __name__ == "__main__":
    main()
