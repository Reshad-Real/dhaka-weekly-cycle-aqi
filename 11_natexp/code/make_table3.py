"""Table 3: pooled effects by diurnal block and shock, with intervals, and the daily AQI change.

Friday rows use the joint-date block bootstrap (out/robust_bootstrap.csv); Saturday rows the
main random-effects pool (out/robust_pooled.csv); Eid and lockdown rows the pooled event
estimates (out/events_pooled.csv); AQI points from out/aqi_by_shock.csv.
Writes out/table3.csv and out/table3.md.  Usage: python -I make_table3.py
"""
import os

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
    B = pd.read_csv(os.path.join(OUT, "robust_bootstrap.csv"))
    for (reg, tr), g in B.groupby(["region", "tracer"]):
        g = g.set_index("block")
        rows.append(dict(shock="Friday", region=reg, tracer=NAMES[tr], interval="joint-date bootstrap",
                         **{u: cell(g.loc[u, "pct"], g.loc[u, "lo_pct"], g.loc[u, "hi_pct"]) for u in UNITS}))
    R = pd.read_csv(os.path.join(OUT, "robust_pooled.csv"))
    R = R[(R.variant == "main") & (R.day == "Sat")]
    for (reg, tr), g in R.groupby(["region", "tracer"]):
        g = g.set_index("block")
        rows.append(dict(shock="Saturday", region=reg, tracer=NAMES[tr], interval="random effects",
                         **{u: cell(g.loc[u, "pct"], g.loc[u, "lo"], g.loc[u, "hi"]) for u in UNITS}))
    E = pd.read_csv(os.path.join(OUT, "events_pooled.csv"))
    E = E[~E.unit.astype(str).str.isdigit()]
    for ev, g in E.groupby("event"):
        g = g.set_index("unit")
        rows.append(dict(shock=ev, region="Dhaka", tracer="PM2.5", interval="random effects (k=%d)" % g.k.max(),
                         **{u: cell(g.loc[u, "pct"], g.loc[u, "lo"], g.loc[u, "hi"]) for u in UNITS}))
    T = pd.DataFrame(rows)
    A = pd.read_csv(os.path.join(OUT, "aqi_by_shock.csv")).set_index("event")
    T["daily AQI (points)"] = [
        cell(A.loc[s, "pooled_aqi_pts"], A.loc[s, "lo"], A.loc[s, "hi"])
        if (s in A.index and r == "Dhaka" and t == "PM2.5") else "" for s, r, t in zip(T.shock, T.region, T.tracer)]
    order = {"Friday": 0, "Saturday": 1, "Eid": 2, "Lockdown": 3}
    T = T.sort_values(["shock", "region", "tracer"], key=lambda c: c.map(order) if c.name == "shock" else c)
    T.to_csv(os.path.join(OUT, "table3.csv"), index=False)
    cols = ["shock", "region", "tracer"] + UNITS + ["daily AQI (points)"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in T.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    note = ("\nPercent change against the same period on Sunday-Wednesday (Friday, Saturday), the "
            "surrounding days (Eid) or the year, month and weather expectation (lockdown); 95% "
            "intervals. Friday intervals from a joint-date moving-block bootstrap that keeps the "
            "dependence between stations; the others from random-effects pooling, which treats "
            "stations as independent and is therefore optimistic. Night = 00-05 h of the calendar "
            "day, so Friday's night block is Thursday night. The number of stations can differ "
            "between the daily and block columns where a station lacks enough complete days "
            "(lockdown: k = 3 daily, 4 per block), so the daily value need not lie inside the "
            "block range.\n")
    open(os.path.join(OUT, "table3.md"), "w", encoding="utf-8").write("\n".join(md) + "\n" + note)
    print("wrote table3.csv and table3.md (%d rows)" % len(T))


if __name__ == "__main__":
    main()
