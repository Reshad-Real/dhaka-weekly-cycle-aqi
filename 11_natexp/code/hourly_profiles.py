"""Hour-by-hour weekend profiles for PM2.5 and the surface traffic tracers, and the
NOx/CO ratio as a fleet-composition signature (OUTLINE.md next steps 2 and 5).

A calendar Friday's first six hours are Thursday night, the last working night; the rest
day's own night falls on Saturday's first hours.  Reading Thursday, Friday and Saturday as
one 72-hour sequence (each hour against the same clock hour on Sunday-Wednesday) shows when
the rest period begins and ends instead of cutting it at midnight.

For each station, tracer and clock hour: log(y) on the day indicator (Thu, Fri, Sat, and
the four working-day placebos) against the remaining Sunday-Wednesday days, year-month fixed
effects and the hour's block-level ERA5 meteorology; event days excluded; HAC errors.
Pooled by DerSimonian-Laird, Dhaka region and national separately.  Tracers:

  PM25         screened stations (as robustness.py)
  NOX, CO      stations whose analyzer passed the TROPOMI screen (Dhaka) / all (national)
  NOX_CO       log(NOx) - log(CO), same site and hour: rises when diesel heavy vehicles
               make up more of the fleet (high NOx per unit CO) and falls when petrol and
               CNG light vehicles do (high CO per unit NOx)

Writes out/hourly_profiles.csv (pooled) and out/hourly_profiles_station.csv.
Usage: python -I hourly_profiles.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_shocks import BLOCKS, calendars, embassy_hourly, met_daily, COVID  # noqa: E402
from cams_pooled import DHAKA, dl_pool  # noqa: E402
import robustness as rb  # noqa: E402
from screens import gas_allowed, screens  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
DAYS = {3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun", 0: "Mon", 1: "Tue", 2: "Wed"}
BLK_OF = {h: b for b, hs in BLOCKS.items() for h in hs}


def hourly_frames(root):
    """{(station, region, tracer, hour): design frame} for every screened station-tracer and
    clock hour: log outcome (or the log NOx/CO ratio), the hour's block meteorology, day of
    week and year-month, event days removed.  Shared with hourly_boot.py."""
    metd, metb = met_daily(root)
    hart, eid, eid_days = calendars(root)
    ok_pm, gas_ok, flat = screens()
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[(cams.time >= "2019-01-01") & (cams.time < "2025-01-01")]
    emb = embassy_hourly(root)
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2026-01-01")]
    allh = pd.concat([cams, emb], ignore_index=True)
    allh["NOX_CO"] = np.log(allh["NOX"].where(allh["NOX"] > 0)) - np.log(allh["CO"].where(allh["CO"] > 0))

    ev_days = set(hart) | set(eid_days)
    frames = {}
    for st, h in allh.groupby("station"):
        reg = "Dhaka" if st in DHAKA else "national"
        h = h.copy()
        h["date"] = h.time.dt.normalize()
        h["hour"] = h.time.dt.hour
        h = h.drop_duplicates(["date", "hour"])
        for y in ("PM25", "NOX", "CO", "NOX_CO"):
            if y == "PM25" and reg == "Dhaka" and st != "US Embassy" and st not in ok_pm:
                continue
            if y in ("NOX", "CO") and not gas_allowed(st, y, reg, gas_ok, flat):
                continue
            if y == "NOX_CO" and not (gas_allowed(st, "NOX", reg, gas_ok, flat)
                                      and gas_allowed(st, "CO", reg, gas_ok, flat)):
                continue
            if y not in h or h[y].notna().sum() < 24 * 300:
                continue
            for hh in range(24):
                x = h[(h.hour == hh)].set_index("date")[[y]].dropna()
                if y != "NOX_CO":
                    x = x[x[y] > 0]
                s = x.join(metb.xs(BLK_OF[hh], level="block"), how="inner")
                s["dow"] = s.index.dayofweek
                s["ym"] = s.index.strftime("%Y-%m")
                s["event"] = s.index.isin(list(ev_days)) | ((s.index >= COVID[0]) & (s.index <= COVID[1]))
                s = s[~s.event]
                s["ly"] = s[y] if y == "NOX_CO" else np.log(s[y])
                s["lvc2"] = s.lvc ** 2
                s["lpr"] = np.log1p(s.pr)
                s = s.dropna(subset=["ly"] + rb.METC).sort_index()
                if len(s) < 300:
                    continue
                frames[(st, reg, y, hh)] = s
    return frames


def main(root):
    rows = []
    for (st, reg, y, hh), s in hourly_frames(root).items():
        for dw, name in DAYS.items():
            r = rb.fit(s, dw)
            if r:
                rows.append(dict(station=st, region=reg, tracer=y, hour=hh, day=name,
                                 b=r[0], se=r[1], n=r[2]))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "hourly_profiles_station.csv"), index=False)
    P = []
    for (reg, y, hh, day), g in S.groupby(["region", "tracer", "hour", "day"]):
        b, se, tau2, i2, k = dl_pool(g.b, g.se)
        f = (lambda v: 100 * (np.exp(v) - 1))
        P.append(dict(region=reg, tracer=y, hour=hh, day=day, b=b, se=se, pct=f(b),
                      lo=f(b - 1.96 * se), hi=f(b + 1.96 * se), z=b / se, k=k))
    P = pd.DataFrame(P)
    band = P[P.day.isin(["Sun", "Mon", "Tue", "Wed"])].groupby(["region", "tracer", "hour"]).pct.agg(
        placebo_min="min", placebo_max="max").reset_index()
    P = P.merge(band, on=["region", "tracer", "hour"])
    P.to_csv(os.path.join(OUT, "hourly_profiles.csv"), index=False)
    pd.set_option("display.width", 260)
    for reg in ("Dhaka", "national"):
        for y in ("PM25", "NOX", "CO", "NOX_CO"):
            q = P[(P.region == reg) & (P.tracer == y) & P.day.isin(["Thu", "Fri", "Sat"])]
            if q.empty:
                continue
            t = q.pivot(index="day", columns="hour", values="pct").reindex(["Thu", "Fri", "Sat"])
            print("\n%s %s (k=%d) percent vs Sun-Wed, by clock hour" % (reg, y, q.k.max()))
            print(t.round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
