"""Dependence-aware intervals for the activity tracers of the whole-city shocks.

The pooled Eid and lockdown rows of out/shock_activity.csv (tropomi_events.py) carry
Newey-West / DerSimonian-Laird intervals, which treat stations as independent and, for Eid,
the event days as many draws.  Here the same estimates get the intervals used for PM2.5 in
events_checks.py:
  Eid       an event bootstrap: the usable Eid events (outside the lockdown) are resampled
            with replacement (B = 1000) and every station is refitted on the resampled
            events, pooled with the station weights of the main fit
  Lockdown  (TROPOMI only; the gas analyzers are not comparable across years) the
            standard error widened by the spread of pseudo-lockdowns on 26 March - 30 May
            of 2019, 2022, 2023 and 2024
Outcomes: TROPOMI NO2 (days with >= 50% of the domain observed), daily NOx and CO at the
gas-valid stations (Dhaka region and national) and daily PM2.5 at the national stations,
selected exactly as in tropomi_events.py.

Writes out/events_tracers.csv.
Usage: python -I events_tracers.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_pooled import DHAKA  # noqa: E402
from cams_shocks import COVID, TROPOMI_MIN_COVERAGE, calendars, met_daily, tropomi_daily  # noqa: E402
from events_checks import PLACEBO_YEARS, eid_fit, lock_fit, pooled  # noqa: E402
from screens import gas_allowed, screens  # noqa: E402
from tropomi_events import frame  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
RNG = np.random.default_rng(20261023)
B_EVENTS = 1000
LY = lambda s: np.log(s.y.values)  # noqa: E731


def pct(b):
    return 100 * (np.exp(b) - 1)


def eid_boot(frames, eid):
    usable = [t for t in eid if not (COVID[0] <= t <= COVID[1])]
    fits = {x: eid_fit(s, eid, LY) for x, s in frames.items()}
    fits = {x: f for x, f in fits.items() if f}
    if not fits:
        return None
    pb, pse, k, w, _ = pooled(fits)
    draws = []
    for _ in range(B_EVENTS):
        ev = [pd.Timestamp(e) for e in RNG.choice(np.array(usable, dtype="datetime64[ns]"),
                                                   len(usable), replace=True)]
        bs = {x: eid_fit(frames[x], ev, LY, point=True) for x in fits}
        bs = {x: v for x, v in bs.items() if v is not None}
        if bs:
            draws.append(sum(w[x] * v for x, v in bs.items()) / sum(w[x] for x in bs))
    draws = np.array(draws)
    # events with at least one observed treatment day in some station's record
    seen = sum(any(((s.index >= t - pd.Timedelta(days=1)) & (s.index <= t + pd.Timedelta(days=3))).any()
                   for s in frames.values()) for t in usable)
    return dict(shock="Eid", k=k, n_events=seen, b=pb, se_main=pse, lo=np.percentile(draws, 2.5),
                hi=np.percentile(draws, 97.5), se=draws.std(), n_draws=len(draws), interval="event bootstrap")


def lock_placebo(s):
    f = lock_fit(s, LY)
    if not f:
        return None
    plac = {}
    for yr in PLACEBO_YEARS:
        p = lock_fit(s, LY, year=yr, placebo=True)
        if p:
            plac[yr] = p[0]
    sd = float(np.std(list(plac.values()), ddof=1))
    tot = np.sqrt(f[1] ** 2 + sd ** 2)
    row = dict(shock="Lockdown", k=1, n_events=1, b=f[0], se_main=f[1], lo=f[0] - 1.96 * tot,
               hi=f[0] + 1.96 * tot, se=tot, placebo_sd=sd, interval="placebo-calibrated")
    row.update({"placebo_%d" % y: pct(b) for y, b in plac.items()})
    return row


def main(root):
    metd, _ = met_daily(root)
    hart, eid, eid_days = calendars(root)
    rows = []
    tr = tropomi_daily(root, TROPOMI_MIN_COVERAGE).loc["2019-01-01":"2025-12-31"]
    s = frame(tr, metd, hart, eid_days)
    r = eid_boot({"TROPOMI": s}, eid)
    rows.append(dict(outcome="TROPOMI NO2", region="domain", **r))
    rows.append(dict(outcome="TROPOMI NO2", region="domain", **lock_placebo(s)))

    ok_pm, gas_ok, flat = screens()
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[(cams.time >= "2019-01-01") & (cams.time < "2026-01-01")]
    cams["date"] = cams.time.dt.normalize()
    groups = {}
    for st, h in cams.groupby("station"):
        reg = "Dhaka" if st in DHAKA else "national"
        for tracer in ("PM25", "NOX", "CO"):
            if tracer == "PM25" and reg == "Dhaka":
                continue        # Dhaka PM2.5 is in events_checks.py
            if tracer in ("NOX", "CO") and not gas_allowed(st, tracer, reg, gas_ok, flat):
                continue
            g = h.groupby("date")[tracer]
            d = g.mean().where(g.count() >= 18).dropna()
            if len(d) < 300:
                continue
            groups.setdefault((reg, tracer), {})[st] = frame(d, metd, hart, eid_days)
    for (reg, tracer), fr in sorted(groups.items()):
        r = eid_boot(fr, eid)
        if r:
            rows.append(dict(outcome=tracer, region=reg, stations=";".join(sorted(fr)), **r))
    T = pd.DataFrame(rows)
    for c in ("b", "lo", "hi"):
        T[c + "_pct"] = pct(T[c])
    T.to_csv(os.path.join(OUT, "events_tracers.csv"), index=False)
    pd.set_option("display.width", 220)
    print(T.drop(columns=[c for c in ("stations",) if c in T]).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
