"""Re-derive every amber number (\\pn{...}) in paper/manuscript.tex from 11_natexp/out and
report any mismatch, and any amber number no claim accounts for.

How it works
------------
Every preliminary number in the manuscript is wrapped in \\pn{...}.  A claim names an
anchor (a short piece of manuscript text that occurs exactly once) and the values of the
next N amber numbers after it, each computed from the result files.  The checker

  * finds each anchor (it must occur exactly once),
  * takes the next N \\pn{} occurrences after it, in order,
  * parses each printed value (signs, $-$, thousands separators, "a--b" ranges) and
    compares it with the derived value rounded half-up to the printed number of decimals
    (or within an explicit tolerance for "about" numbers),
  * reports every \\pn{} that no claim consumed, and every \\pn{} consumed twice.

Exit status is non-zero on any mismatch, unmatched anchor or uncovered number, so the
script can gate a build.

Usage:  python -I verify_claims.py              check
        python -I verify_claims.py --inventory  list every amber number with context
"""
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
TEX = os.environ.get("VERIFY_TEX", os.path.join(HERE, "..", "paper", "manuscript.tex"))
ROOT = os.path.join(HERE, "..", "..")

PN = re.compile(r"\\pn\{((?:[^{}]|\{[^{}]*\})*)\}")


# ------------------------------------------------------------------ parsing the text
def amber(text):
    out = []
    for m in PN.finditer(text):
        line_start = text.rfind(chr(10), 0, m.start()) + 1
        if text[line_start:m.start()].lstrip().startswith("%"):
            continue                       # inside a LaTeX comment
        if not re.search(r"[0-9]", m.group(1)):
            continue                       # e.g. the banner's colour sample
        out.append((m.start(), m.group(1)))
    return out


def parse_printed(raw):
    """Printed amber content -> (kind, values, decimals).  kind: 'num' or 'range'."""
    t = raw.replace("$", "").replace("{", "").replace("}", "").replace(",", "").strip()
    t = t.replace("\u2212", "-")
    if "--" in t:
        a, b = t.split("--")
        return "range", [float(a), float(b)], [dec(a), dec(b)]
    return "num", [float(t)], [dec(t)]


def dec(s):
    s = s.strip().lstrip("+-")
    return len(s.split(".")[1]) if "." in s else 0


def round_half_up(x, d):
    q = Decimal(1).scaleb(-d)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


# ------------------------------------------------------------------ result files
def csv(name):
    return pd.read_csv(os.path.join(OUT, name))


class R:
    """Lazy accessors for the result files used by the claims."""

    def __init__(self):
        self.mob = csv("mobility_shocks.csv").set_index("shock")
        self.act = csv("shock_activity.csv")
        self.boot = csv("robust_bootstrap.csv")
        self.pool = csv("robust_pooled.csv")
        self.hp = csv("hourly_profiles.csv")
        self.blh = csv("blh_by_hour.csv")
        self.share = csv("pm_share_by_hour.csv")
        self.wcheck = csv("weighting_check.csv")
        self.vent = csv("friday_by_block_vc.csv")
        self.ev = csv("events_pooled.csv")
        self.evst = csv("events_station_units.csv")
        self.aqi = csv("aqi_by_shock.csv").set_index("event")
        self.scen = csv("aqi_scenarios.csv").set_index("scenario")
        self.scen_st = csv("aqi_scenarios_stations.csv")
        self.qc = csv("cams_qc_report.csv").set_index("rule")
        self.val = csv("cams_validation.csv").set_index("station")
        self.erel = csv("eid_relative_day.csv")
        self.eev = csv("eid_by_event.csv")
        with open(os.path.join(ROOT, "06_results", "metrics", "lockdown_test.json")) as f:
            self.old_lock = json.load(f)

    # -- helpers
    def mobility(self, shock_prefix, col):
        rows = self.mob[self.mob.index.str.startswith(shock_prefix)]
        return rows[col]

    def activity(self, shock, outcome, region, field="pct"):
        q = self.act[(self.act.shock == shock) & (self.act.outcome == outcome) & (self.act.region == region)]
        assert len(q) == 1, (shock, outcome, region, len(q))
        return float(q[field].iloc[0])

    def bootstrap(self, region, tracer, block, field="pct"):
        q = self.boot[(self.boot.region == region) & (self.boot.tracer == tracer) & (self.boot.block == block)]
        assert len(q) == 1, (region, tracer, block)
        return float(q[field].iloc[0])

    def pooled(self, variant, region, tracer, day, block, field="pct"):
        q = self.pool[(self.pool.variant == variant) & (self.pool.region == region) & (self.pool.tracer == tracer)
                      & (self.pool.day == day) & (self.pool.block == block)]
        assert len(q) == 1, (variant, region, tracer, day, block, len(q))
        return float(q[field].iloc[0])

    def hourly(self, region, tracer, day, hours, field="pct"):
        q = self.hp[(self.hp.region == region) & (self.hp.tracer == tracer) & (self.hp.day == day)
                    & self.hp.hour.isin(list(hours))]
        return q.set_index("hour")[field].reindex(list(hours))

    def placebo_range(self, region, block):
        q = self.pool[(self.pool.variant == "main") & (self.pool.region == region) & (self.pool.tracer == "PM25")
                      & (self.pool.block == block) & self.pool.day.isin(["Sun", "Mon", "Tue", "Wed"])]
        return float(q.pct.min()), float(q.pct.max())

    def event(self, event, unit, field="pct"):
        q = self.ev[(self.ev.event == event) & (self.ev.unit.astype(str) == str(unit))]
        assert len(q) == 1, (event, unit)
        return float(q[field].iloc[0])

    def event_hours(self, event):
        q = self.ev[(self.ev.event == event) & self.ev.unit.astype(str).str.isdigit()]
        return q.pct

    def event_station(self, event, station):
        q = self.evst[(self.evst.event == event) & (self.evst.station == station) & (self.evst.unit == "daily")]
        assert len(q) == 1, (event, station)
        return float(100 * (np.exp(q.b.iloc[0]) - 1))

    def blh_exact(self, season, hours):
        """Median ERA5 boundary-layer height over all hours of a block and season, 2019-2025,
        recomputed from the raw file exactly as blh_weighting.py prints it."""
        if not hasattr(self, "_met"):
            m = pd.read_csv(os.path.join(ROOT, "02_data", "raw", "meteo_dhaka_hourly_filled.csv"),
                            parse_dates=["time"])
            m = m[(m.time >= "2019-01-01") & (m.time < "2026-01-01")]
            seas = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
                    5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
                    10: "Post-monsoon", 11: "Post-monsoon"}
            m["season"] = m.time.dt.month.map(seas)
            m["hour"] = m.time.dt.hour
            self._met = m
        m = self._met
        return float(m[(m.season == season) & m.hour.isin(list(hours))].boundary_layer_height.median())

    def block_share(self, season, block_hours):
        q = self.share[(self.share.season == season) & self.share.hour.isin(list(block_hours))]
        return 100 * float(q.share.sum())

    def weighting(self, variant, region, field):
        q = self.wcheck[(self.wcheck.variant == variant) & (self.wcheck.region == region)]
        return float(q[field].iloc[0])

    def ventilation(self, region, block, tercile, field="pct"):
        q = self.vent[(self.vent.region == region) & (self.vent.block == block) & (self.vent.vc_tercile == tercile)]
        return float(q[field].iloc[0])

    def eid_rel(self, station, days):
        q = self.erel[(self.erel.station == station) & (self.erel.tracer == "PM25") & self.erel.rel.isin(list(days))]
        return q.pct


# ------------------------------------------------------------------ expected values
# Each value is one of:
#   number                 compared at the printed precision (half-up)
#   ("abs", number)        compared as a magnitude (the text says "x percent lower")
#   ("about", number, tol) compared within an absolute tolerance (no longer used: every
#                          number in the manuscript is printed exactly)
#   ("range", lo, hi)      a printed "a--b" range
#   ("rangeabout", lo, hi, tol)
def claims(r):
    A = lambda v: ("abs", v)
    BL = ["night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23"]
    PM = "PM25"
    C = []

    def add(name, anchor, values):
        C.append((name, anchor, values))

    # ---------------------------------------------------------------- Data section
    rep = csv("cams_parse_report.csv")
    add("rows without a date", "dropped rather than reconstructed:",
        [int(rep[rep.file.str.contains("2012") & rep.station.isna()].dropped_no_date.iloc[0])])
    add("QC stuck and PM2.5>PM10 counts", "Over the full\nrecord these rules removed",
        [r.qc.loc["Q2 stuck runs PM25", "n"], r.qc.loc["Q3 PM2.5 > 1.1 PM10", "n"]])
    dh = ["BARC", "Darussalam", "DoE", "Savar", "Gazipur", "Narayanganj", "Narsingdi"]
    rr = r.val.loc[dh, "daily_r_embassy"]
    add("PM2.5 r range with Embassy", "correlates with the Embassy record at $r=$",
        [("range", rr.min(), rr.max())])
    cv = pd.read_csv(os.path.join(ROOT, "06_results", "metrics", "blh_gapfill_cv.csv"))
    met = pd.read_csv(os.path.join(ROOT, "02_data", "raw", "meteo_dhaka_hourly_filled.csv"),
                      usecols=["blh_filled"])
    add("BLH gap fill", "product of mixing height and wind speed. A",
        [int(met.blh_filled.sum()), len(cv), cv.r2.mean(), cv.rmse.mean(), cv.bias.mean()])

    sys.path.insert(0, HERE)            # -I leaves the script's folder off the path
    import aqi_scenarios
    emb = aqi_scenarios.embassy_hourly(ROOT)
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2026-01-01")].drop_duplicates("time")
    g = emb.assign(date=emb.time.dt.normalize()).groupby("date").PM25
    dd = g.mean()[g.count() >= 18]
    win = dd[dd.index.month.isin([12, 1, 2])]
    add("Embassy level (Introduction)", "At the United States Embassy monitor in Dhaka, daily",
        [dd.mean(), dd.mean() / 5.0, 100 * (win > 150).mean()])
    fridays = int((pd.date_range("2019-01-01", "2024-12-31").dayofweek == 4).sum())
    add("Fridays 2019-2024 (Introduction)", "a partial rest day) provides", [fridays])

    # ---------------------------------------------------------------- 4.1 activity
    lock_t = r.mobility("lockdown", "transit").iloc[0]
    lock_w = r.mobility("lockdown", "workplaces").iloc[0]
    lock_r = r.mobility("lockdown", "residential").iloc[0]
    eid_w = r.mobility("Eid", "workplaces")
    eid_t = r.mobility("Eid", "transit")
    st_t = r.mobility("strike", "transit").iloc[0]
    st_w = r.mobility("strike", "workplaces").iloc[0]
    st_n = int(r.mobility("strike", "n_days").iloc[0])
    add("mobility lockdown, Eid, strikes (text)", "The lockdown\nlowered national mobility",
        [A(lock_t), A(lock_w), lock_r, A(eid_w.mean()), A(eid_w.max()), A(eid_w.min()), eid_t.mean(), st_n, st_t, st_w])
    add("strike NO2 and Dhaka NOx; Eid and lockdown NO2 (text)", "unchanged; over all",
        [113, r.activity("Strike days", "TROPOMI NO2", "domain"),
         r.activity("Strike days", "TROPOMI NO2", "domain", "lo"), r.activity("Strike days", "TROPOMI NO2", "domain", "hi"),
         r.activity("Strike days", "NOX", "Dhaka"), r.activity("Strike days", "NOX", "Dhaka", "lo"),
         r.activity("Strike days", "NOX", "Dhaka", "hi"),
         A(r.activity("Eid", "TROPOMI NO2", "domain")), A(r.activity("Lockdown", "TROPOMI NO2", "domain"))])

    def tri(shock, outcome, region):
        return [r.activity(shock, outcome, region, k) for k in ("pct", "lo", "hi")]
    add("Table 2 Friday", "Friday & weekly rest day",
        [fridays] + tri("Friday", "TROPOMI NO2", "domain") + tri("Friday", "NOX", "national") + tri("Friday", "CO", "national"))
    add("Table 2 Saturday", "Saturday & partial rest day",
        tri("Saturday", "TROPOMI NO2", "domain") + tri("Saturday", "NOX", "national") + tri("Saturday", "CO", "national"))
    add("Table 2 Eid", "Eid & $-1$ to $+3$ days",
        [eid_w.mean(), eid_t.mean()] + tri("Eid", "TROPOMI NO2", "domain") + tri("Eid", "NOX", "Dhaka")
        + [r.activity("Eid", "CO", "Dhaka")])
    add("Table 2 lockdown", "Lockdown & 26 Mar -- 30 May 2020",
        [lock_t, lock_w, lock_r] + tri("Lockdown", "TROPOMI NO2", "domain"))
    add("Table 2 strikes", "Strike days & ACLED-derived",
        [113, st_n, st_t, st_w] + tri("Strike days", "TROPOMI NO2", "domain") + tri("Strike days", "NOX", "national")
        + tri("Strike days", "CO", "national"))

    nox_mid = r.hourly("national", "NOX", "Fri", range(13, 18))
    co_mid = r.hourly("national", "CO", "Fri", range(13, 18))
    co_eve = r.hourly("national", "CO", "Fri", range(18, 24))
    add("weekly tracers (text)", "result. The \\notwo{} column is",
        [A(r.activity("Friday", "TROPOMI NO2", "domain")), A(r.activity("Saturday", "TROPOMI NO2", "domain")),
         A(nox_mid.max()), A(nox_mid.min()), A(co_mid.max()), A(co_mid.min()), A(co_eve.max()), A(co_eve.min()),
         A(r.bootstrap("national", "NOX", "afternoon 12-17")), r.bootstrap("national", "NOX", "afternoon 12-17", "lo_pct"),
         r.bootstrap("national", "NOX", "afternoon 12-17", "hi_pct"), A(r.bootstrap("national", "NOX", "evening 18-23")),
         A(r.bootstrap("national", "CO", "afternoon 12-17")), A(r.bootstrap("national", "CO", "evening 18-23")),
         r.bootstrap("national", "NOX", "night 00-05"), r.bootstrap("national", "NOX", "night 00-05", "lo_pct"),
         r.bootstrap("national", "NOX", "night 00-05", "hi_pct"),
         r.bootstrap("Dhaka", "NOX", "night 00-05"), r.bootstrap("Dhaka", "NOX", "night 00-05", "lo_pct"),
         r.bootstrap("Dhaka", "NOX", "night 00-05", "hi_pct"),
         A(r.pooled("main", "national", "NOX", "Sat", "night 00-05")),
         r.pooled("main", "national", "NOX", "Sat", "night 00-05", "lo"),
         r.pooled("main", "national", "NOX", "Sat", "night 00-05", "hi"),
         -r.hourly("Dhaka", "NOX", "Sat", range(0, 6)).max(),
         -r.hourly("Dhaka", "NOX", "Sat", range(0, 7)).min()])
    nc_nat = r.hourly("national", "NOX_CO", "Fri", range(0, 4))
    nc_dh = r.hourly("Dhaka", "NOX_CO", "Fri", range(2, 6))
    dh_nc = r.hp[(r.hp.region == "Dhaka") & (r.hp.tracer == "NOX_CO")]
    add("NOx/CO signature", "more of the fleet, is",
        [nc_nat.min(), nc_nat.max(), float((dh_nc.hi - dh_nc.lo).median())])

    # ---------------------------------------------------------------- 4.2 PM2.5
    fri_am = r.hourly("Dhaka", PM, "Fri", range(0, 12))
    fri_pm = r.hourly("Dhaka", PM, "Fri", range(13, 24))
    paft = r.placebo_range("Dhaka", "afternoon 12-17")
    peve = r.placebo_range("Dhaka", "evening 18-23")
    add("Dhaka hourly and blocks", "midnight to late morning (",
        [fri_am.min(), fri_am.max(), A(fri_pm.max()), A(fri_pm.min()),
         A(r.bootstrap("Dhaka", PM, "afternoon 12-17")), r.bootstrap("Dhaka", PM, "afternoon 12-17", "lo_pct"),
         r.bootstrap("Dhaka", PM, "afternoon 12-17", "hi_pct"),
         A(r.bootstrap("Dhaka", PM, "evening 18-23")), r.bootstrap("Dhaka", PM, "evening 18-23", "lo_pct"),
         r.bootstrap("Dhaka", PM, "evening 18-23", "hi_pct"),
         paft[0], paft[1], peve[0], peve[1],
         A(r.bootstrap("Dhaka", PM, "night 00-05")), r.bootstrap("Dhaka", PM, "night 00-05", "lo_pct"),
         r.bootstrap("Dhaka", PM, "night 00-05", "hi_pct"), A(r.bootstrap("Dhaka", PM, "morning 06-11")),
         A(r.bootstrap("Dhaka", PM, "daily")), r.bootstrap("Dhaka", PM, "daily", "lo_pct"),
         r.bootstrap("Dhaka", PM, "daily", "hi_pct")])
    nat_h = r.hourly("national", PM, "Fri", range(12, 24))
    sat_dh = r.hourly("Dhaka", PM, "Sat", range(0, 8))
    sat_nat = r.hourly("national", PM, "Sat", range(0, 6))
    add("national blocks, rest night, Saturday", "The nine stations outside the Dhaka region reproduce",
        [A(r.bootstrap("national", PM, "afternoon 12-17")), r.bootstrap("national", PM, "afternoon 12-17", "lo_pct"),
         r.bootstrap("national", PM, "afternoon 12-17", "hi_pct"),
         A(r.bootstrap("national", PM, "evening 18-23")), r.bootstrap("national", PM, "evening 18-23", "lo_pct"),
         r.bootstrap("national", PM, "evening 18-23", "hi_pct"), nat_h.min(),
         r.bootstrap("national", PM, "night 00-05"), r.bootstrap("national", PM, "daily"),
         r.bootstrap("national", PM, "daily", "lo_pct"), r.bootstrap("national", PM, "daily", "hi_pct"),
         -sat_dh.mean(), A(sat_nat.min()),
         A(r.pooled("main", "Dhaka", PM, "Sat", "afternoon 12-17")), A(r.pooled("main", "Dhaka", PM, "Sat", "evening 18-23")),
         A(r.pooled("main", "Dhaka", PM, "Sat", "daily")), r.pooled("main", "Dhaka", PM, "Sat", "daily", "lo"),
         r.pooled("main", "Dhaka", PM, "Sat", "daily", "hi"),
         A(r.pooled("main", "Dhaka", PM, "Sat", "night 00-05")), r.pooled("main", "Dhaka", PM, "Sat", "night 00-05", "lo"),
         r.pooled("main", "Dhaka", PM, "Sat", "night 00-05", "hi"),
         A(r.bootstrap("national", PM, "afternoon 12-17")), A(r.bootstrap("national", "NOX", "afternoon 12-17")),
         A(r.bootstrap("national", "CO", "afternoon 12-17")),
         A(r.bootstrap("national", PM, "evening 18-23")), A(r.bootstrap("national", "NOX", "evening 18-23")),
         A(r.bootstrap("national", "CO", "evening 18-23"))])

    loo = r.pool[(r.pool.region == "Dhaka") & (r.pool.tracer == PM) & (r.pool.day == "Fri")
                 & r.pool.variant.str.startswith("loo_")]
    la = loo[loo.block == "afternoon 12-17"].z
    le = loo[loo.block == "evening 18-23"].z
    pz = lambda v, b: r.pooled(v, "Dhaka", PM, "Fri", b, "z")
    pp = lambda v, b: r.pooled(v, "Dhaka", PM, "Fri", b)
    q = r.pool[(r.pool.variant == "main") & (r.pool.day == "Fri") & (r.pool.tracer == PM)]
    qa = float(q[(q.region == "Dhaka") & (q.block == "afternoon 12-17")].q_BH.iloc[0])
    qe = float(q[(q.region == "Dhaka") & (q.block == "evening 18-23")].q_BH.iloc[0])
    qn = float(q[(q.region == "national") & (q.block == "evening 18-23")].q_BH.iloc[0])
    infl = float(r.boot[(r.boot.region == "Dhaka") & (r.boot.tracer == PM) & (r.boot.block == "afternoon 12-17")].se_inflation.iloc[0])
    add("robustness", "With week-clustered errors the Dhaka",
        [pz("cluster", "afternoon 12-17"), pz("cluster", "evening 18-23"), la.max(), la.min(), le.max(), le.min(),
         A(pp("nomet", "afternoon 12-17")), A(pp("nomet", "evening 18-23")),
         A(pp("y2022_25", "afternoon 12-17")), A(pp("y2022_25", "evening 18-23")),
         pz("y2022_25", "afternoon 12-17"), pz("y2022_25", "evening 18-23"),
         A(pp("y2019_21", "afternoon 12-17")), A(pp("y2019_21", "evening 18-23")),
         pz("y2019_21", "afternoon 12-17"), pz("y2019_21", "evening 18-23"),
         A(pp("screen_none", "afternoon 12-17")), A(pp("screen_none", "evening 18-23")),
         int(r.pooled("screen_strict", "Dhaka", PM, "Fri", "afternoon 12-17", "k")),
         A(pp("screen_strict", "afternoon 12-17")), A(pp("screen_strict", "evening 18-23")),
         A(r.pooled("screen_none", "national", "NOX", "Fri", "afternoon 12-17")),
         A(r.bootstrap("national", "NOX", "afternoon 12-17")),
         qa, qe, qn, infl])

    # ---------------------------------------------------------------- 4.3 weighting
    add("weighting opening", "The boundary layer explains why afternoon and evening responses of",
        [A(r.bootstrap("Dhaka", PM, "afternoon 12-17")), A(r.bootstrap("Dhaka", PM, "evening 18-23")),
         A(r.bootstrap("Dhaka", PM, "daily"))])
    blh = r.blh_exact
    add("BLH by block, winter", "In winter the median ERA5 mixing height is",
        [blh("Winter", range(0, 6)), blh("Winter", range(6, 12)), blh("Winter", range(12, 18)),
         blh("Winter", range(18, 24)), blh("Monsoon", range(0, 6))])
    sh = lambda s, h: r.block_share(s, h)
    dry = ["Winter", "Pre-monsoon", "Post-monsoon"]
    aft = [sh(s, range(12, 18)) for s in dry]
    nm = [sh(s, range(0, 12)) for s in dry + ["Monsoon"]]
    eve = [sh(s, range(18, 24)) for s in dry]
    add("block shares", "the afternoon block carries only",
        [min(aft), max(aft), sh("Monsoon", range(12, 18)), min(nm), max(nm), min(eve), max(eve)])
    w = lambda v, rg, f: r.weighting(v, rg, f)
    add("weighting identity", "Weighting the pooled block effects by these shares",
        [w("main", "Dhaka", "predicted_daily_pct"), w("main", "Dhaka", "observed_daily_pct"),
         w("main", "national", "predicted_daily_pct"), w("main", "national", "observed_daily_pct"),
         w("wet", "Dhaka", "predicted_daily_pct"), w("wet", "Dhaka", "observed_daily_pct"),
         w("dry", "Dhaka", "predicted_daily_pct"), w("dry", "Dhaka", "observed_daily_pct")])
    v = lambda rg, b, t, f="pct": r.ventilation(rg, b, t, f)
    add("ventilation terciles", "Within the afternoon\nblock",
        [v("national", "afternoon 12-17", "low"), v("national", "afternoon 12-17", "low", "lo"),
         v("national", "afternoon 12-17", "low", "hi"), v("national", "afternoon 12-17", "high"),
         v("Dhaka", "afternoon 12-17", "low"), v("Dhaka", "afternoon 12-17", "low", "lo"),
         v("Dhaka", "afternoon 12-17", "low", "hi"), v("Dhaka", "afternoon 12-17", "high"),
         v("Dhaka", "afternoon 12-17", "mid")])

    # ---------------------------------------------------------------- 4.4 whole-city
    eh = r.event_hours("Eid")
    es = lambda st: r.event_station("Eid", st)
    CITY = ("US Embassy", "BARC", "Darussalam")
    city_pre = pd.concat([r.eid_rel(s, range(-4, -1)) for s in CITY])
    city = pd.concat([r.eid_rel(s, range(0, 5)) for s in CITY])
    city_post = pd.concat([r.eid_rel(s, range(5, 9)) for s in CITY])
    corr = pd.concat([r.eid_rel(s, range(-5, 0)) for s in ("Narsingdi", "Savar")])
    eh_night = r.ev[(r.ev.event == "Eid") & r.ev.unit.astype(str).isin(["0", "1", "2"])].pct
    add("Eid (text)", "Over the Eid windows, pooled",
        [-eh.max(), -eh.min(), -eh_night.max(), -eh_night.min(),
         A(r.event("Eid", "daily")), r.event("Eid", "daily", "lo"),
         r.event("Eid", "daily", "hi"), r.event("Eid", "daily", "I2"),
         A(es("US Embassy")), A(es("BARC")), A(es("Darussalam")), A(es("Gazipur")), es("Narsingdi"),
         city_pre.abs().max(), -city.max(), -city.min(), -city_post.max(), -city_post.min(),
         int((corr > 0).sum()), len(corr), corr.max(),
         A(r.activity("Eid", "TROPOMI NO2", "domain")), A(r.activity("Eid", "NOX", "Dhaka")),
         A(r.activity("Eid", "PM25", "national"))])
    lh = r.event_hours("Lockdown")
    ls = lambda st: r.event_station("Lockdown", st)
    add("lockdown (text)", "The lockdown lowered pooled",
        [-lh.max(), -lh.min(), A(r.event("Lockdown", "night 00-05")),
         r.event("Lockdown", "night 00-05", "lo"), r.event("Lockdown", "night 00-05", "hi"),
         A(r.event("Lockdown", "daily")), r.event("Lockdown", "daily", "lo"), r.event("Lockdown", "daily", "hi"),
         ls("US Embassy"), ls("BARC"), ls("Darussalam"), ls("Narayanganj"),
         A(r.old_lock["observed_drop_matched_pct"]), r.old_lock["n_lockdown_days"],
         r.old_lock["n_matched_control_days"]])

    # ---------------------------------------------------------------- 4.5 AQI
    a = lambda e, f="pooled_aqi_pts": float(r.aqi.loc[e, f])
    s = r.scen
    sc = lambda k, f: float(s.loc[k, f])
    rest_rng = r.scen_st[r.scen_st.scenario == "rest_day"].d_mean_aqi
    night_gain = 100 * (sc("cut_night_only", "d_mean_aqi") / sc("cut_day_only", "d_mean_aqi") - 1)
    add("AQI by shock", "In index points, a Friday lowers",
        [A(a("Friday")), a("Friday", "lo"), a("Friday", "hi"), A(a("Saturday")), a("Saturday", "lo"), a("Saturday", "hi"),
         a("Friday", "mean_aqi"), A(a("Eid")), a("Eid", "lo"), a("Eid", "hi"),
         A(a("Lockdown")), a("Lockdown", "lo"), a("Lockdown", "hi")])
    add("scenarios (text)", "Against an Embassy baseline of",
        [sc("baseline", "mean_pm"), sc("baseline", "mean_aqi"), sc("baseline", "days150"),
         A(sc("rest_day", "d_mean_aqi")), A(sc("rest_day", "d_days150")), A(sc("rest_day", "d_days200")),
         sc("rest_day_lo", "d_mean_aqi"), sc("rest_day_hi", "d_mean_aqi"),
         A(sc("cut_allday", "d_mean_aqi")), A(sc("cut_allday", "d_days150")),
         A(sc("cut_day_only", "d_mean_aqi")), A(sc("cut_night_only", "d_mean_aqi")), night_gain,
         A(rest_rng.max()), A(rest_rng.min()),
         A(sc("eid_like", "d_mean_aqi")), A(sc("eid_like", "d_days150")),
         A(sc("lockdown_like", "d_mean_aqi")), A(sc("lockdown_like", "d_days150")),
         sc("lockdown_like", "mean_pm"), sc("lockdown_like", "mean_pm") / 35.0])

    # ---------------------------------------------------------------- abstract
    add("abstract", "measuring activity with same-site",
        [A(r.bootstrap("Dhaka", PM, "afternoon 12-17")), A(r.bootstrap("Dhaka", PM, "afternoon 12-17", "hi_pct")),
         A(r.bootstrap("Dhaka", PM, "afternoon 12-17", "lo_pct")),
         A(r.bootstrap("Dhaka", PM, "evening 18-23")), A(r.bootstrap("Dhaka", PM, "evening 18-23", "hi_pct")),
         A(r.bootstrap("Dhaka", PM, "evening 18-23", "lo_pct")),
         A(r.bootstrap("Dhaka", PM, "daily")), A(a("Friday")), A(sc("rest_day", "d_mean_aqi")),
         A(r.event("Eid", "daily")), A(r.event("Lockdown", "daily")), A(a("Eid")), A(a("Lockdown"))])

    # ---------------------------------------------------------------- discussion
    add("discussion 5.1", "On the weekly rest day, \\pmtf{} falls by",
        [A(r.bootstrap("Dhaka", PM, "afternoon 12-17")), A(r.bootstrap("Dhaka", PM, "evening 18-23")),
         min(sh(s, range(0, 12)) for s in dry), max(sh(s, range(0, 12)) for s in dry), min(aft), max(aft),
         A(r.bootstrap("Dhaka", PM, "afternoon 12-17")), A(r.bootstrap("Dhaka", PM, "evening 18-23")),
         A(r.bootstrap("Dhaka", PM, "daily")), A(a("Friday")),
         blh("Winter", [15]), blh("Winter", [18]), min(eve), max(eve)])
    add("discussion evening", "largest at the national stations (",
        [r.bootstrap("national", PM, "evening 18-23")])
    add("discussion before noon", "identical hourly cut buys", [night_gain])
    add("discussion 5.2 rest night", "would make it; the rest night sees lower",
        [-r.pooled("main", "national", "NOX", "Sat", "night 00-05")])
    add("discussion 5.2 Dhaka rest night", "On the rest night in Dhaka",
        [-sat_dh.mean(), -r.hourly("Dhaka", "NOX", "Sat", range(0, 6)).max(),
         -r.hourly("Dhaka", "NOX", "Sat", range(0, 7)).min(),
         -r.pooled("main", "national", PM, "Sat", "night 00-05"),
         -r.pooled("main", "national", "NOX", "Sat", "night 00-05"),
         A(r.bootstrap("national", PM, "afternoon 12-17")), A(r.bootstrap("national", "NOX", "afternoon 12-17"))])
    add("discussion 5.3 tracers", "over the day: surface \\nox{} and CO",
        [A(r.bootstrap("national", "NOX", "daily")), A(r.bootstrap("national", "CO", "daily")),
         A(max(r.bootstrap("national", t, b) for t in ("NOX", "CO") for b in ("afternoon 12-17", "evening 18-23"))),
         A(min(r.bootstrap("national", t, b) for t in ("NOX", "CO") for b in ("afternoon 12-17", "evening 18-23")))])
    add("discussion 5.3 Eid and lockdown", "reach the night and suppress far more than traffic",
        [A(r.event("Eid", "daily")), A(r.event("Lockdown", "daily"))])
    I2 = [r.event(e, "daily", "I2") for e in ("Eid", "Lockdown")]
    add("limitations I2", "effects for the whole-city shocks are heterogeneous", I2)

    # ---------------------------------------------------------------- conclusions
    add("conclusions 1", "tracers fall from morning to midnight",
        [A(r.bootstrap("Dhaka", PM, "afternoon 12-17")), A(r.bootstrap("Dhaka", PM, "evening 18-23")),
         A(r.bootstrap("national", PM, "evening 18-23"))])
    add("conclusions 2", "morning layer, the hours the rest day does not reach",
        [A(r.bootstrap("Dhaka", PM, "daily")), A(a("Friday")), A(sc("rest_day", "d_mean_aqi"))])
    add("conclusions 3", "suppress activity of every kind around the clock",
        [A(r.event("Eid", "daily")), A(r.event("Lockdown", "daily")), A(a("Eid")), A(a("Lockdown"))])
    return C


def supp_claims(r):
    """Claims for the numbers in supplementary.tex prose (its tables are generated)."""
    C = []
    add = lambda name, anchor, values: C.append((name, anchor, values))
    rep = csv("cams_parse_report.csv")
    f12 = rep[rep.file.str.contains("2012") & rep.station.notna()].set_index("station").dropped_no_date
    tot = int(rep[rep.file.str.contains("2012") & rep.station.isna()].dropped_no_date.iloc[0])
    six = ["Rajshahi", "Khulna", "Sangsad", "Rangpur", "Narsingdi", "Barishal"]
    add("S1 dropped blocks", "in time and were dropped:",
        [int(f12[s]) for s in six] + [tot - int(sum(f12[s] for s in six)), tot])
    q = pd.read_csv(os.path.join(HERE, "..", "data", "cams_hourly_qc.csv.gz"),
                    usecols=["station", "time", "PM25"], parse_dates=["time"])
    q19 = q[q.time >= "2019-01-01"]
    daily = q19.assign(date=q19.time.dt.normalize()).groupby(["station", "date"]).PM25.mean().unstack(0)
    dh = ["BARC", "Darussalam", "DoE", "Savar", "Gazipur", "Narayanganj", "Narsingdi"]
    dmed = float(np.median([daily[s].median() for s in dh]))
    btv = float(daily["BTV Chattogram"].median())
    add("S2 BTV level", "The host\nlisting settles its identity", [btv, btv / dmed, dmed])
    eid = pd.read_csv(os.path.join(ROOT, "Traffic", "data_availability", "02_event_calendars", "eid_dates.csv"),
                      parse_dates=["date"])
    ns = q[q.station == "Narsingdi"].set_index("time").PM25
    cov = []
    for d in eid.date:
        if d < ns.index.min():
            continue
        w = pd.date_range(d - pd.Timedelta(days=1), d + pd.Timedelta(days=3, hours=23), freq="h")
        cov.append((int(ns.reindex(w).notna().sum()), len(w)))
    fl = csv("cams_flatline_check.csv")
    cv_ns = float(fl[(fl.station == "Narsingdi") & (fl.tracer == "NOX")].daily_cv30.iloc[0])
    add("S7 Narsingdi", "at Narsingdi rose (",
        [r.event_station("Eid", "Narsingdi"), 100 * sum(a for a, _ in cov) / sum(b for _, b in cov), len(cov),
         min(100 * a / b for a, b in cov), max(100 * a / b for a, b in cov), cv_ns])
    CITY = ("US Embassy", "BARC", "Darussalam")
    pre = pd.concat([r.eid_rel(s, range(-4, -1)) for s in CITY])
    city = pd.concat([r.eid_rel(s, range(0, 5)) for s in CITY])
    post = pd.concat([r.eid_rel(s, range(5, 9)) for s in CITY])
    corr = pd.concat([r.eid_rel(s, range(-5, 0)) for s in ("Narsingdi", "Savar")])
    add("S7 timing", "At the city sites \\pmtf{}\nstays within",
        [pre.abs().max(), -city.max(), -city.min(), -post.max(), -post.min(),
         int((corr > 0).sum()), len(corr), corr.max()])
    ev = r.eev[(r.eev.station == "Narsingdi") & (r.eev.tracer == "PM25")].pct
    add("S7 by event", "the Narsingdi effect is positive in", [int((ev > 0).sum()), len(ev)])
    return C


# ------------------------------------------------------------------ comparison
def compare(expected, kind, vals, decs):
    if isinstance(expected, tuple):
        tag = expected[0]
        if tag == "abs":
            e = [abs(expected[1])]
            return kind == "num" and close(e[0], abs(vals[0]), decs[0]), e
        if tag == "about":
            return kind == "num" and abs(abs(expected[1]) - abs(vals[0])) <= expected[2] + 1e-9 and \
                np.sign(expected[1]) * np.sign(vals[0]) >= 0 or (kind == "num" and abs(expected[1] - vals[0]) <= expected[2] + 1e-9), [expected[1]]
        if tag == "range":
            ok = kind == "range" and close(expected[1], vals[0], decs[0]) and close(expected[2], vals[1], decs[1])
            return ok, [expected[1], expected[2]]
        if tag == "all":            # one printed number standing for several derived ones
            res = [compare(x, kind, vals, decs)[0] for x in expected[1]]
            return all(res), [x[1] if isinstance(x, tuple) else x for x in expected[1]]
        if tag == "bound_above":    # printed number is an upper bound ("stays below 400")
            return kind == "num" and vals[0] >= expected[1], [expected[1]]
        if tag == "rangeabout":
            ok = kind == "range" and abs(expected[1] - vals[0]) <= expected[3] + 0.5 * 10 ** -decs[0] + 1e-9 and \
                abs(expected[2] - vals[1]) <= expected[3] + 0.5 * 10 ** -decs[1] + 1e-9
            return ok, [expected[1], expected[2]]
        raise ValueError(tag)
    return kind == "num" and close(float(expected), vals[0], decs[0]), [float(expected)]


def refmt(raw, exp, kind, decs):
    """The derived value written in the printed number's style, or None when a person must
    rewrite the sentence (one printed number standing for several values, or a bound)."""
    tag = exp[0] if isinstance(exp, tuple) else "exact"
    if tag in ("all", "bound_above"):
        return None
    math = "$" in raw
    comma = "," in raw

    def one(v, d, part):
        signed = part.strip("$").strip()[:1] in "+-"
        s = ("%+." if signed else "%.") + str(d) + "f"
        v = round_half_up(v, d)
        txt = s % v
        if txt in ("-0", "+0") or txt.startswith(("-0.0", "+0.0")) and float(txt) == 0:
            txt = txt.replace("-", "+")
        if comma:
            ip, _, fp = txt.lstrip("+-").partition(".")
            txt = (txt[0] if txt[0] in "+-" else "") + "{:,}".format(int(ip)) + ("." + fp if fp else "")
        return txt

    if tag == "range" or tag == "rangeabout":
        parts = raw.replace("$", "").split("--")
        a = one(exp[1], decs[0], parts[0])
        b = one(exp[2], decs[1], parts[1])
        return ("$%s$--$%s$" % (a, b)) if math else "%s--%s" % (a, b)
    v = abs(exp[1]) if tag == "abs" else (exp[1] if tag == "about" else float(exp))
    if tag == "about" and not raw.replace("$", "").strip()[:1] in "+-":
        v = abs(v)
    txt = one(v, decs[0], raw)
    return "$%s$" % txt if math else txt


def close(expected, printed, d):
    return abs(round_half_up(expected, d) - printed) < 1e-9 or abs(expected - printed) <= 0.5 * 10 ** -d + 1e-9


def main():
    text = open(TEX, encoding="utf-8").read()
    items = amber(text)
    if "--inventory" in sys.argv:
        for i, (pos, raw) in enumerate(items):
            line = text.count("\n", 0, pos) + 1
            ctx = re.sub(r"\s+", " ", text[max(0, pos - 60):pos])
            print("%3d  L%-4d %-16s ...%s" % (i, line, raw, ctx[-60:]))
        return 0
    r = R()
    used, kinds, fixes = {}, {}, []
    fails, ok_n = [], 0
    for name, anchor, values in claims(r):
        pat = r"\s+".join(re.escape(w) for w in anchor.split())
        hits = [m.start() for m in re.finditer(pat, text)]
        if len(hits) != 1:
            fails.append("ANCHOR  %-40s anchor found %d times: %r" % (name, len(hits), anchor[:50]))
            continue
        after = [k for k, (pos, _) in enumerate(items) if pos > hits[0]][:len(values)]
        if len(after) < len(values):
            fails.append("ANCHOR  %-40s fewer amber numbers after anchor than claimed" % name)
            continue
        for j, (k, exp) in enumerate(zip(after, values)):
            used.setdefault(k, []).append(name)
            pos, raw = items[k]
            try:
                kind, vals, decs = parse_printed(raw)
            except ValueError:
                fails.append("PARSE   %-40s #%d cannot parse %r" % (name, j, raw))
                continue
            ok, e = compare(exp, kind, vals, decs)
            tag = exp[0] if isinstance(exp, tuple) else "exact"
            kinds[{"abs": "exact", "range": "exact", "all": "exact"}.get(tag, tag)] = \
                kinds.get({"abs": "exact", "range": "exact", "all": "exact"}.get(tag, tag), 0) + 1
            if ok:
                ok_n += 1
            else:
                line = text.count("\n", 0, pos) + 1
                fails.append("MISMATCH %-39s #%-2d L%-4d printed %-12s derived %s" %
                             (name, j, line, raw, ", ".join("%.4g" % x for x in e)))
                new = refmt(raw, exp, kind, decs)
                if new is not None:
                    fixes.append((pos, raw, new, line, name))
    # the highlights file carries no \pn{} marks: check its numbers line by line
    hl_path = os.path.join(os.path.dirname(TEX), "highlights.txt")
    if os.path.exists(hl_path):
        nm = [r.block_share(s, range(0, 12)) for s in ("Winter", "Pre-monsoon", "Post-monsoon", "Monsoon")]
        want = [[], [-r.bootstrap("Dhaka", "PM25", "afternoon 12-17"), -r.bootstrap("Dhaka", "PM25", "evening 18-23")],
                [min(nm), max(nm)], [-float(r.scen.loc["rest_day", "d_mean_aqi"])],
                [-r.event("Eid", "daily"), -r.event("Lockdown", "daily")]]
        lines = open(hl_path, encoding="utf-8").read().splitlines()
        for i, (line, exp) in enumerate(zip(lines, want)):
            nums = re.findall(r"\d+\.\d+|\d+", line.replace("PM2.5", ""))
            ok = len(nums) == len(exp) and all(close(e, float(n), dec(n)) for e, n in zip(exp, nums))
            if not ok:
                fails.append("HIGHLIGHT line %d: printed %s, derived %s" %
                             (i + 1, nums, ", ".join("%.4g" % e for e in exp)))
        if len(lines) > 5 or any(len(x) > 85 for x in lines):
            fails.append("HIGHLIGHT more than 5 lines or a line over 85 characters")
    # the supplement: same rules, its own claims
    supp_path = os.path.join(os.path.dirname(TEX), "supplementary.tex")
    n_supp = ok_supp = 0
    if os.path.exists(supp_path):
        st = open(supp_path, encoding="utf-8").read()
        sitems = amber(st)
        n_supp = len(sitems)
        sused = set()
        for name, anchor, values in supp_claims(r):
            pat = r"\s+".join(re.escape(w) for w in anchor.split())
            hits = [m.start() for m in re.finditer(pat, st)]
            if len(hits) != 1:
                fails.append("SUPP ANCHOR %-36s anchor found %d times" % (name, len(hits)))
                continue
            after = [k for k, (pos, _) in enumerate(sitems) if pos > hits[0]][:len(values)]
            for j, (k, exp) in enumerate(zip(after, values)):
                sused.add(k)
                pos, raw = sitems[k]
                kind, vals, decs = parse_printed(raw)
                ok, e = compare(exp, kind, vals, decs)
                if ok:
                    ok_supp += 1
                else:
                    fails.append("SUPP MISMATCH %-34s #%-2d L%-4d printed %-10s derived %s" %
                                 (name, j, st.count("\n", 0, pos) + 1, raw, ", ".join("%.4g" % x for x in e)))
        for k in range(n_supp):
            if k not in sused:
                pos, raw = sitems[k]
                fails.append("SUPP UNCOVERED L%-4d %s" % (st.count("\n", 0, pos) + 1, raw))
    uncovered = [k for k in range(len(items)) if k not in used]
    double = [k for k, v in used.items() if len(v) > 1]
    for k in uncovered:
        pos, raw = items[k]
        ctx = re.sub(r"\s+", " ", text[max(0, pos - 50):pos])
        fails.append("UNCOVERED L%-4d %-14s ...%s" % (text.count("\n", 0, pos) + 1, raw, ctx[-50:]))
    for k in double:
        fails.append("DOUBLE   amber #%d claimed by %s" % (k, used[k]))
    for f in fails:
        print(f)
    if "--fix" in sys.argv and fixes:
        # Rewrite mismatched numbers in their printed style, last first so positions hold.
        # Only the number changes; the sentence around it is left for a person to reread.
        out = text
        for pos, raw, new, line, name in sorted(fixes, reverse=True):
            old = r"\pn{%s}" % raw
            assert out[pos:pos + len(old)] == old, (line, raw)
            out = out[:pos] + r"\pn{%s}" % new + out[pos + len(old):]
            print("FIXED    L%-4d %-12s -> %-12s (%s)" % (line, raw, new, name))
        open(TEX, "w", encoding="utf-8").write(out)
        print("%d numbers rewritten in %s; reread the sentences around them" % (len(fixes), TEX))
    print("\n%d of %d amber numbers verified; %d problems" % (ok_n, len(items), len(fails)))
    if n_supp:
        print("supplement: %d of %d checked numbers verified" % (ok_supp, n_supp))
    print("checks by type: %s  (exact = equal at the printed precision, half-up; about = within a "
          "stated tolerance, used for numbers the text gives as 'about'; bound_above = printed upper "
          "bound)" % ", ".join("%s %d" % kv for kv in sorted(kinds.items())))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
