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
        self.bd = csv("robust_bootstrap_days.csv")
        self.bh = csv("robust_bh_bootstrap.csv")
        self.bl = csv("robust_bootstrap_blocklen.csv")
        self.pool = csv("robust_pooled.csv")
        self.hp = csv("hourly_profiles.csv")
        self.share = csv("pm_share_by_hour.csv")
        self.wcheck = csv("weighting_check.csv")
        self.ev = csv("events_pooled.csv")
        self.evst = csv("events_station_units.csv")
        self.aqi = csv("aqi_by_shock.csv").set_index("event")
        self.scen = csv("aqi_scenarios.csv").set_index("scenario")
        self.qc = csv("cams_qc_report.csv").set_index("rule")
        self.val = csv("cams_validation.csv").set_index("station")
        self.eidc = csv("eid_checks.csv")
        self.etraj = csv("eid_trajectory.csv")
        self.lock = csv("lockdown_checks.csv")
        self.etr = csv("events_tracers.csv")
        self.rec = csv("reconstruct.csv")
        self.mech = csv("mechanism.csv")
        self.dom = csv("aqi_dominance.csv")
        self.ann = csv("aqi_annualization.csv")
        self.gap = csv("gapfill_errors.csv").set_index("subset")
        self.strike = csv("strike_local.csv")
        self.flat = csv("flatline_diagnostics.csv")
        self.miss = csv("missingness_dow.csv")
        self.clock = csv("clock_check.csv").set_index("station")
        self.eev = csv("eid_by_event.csv")

    # -- helpers
    def mobility(self, shock_prefix, col):
        rows = self.mob[self.mob.index.str.startswith(shock_prefix)]
        return rows[col]

    def activity(self, shock, outcome, region, field="pct"):
        q = self.act[(self.act.shock == shock) & (self.act.outcome == outcome) & (self.act.region == region)]
        assert len(q) == 1, (shock, outcome, region, len(q))
        return float(q[field].iloc[0])

    def boot(self, region, tracer, block, day="Fri", field="pct"):
        """Joint-date bootstrap table (robust_bootstrap_days.csv)."""
        q = self.bd[(self.bd.region == region) & (self.bd.tracer == tracer) & (self.bd.block == block)
                    & (self.bd.day == day)]
        assert len(q) == 1, (region, tracer, block, day)
        return float(q[field].iloc[0])

    def bootaqi(self, region, day, field="b_point"):
        return self.boot(region, "AQI", "daily", day, field)

    def q_bh(self, region, tracer, block, day="Fri"):
        q = self.bh[(self.bh.region == region) & (self.bh.tracer == tracer) & (self.bh.block == block)
                    & (self.bh.day == day)]
        assert len(q) == 1
        return float(q.q_boot.iloc[0])

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

    def eid(self, analysis, outcome="PM25", field=None):
        q = self.eidc[(self.eidc.analysis == analysis) & (self.eidc.outcome == outcome)]
        assert len(q) == 1, (analysis, outcome)
        f = field or ("b_pct" if outcome == "PM25" else "b")
        return float(q[f].iloc[0])

    def lk(self, analysis, outcome="PM25", field=None):
        q = self.lock[(self.lock.analysis == analysis) & (self.lock.outcome == outcome)]
        assert len(q) == 1, (analysis, outcome)
        f = field or ("b_pct" if outcome == "PM25" else "b")
        return float(q[f].iloc[0])

    def tracer_ev(self, shock, outcome, region, field):
        q = self.etr[(self.etr.shock == shock) & (self.etr.outcome == outcome) & (self.etr.region == region)]
        assert len(q) == 1, (shock, outcome, region)
        return float(q[field].iloc[0])

    def traj(self, group, rels):
        q = self.etraj[(self.etraj.group == group) & self.etraj.rel.isin(list(rels))]
        return q.set_index("rel").pct

    def recon(self, region, season, field):
        q = self.rec[(self.rec.region == region) & (self.rec.season == season)]
        assert len(q) == 1
        return float(q[field].iloc[0])

    def mechanism(self, test, region, block, field="est"):
        q = self.mech[(self.mech.test == test) & (self.mech.region == region) & (self.mech.block == block)]
        assert len(q) == 1, (test, region, block)
        return float(q[field].iloc[0])

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


# ------------------------------------------------------------------ expected values
# Each value is one of:
#   number                 compared at the printed precision (half-up)
#   ("abs", number)        compared as a magnitude (the text says "x percent lower")
#   ("about", number, tol) compared within an absolute tolerance (no longer used: every
#                          number in the manuscript is printed exactly)
#   ("range", lo, hi)      a printed "a--b" range
#   ("bound_above", x)     the printed number is an upper bound ("q < 0.001")
def claims(r):
    A = lambda v: ("abs", v)
    PM = "PM25"
    AFT, EVE, NIGHT, MORN, DAY = "afternoon 12-17", "evening 18-23", "night 00-05", "morning 06-11", "daily"
    C = []

    def add(name, anchor, values):
        C.append((name, anchor, values))

    def tri_boot(region, tracer, block, day="Fri"):
        return [r.boot(region, tracer, block, day, f) for f in ("pct", "lo_pct", "hi_pct")]

    sh = lambda s, h: r.block_share(s, h)
    dry = ["Winter", "Pre-monsoon", "Post-monsoon"]
    aft = [sh(s, range(12, 18)) for s in dry]
    nm = [sh(s, range(0, 12)) for s in dry + ["Monsoon"]]
    eve = [sh(s, range(18, 24)) for s in dry]
    sc = lambda k, f: float(r.scen.loc[k, f])
    night_gain = 100 * (sc("cut_night_only", "d_mean_aqi") / sc("cut_day_only", "d_mean_aqi") - 1)

    # ---------------------------------------------------------------- abstract
    add("abstract", "\\nox{} falls from morning and \\pmtf{} from midday:",
        [A(r.boot("Dhaka", PM, AFT)), A(r.boot("Dhaka", PM, AFT, field="hi_pct")),
         A(r.boot("Dhaka", PM, AFT, field="lo_pct")),
         A(r.boot("Dhaka", PM, EVE)), A(r.boot("Dhaka", PM, EVE, field="hi_pct")),
         A(r.boot("Dhaka", PM, EVE, field="lo_pct"))]
        + tri_boot("Dhaka", PM, DAY)
        + [A(r.eid("main")), A(r.eid("main", "AQI")), A(sc("rest_day", "d_mean_aqi"))])

    # ---------------------------------------------------------------- introduction
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
    add("Fridays 2019-2024 (Introduction)", "partial rest day) recurs", [fridays])

    # ---------------------------------------------------------------- data
    rep = csv("cams_parse_report.csv")
    add("rows without a date", "dropped rather than reconstructed:",
        [int(rep[rep.file.str.contains("2012") & rep.station.isna()].dropped_no_date.iloc[0])])
    add("QC stuck and PM2.5>PM10 counts", "Over the full record these rules removed",
        [r.qc.loc["Q2 stuck runs PM25", "n"], r.qc.loc["Q3 PM2.5 > 1.1 PM10", "n"]])
    whole = r.miss[r.miss.hours == "all"].diff_pp
    blk = r.miss[r.miss.hours != "all"]
    dpm = r.miss[(r.miss.station == "DoE") & (r.miss.pollutant == "PM25") & (r.miss.hours == EVE)].diff_pp
    add("missingness by day of week", "the share of valid hours on Fridays differs from that on Sunday to Wednesday by",
        [whole.min(), whole.max(), int((blk.diff_pp.abs() > 5).sum()), len(blk), float(dpm.iloc[0])])
    dh = ["BARC", "Darussalam", "DoE", "Savar", "Gazipur", "Narayanganj", "Narsingdi"]
    rr = r.val.loc[dh, "daily_r_embassy"]
    add("PM2.5 r range with Embassy", "correlates with the Embassy record at $r=$",
        [("range", rr.min(), rr.max())])
    others = r.clock.drop("Savar").sr_centroid_h.dropna()
    add("Savar clock", "Savar's \\pmtf{} peaks",
        [int(r.val.loc["Savar", "best_lag_h"]), r.clock.loc["Savar", "sr_centroid_h"],
         ("range", others.min(), others.max()), int(r.clock.loc["Savar", "lag_PM25"])])
    fl = r.flat[r.flat.flat]
    kept = r.flat[~r.flat.flat]
    dco = r.flat[(r.flat.station == "Darussalam") & (r.flat.pollutant == "CO")]
    add("fault rule", "their mean hourly profiles vary by only",
        [("range", fl.diurnal_range.min(), fl.diurnal_range.max()), kept.diurnal_range.median(),
         fl.r_daily_pm25.min(), fl.r_daily_pm25.max(), float(dco.r_daily_pm25.iloc[0])])
    import cams_shocks
    trop = cams_shocks.tropomi_daily(ROOT, cams_shocks.TROPOMI_MIN_COVERAGE).loc["2019-01-01":"2025-12-31"]
    add("TROPOMI coverage", "are observed are dropped, leaving",
        [int(trop.notna().sum()), len(pd.date_range("2019-01-01", "2025-12-31"))])
    cv = pd.read_csv(os.path.join(ROOT, "06_results", "metrics", "blh_gapfill_cv.csv"))
    met = pd.read_csv(os.path.join(ROOT, "02_data", "raw", "meteo_dhaka_hourly_filled.csv"),
                      usecols=["blh_filled"])
    add("BLH gap fill", "hours stay finite. A",
        [int(met.blh_filled.sum()), len(cv), cv.r2.mean(), cv.rmse.mean(), cv.bias.mean(),
         r.gap.loc["hours 00-05", "rmse"], r.gap.loc["observed < 200 m", "rmse"], r.gap.loc["observed < 200 m", "bias"]])
    add("usable Eid events", "fell inside the lockdown, leaving", [int(r.eid("main", field="n_events"))])
    hart = pd.read_csv(os.path.join(ROOT, "Traffic", "data_availability", "02_event_calendars",
                                    "hartal_event_days_DERIVED.csv"), parse_dates=["date"])
    hd = pd.read_csv(os.path.join(ROOT, "Traffic", "data_availability", "02_event_calendars",
                                  "hartal_event_days_DHAKA_DISTRICT_DERIVED.csv"), parse_dates=["date"])
    in_period = lambda d: d[(d.date >= "2019-01-01") & (d.date <= "2024-12-31")]
    add("strike calendars", "terms require. Of the", [len(in_period(hart)), len(in_period(hd))])

    # ---------------------------------------------------------------- methods
    add("BH family size", "procedure over one family of", [len(r.bh)])
    add("Eid events resampled", "the main interval resamples the", [int(r.eid("event_boot", field="n_events"))])
    cams_dom = 100 * r.dom[r.dom.source.str.startswith("CAMS")].share_other_higher
    add("PM10 dominance", "exceeds it at the CAMS stations on", [("range", cams_dom.min(), cams_dom.max())])
    a0 = r.ann.iloc[0]
    add("annualization", "valid-day coverage is",
        [100 * r.ann.coverage.min(), 100 * r.ann.coverage.max(), abs(a0.days_gt150_simple - a0.days_gt150_balanced)])

    # ---------------------------------------------------------------- 4.1 activity
    lock_t = r.mobility("lockdown", "transit").iloc[0]
    lock_w = r.mobility("lockdown", "workplaces").iloc[0]
    lock_r = r.mobility("lockdown", "residential").iloc[0]
    eid_w = r.mobility("Eid", "workplaces")
    eid_t = r.mobility("Eid", "transit")
    st_t = r.mobility("strike", "transit").iloc[0]
    st_w = r.mobility("strike", "workplaces").iloc[0]
    st_n = int(r.mobility("strike", "n_days").iloc[0])
    add("mobility and NO2 for Eid and lockdown (text)", "The lockdown lowered national mobility",
        [A(lock_t), A(lock_w), lock_r, A(eid_w.mean()), A(eid_w.max()), A(eid_w.min()), eid_t.mean(),
         A(r.activity("Eid", "TROPOMI NO2", "domain")), A(r.activity("Lockdown", "TROPOMI NO2", "domain"))])
    sd = r.strike[r.strike.calendar == "Dhaka district only"].set_index("outcome")
    add("strikes (text)", "Strike days changed activity little. On the",
        [st_n, st_t, st_w, r.activity("Strike days", "NOX", "Dhaka"), r.activity("Strike days", "NOX", "Dhaka", "lo"),
         r.activity("Strike days", "NOX", "Dhaka", "hi"), A(r.activity("Strike days", "TROPOMI NO2", "domain")),
         r.activity("Strike days", "TROPOMI NO2", "domain", "lo"), r.activity("Strike days", "TROPOMI NO2", "domain", "hi"),
         A(sd.loc["TROPOMI NO2", "pct"]), sd.loc["TROPOMI NO2", "lo"], sd.loc["TROPOMI NO2", "hi"]])

    def tri(shock, outcome, region):
        return [r.activity(shock, outcome, region, k) for k in ("pct", "lo", "hi")]

    def tri_ev(shock, outcome, region):
        return [r.tracer_ev(shock, outcome, region, k) for k in ("b_pct", "lo_pct", "hi_pct")]
    add("Table 2 Friday", "Friday & weekly rest day;",
        [fridays] + tri("Friday", "TROPOMI NO2", "domain") + tri_boot("national", "NOX", DAY)
        + tri_boot("national", "CO", DAY))
    add("Table 2 Saturday", "Saturday & partial rest day &",
        tri("Saturday", "TROPOMI NO2", "domain") + tri_boot("national", "NOX", DAY, "Sat")
        + tri_boot("national", "CO", DAY, "Sat"))
    add("Table 2 Eid", "Eid & $-1$ to $+3$ days;",
        [int(r.eid("main", field="n_events")), eid_w.mean(), eid_t.mean()] + tri_ev("Eid", "TROPOMI NO2", "domain")
        + tri_ev("Eid", "NOX", "Dhaka") + tri_ev("Eid", "CO", "Dhaka"))
    add("Table 2 lockdown", "Lockdown & 26 Mar -- 30 May 2020 &",
        [lock_t, lock_w, lock_r] + tri_ev("Lockdown", "TROPOMI NO2", "domain"))
    add("Table 2 strikes", "Strike days & ACLED-derived;",
        [len(in_period(hart)), st_n, st_t, st_w] + tri("Strike days", "TROPOMI NO2", "domain")
        + tri("Strike days", "NOX", "national") + tri("Strike days", "CO", "national"))

    nox_mid = r.hourly("national", "NOX", "Fri", range(13, 18))
    co_mid = r.hourly("national", "CO", "Fri", range(13, 18))
    co_eve = r.hourly("national", "CO", "Fri", range(18, 24))
    add("weekly tracers (text)", "result. The \\notwo{} column is",
        [A(r.activity("Friday", "TROPOMI NO2", "domain")), A(r.activity("Saturday", "TROPOMI NO2", "domain")),
         A(nox_mid.max()), A(nox_mid.min()), A(co_mid.max()), A(co_mid.min()), A(co_eve.max()), A(co_eve.min()),
         A(r.boot("national", "NOX", AFT)), r.boot("national", "NOX", AFT, field="lo_pct"),
         r.boot("national", "NOX", AFT, field="hi_pct"), A(r.boot("national", "NOX", EVE)),
         A(r.boot("national", "CO", AFT)), A(r.boot("national", "CO", EVE)),
         A(r.boot("national", "NOX", NIGHT)), r.boot("national", "NOX", NIGHT, field="lo_pct"),
         r.boot("national", "NOX", NIGHT, field="hi_pct"),
         A(r.boot("Dhaka", "NOX", NIGHT)), r.boot("Dhaka", "NOX", NIGHT, field="lo_pct"),
         r.boot("Dhaka", "NOX", NIGHT, field="hi_pct"),
         A(r.boot("national", "NOX", NIGHT, "Sat")), r.boot("national", "NOX", NIGHT, "Sat", "lo_pct"),
         r.boot("national", "NOX", NIGHT, "Sat", "hi_pct"),
         A(r.boot("Dhaka", "NOX", NIGHT, "Sat")), r.boot("Dhaka", "NOX", NIGHT, "Sat", "lo_pct"),
         r.boot("Dhaka", "NOX", NIGHT, "Sat", "hi_pct")])
    nc_nat = r.hourly("national", "NOX_CO", "Fri", range(0, 4))
    nc_dh = r.hourly("Dhaka", "NOX_CO", "Fri", range(3, 6))
    nc_dh_sat = r.hourly("Dhaka", "NOX_CO", "Sat", range(3, 8))
    add("NOx/CO ratio", "where both analyzers pass, it is",
        [nc_nat.min(), nc_nat.max(), nc_dh.min(), nc_dh.max(), A(nc_dh_sat.max()), A(nc_dh_sat.min())])

    # ---------------------------------------------------------------- 4.2 PM2.5
    fri_am = r.hourly("Dhaka", PM, "Fri", range(0, 12))
    fri_pm = r.hourly("Dhaka", PM, "Fri", range(13, 24))
    paft = r.placebo_range("Dhaka", AFT)
    peve = r.placebo_range("Dhaka", EVE)
    add("Dhaka hourly and blocks", "midnight to late morning (",
        [fri_am.min(), fri_am.max(), A(fri_pm.max()), A(fri_pm.min())]
        + [A(r.boot("Dhaka", PM, AFT)), r.boot("Dhaka", PM, AFT, field="lo_pct"), r.boot("Dhaka", PM, AFT, field="hi_pct"),
           A(r.boot("Dhaka", PM, EVE)), r.boot("Dhaka", PM, EVE, field="lo_pct"), r.boot("Dhaka", PM, EVE, field="hi_pct"),
           paft[0], paft[1], peve[0], peve[1], len(r.bh), r.q_bh("Dhaka", PM, EVE), r.q_bh("Dhaka", PM, AFT),
           A(r.boot("Dhaka", PM, NIGHT)), r.boot("Dhaka", PM, NIGHT, field="lo_pct"),
           r.boot("Dhaka", PM, NIGHT, field="hi_pct"), A(r.boot("Dhaka", PM, MORN)),
           A(r.boot("Dhaka", PM, DAY)), r.boot("Dhaka", PM, DAY, field="lo_pct"), r.boot("Dhaka", PM, DAY, field="hi_pct")])
    nat_h = r.hourly("national", PM, "Fri", range(12, 24))
    add("national blocks", "show the same timing with a stronger evening:",
        [A(r.boot("national", PM, AFT)), r.boot("national", PM, AFT, field="lo_pct"),
         r.boot("national", PM, AFT, field="hi_pct"), r.q_bh("national", PM, AFT),
         A(r.boot("national", PM, EVE)), r.boot("national", PM, EVE, field="lo_pct"),
         r.boot("national", PM, EVE, field="hi_pct"), ("bound_above", r.q_bh("national", PM, EVE)),
         nat_h.min(), r.boot("national", PM, NIGHT), r.boot("national", PM, DAY),
         r.boot("national", PM, DAY, field="lo_pct"), r.boot("national", PM, DAY, field="hi_pct")])
    i2 = lambda reg, b: r.pooled("main", reg, PM, "Fri", b, "I2")
    add("heterogeneity", "The station effects are homogeneous in Dhaka",
        [i2("Dhaka", AFT), i2("Dhaka", EVE), i2("national", AFT), i2("national", EVE)])
    add("Saturday and Thursday", "shows a smaller version of the Friday pattern",
        [A(r.boot("Dhaka", PM, AFT, "Sat")), A(r.boot("Dhaka", PM, EVE, "Sat")),
         A(r.boot("Dhaka", PM, DAY, "Sat")), r.boot("Dhaka", PM, DAY, "Sat", "lo_pct"),
         r.boot("Dhaka", PM, DAY, "Sat", "hi_pct"),
         A(r.boot("Dhaka", PM, NIGHT, "Sat")), r.boot("Dhaka", PM, NIGHT, "Sat", "lo_pct"),
         r.boot("Dhaka", PM, NIGHT, "Sat", "hi_pct"),
         A(r.boot("Dhaka", PM, MORN, "Thu")), r.boot("Dhaka", PM, MORN, "Thu", "lo_pct"),
         r.boot("Dhaka", PM, MORN, "Thu", "hi_pct"),
         A(r.boot("national", PM, AFT)), A(r.boot("national", "NOX", AFT)), A(r.boot("national", "CO", AFT)),
         A(r.boot("national", PM, EVE)), A(r.boot("national", "NOX", EVE)), A(r.boot("national", "CO", EVE))])

    loo = r.pool[(r.pool.region == "Dhaka") & (r.pool.tracer == PM) & (r.pool.day == "Fri")
                 & r.pool.variant.str.startswith("loo_")]
    la = loo[loo.block == AFT].z
    le = loo[loo.block == EVE].z
    pz = lambda v, b: r.pooled(v, "Dhaka", PM, "Fri", b, "z")
    pp = lambda v, b: r.pooled(v, "Dhaka", PM, "Fri", b)
    infl = r.boot("Dhaka", PM, AFT, field="se_inflation")
    m14 = r.bl[r.bl.block_days == 14].set_index(["region", "block"])
    moves = []
    for L in (7, 28):
        o = r.bl[r.bl.block_days == L].set_index(["region", "block"])
        moves += list((o.lo_pct - m14.lo_pct).abs()) + list((o.hi_pct - m14.hi_pct).abs())
    add("robustness", "With week-clustered errors the Dhaka",
        [pz("cluster", AFT), pz("cluster", EVE), la.max(), la.min(), le.max(), le.min(),
         A(pp("nomet", AFT)), A(pp("nomet", EVE)), A(pp("no_imputed", AFT)), A(pp("no_imputed", EVE)),
         A(pp("balanced", AFT)), A(pp("balanced", EVE)),
         A(pp("y2022_25", AFT)), A(pp("y2022_25", EVE)), pz("y2022_25", AFT), pz("y2022_25", EVE),
         A(pp("y2019_21", AFT)), A(pp("y2019_21", EVE)), pz("y2019_21", AFT), pz("y2019_21", EVE),
         A(pp("screen_none", AFT)), A(pp("screen_none", EVE)),
         int(r.pooled("screen_strict", "Dhaka", PM, "Fri", AFT, "k")),
         A(pp("screen_strict", AFT)), A(pp("screen_strict", EVE)),
         A(r.pooled("screen_none", "national", "NOX", "Fri", AFT)), A(r.boot("national", "NOX", AFT)),
         A(r.pooled("screen_none", "Dhaka", "NOX", "Fri", AFT)), A(r.pooled("main", "Dhaka", "NOX", "Fri", AFT)),
         infl, max(moves)])

    # ---------------------------------------------------------------- 4.3 daily mean
    blh = r.blh_exact
    add("BLH and block shares", "In winter the median ERA5 boundary-layer height is",
        [blh("Winter", range(0, 6)), blh("Winter", range(6, 12)), blh("Winter", range(12, 18)),
         blh("Winter", range(18, 24)), blh("Monsoon", range(0, 6)),
         min(aft), max(aft), sh("Monsoon", range(12, 18)), min(nm), max(nm), min(eve), max(eve)])
    w = lambda v, rg, f: r.weighting(v, rg, f)
    rc = r.recon
    add("accounting and reconstruction", "The accounting of Eq.~\\eqref{eq:weight} predicts a daily change of",
        [w("main", "Dhaka", "predicted_daily_pct"), w("main", "Dhaka", "observed_daily_pct"),
         w("main", "national", "predicted_daily_pct"), w("main", "national", "observed_daily_pct"),
         int(rc("Dhaka", "all", "n_fridays")),
         rc("Dhaka", "all", "recon_pct"), rc("Dhaka", "all", "recon_lo"), rc("Dhaka", "all", "recon_hi"),
         rc("Dhaka", "all", "direct_pct"), rc("Dhaka", "all", "direct_lo"), rc("Dhaka", "all", "direct_hi"),
         rc("Dhaka", "all", "diff_pp"), rc("Dhaka", "all", "diff_lo"), rc("Dhaka", "all", "diff_hi"),
         rc("national", "all", "recon_pct"), rc("national", "all", "direct_pct"),
         w("dry", "Dhaka", "predicted_daily_pct"), w("dry", "Dhaka", "observed_daily_pct"),
         rc("Dhaka", "dry", "recon_pct"), rc("Dhaka", "dry", "direct_pct"),
         rc("Dhaka", "dry", "diff_pp"), rc("Dhaka", "dry", "diff_lo"), rc("Dhaka", "dry", "diff_hi")])
    mk = r.mechanism
    ev_int = [mk(t, rg, EVE) for t in ("vc_interaction_per_doubling", "blh_interaction_per_doubling")
              for rg in ("Dhaka", "national")]
    add("boundary-layer tests", "per doubling of the ventilation coefficient it",
        [mk("vc_interaction_per_doubling", "Dhaka", AFT), mk("vc_interaction_per_doubling", "Dhaka", AFT, "lo"),
         mk("vc_interaction_per_doubling", "Dhaka", AFT, "hi"),
         mk("vc_interaction_per_doubling", "national", AFT), mk("vc_interaction_per_doubling", "national", AFT, "lo"),
         mk("vc_interaction_per_doubling", "national", AFT, "hi"),
         mk("blh_interaction_per_doubling", "Dhaka", AFT), mk("blh_interaction_per_doubling", "Dhaka", AFT, "lo"),
         mk("blh_interaction_per_doubling", "Dhaka", AFT, "hi"),
         mk("blh_interaction_per_doubling", "national", AFT), mk("blh_interaction_per_doubling", "national", AFT, "lo"),
         mk("blh_interaction_per_doubling", "national", AFT, "hi"),
         mk("tercile_low_minus_high", "Dhaka", AFT), mk("tercile_low_minus_high", "Dhaka", AFT, "lo"),
         mk("tercile_low_minus_high", "Dhaka", AFT, "hi"),
         mk("tercile_low_minus_high", "national", AFT), mk("tercile_low_minus_high", "national", AFT, "lo"),
         mk("tercile_low_minus_high", "national", AFT, "hi"),
         min(ev_int), max(ev_int),
         A(mk("friday_with_lagged_met", "Dhaka", AFT)), A(mk("friday_with_lagged_met", "Dhaka", EVE)),
         mk("carryover_rho", "Dhaka", AFT), A(mk("friday_short_run_carryover", "Dhaka", AFT)),
         A(mk("friday_short_run_carryover", "Dhaka", EVE))])

    # ---------------------------------------------------------------- 4.4 whole-city
    loo_e = r.eidc[(r.eidc.outcome == PM) & r.eidc.analysis.str.startswith("loo_")].b_pct
    city08 = r.traj("city", range(0, 9))
    corr_pre = r.traj("corridor", range(-10, 0))
    add("Eid (text)", "Over the Eid windows, pooled Dhaka-region",
        [A(r.eid("main")), int(r.eid("event_boot", field="n_events")), r.eid("event_boot", field="lo_pct"),
         r.eid("event_boot", field="hi_pct"), loo_e.max(), loo_e.min(), r.eid("fitr"), r.eid("adha"),
         r.eid("excl_2020_2021"), A(r.eid("treat_0_4")), A(r.eid("treat_0_8")), A(r.eid("comp_clean")),
         A(r.eid("fitr_post_only")), r.eid("fitr_post_only", field="lo_pct"), r.eid("fitr_post_only", field="hi_pct"),
         A(float(r.traj("city", [-1]).iloc[0])), A(city08.max()), A(city08.min()),
         A(float(r.traj("city", [10]).iloc[0]))])
    add("Eid heterogeneity and tracers", "The station effects are heterogeneous ($I^2=$",
        [r.eid("main", field="I2"), r.event_station("Eid", "Narsingdi"), corr_pre.min(), corr_pre.max(),
         A(r.tracer_ev("Eid", "TROPOMI NO2", "domain", "b_pct")), A(r.tracer_ev("Eid", "NOX", "Dhaka", "b_pct")),
         A(r.tracer_ev("Eid", "PM25", "national", "b_pct"))])
    lh = r.event_hours("Lockdown")
    add("lockdown (text)", "The lockdown lowered pooled \\pmtf{} by",
        [A(r.lk("main")), r.lk("main", field="lo_pct"), r.lk("main", field="hi_pct"), A(lh.max()), A(lh.min()),
         r.lk("placebo_2019"), r.lk("placebo_2022"), r.lk("placebo_2023"), r.lk("placebo_2024"),
         r.lk("main_with_placebo_spread", field="lo_pct"), r.lk("main_with_placebo_spread", field="hi_pct"),
         r.lk("woy"), r.lk("nomet"), r.lk("excl_later")])

    # ---------------------------------------------------------------- 4.5 sub-index
    add("sub-index by shock", "In index points, a Friday changes the daily sub-index by",
        [r.bootaqi("Dhaka", "Fri"), r.bootaqi("Dhaka", "Fri", "lo_b"), r.bootaqi("Dhaka", "Fri", "hi_b"),
         r.bootaqi("Dhaka", "Sat"), r.bootaqi("Dhaka", "Sat", "lo_b"), r.bootaqi("Dhaka", "Sat", "hi_b"),
         float(r.aqi.loc["Friday", "mean_aqi"]),
         r.bootaqi("national", "Fri"), r.bootaqi("national", "Fri", "lo_b"), r.bootaqi("national", "Fri", "hi_b"),
         A(r.eid("main", "AQI")), r.eid("event_boot", "AQI", "lo"), r.eid("event_boot", "AQI", "hi"),
         A(r.lk("main", "AQI")), r.lk("main_with_placebo_spread", "AQI", "lo"),
         r.lk("main_with_placebo_spread", "AQI", "hi")])
    add("scenarios (text)", "working-day baseline of",
        [sc("baseline", "mean_pm"), sc("baseline", "mean_aqi"), sc("baseline", "days150"),
         A(sc("rest_day", "d_mean_aqi")), sc("rest_day", "d_mean_aqi_lo"), sc("rest_day", "d_mean_aqi_hi"),
         A(sc("rest_day", "d_days150")), sc("rest_day", "d_days150_lo"), sc("rest_day", "d_days150_hi"),
         A(sc("rest_day", "d_days200")),
         A(sc("rest_day_calendar", "d_mean_aqi")), sc("rest_day_calendar", "d_mean_aqi_lo"),
         sc("rest_day_calendar", "d_mean_aqi_hi"),
         A(sc("cut_allday", "d_mean_aqi")), sc("cut_allday", "d_mean_aqi_lo"), sc("cut_allday", "d_mean_aqi_hi"),
         A(sc("cut_day_only", "d_mean_aqi")), A(sc("cut_night_only", "d_mean_aqi")), night_gain])

    # ---------------------------------------------------------------- discussion
    add("discussion 5.1", "On the weekly rest day, \\pmtf{} in Dhaka is",
        [A(r.boot("Dhaka", PM, AFT)), A(r.boot("Dhaka", PM, EVE)),
         min(sh(s, range(0, 12)) for s in dry), max(sh(s, range(0, 12)) for s in dry), min(aft), max(aft),
         A(w("main", "Dhaka", "predicted_daily_pct")), A(rc("Dhaka", "all", "diff_pp"))])
    add("discussion mixing height", "In winter the median mixing height falls from",
        [blh("Winter", [15]), blh("Winter", [18])])
    add("discussion rest-night NOx", "the rest night sees lower \\nox, by",
        [A(r.boot("national", "NOX", NIGHT, "Sat"))])
    sat_dh = r.hourly("Dhaka", PM, "Sat", range(0, 8))
    add("discussion rest night", "On the rest night in Dhaka, \\pmtf{} falls by",
        [-sat_dh.mean(), -r.hourly("Dhaka", "NOX", "Sat", range(0, 6)).max(),
         -r.hourly("Dhaka", "NOX", "Sat", range(0, 7)).min(),
         A(r.boot("national", PM, NIGHT, "Sat")), A(r.boot("national", "NOX", NIGHT, "Sat")),
         A(r.boot("national", PM, AFT)), A(r.boot("national", "NOX", AFT))])
    tr4 = [r.boot("national", t, b) for t in ("NOX", "CO") for b in (AFT, EVE)]
    add("discussion tracers over the day", "pooled nationally, fall by",
        [A(r.boot("national", "NOX", DAY)), A(r.boot("national", "CO", DAY)), A(max(tr4)), A(min(tr4))])
    add("discussion Eid", "traffic, lowered daily \\pmtf{} at the Dhaka-region monitors by", [A(r.eid("main"))])
    add("transferability", "the national daily mean falls more (",
        [A(r.boot("national", PM, DAY)), A(r.boot("Dhaka", PM, DAY))])
    add("limitations I2", "effects for the whole-city shocks are heterogeneous",
        [r.eid("main", field="I2"), r.lk("main", field="I2")])

    # ---------------------------------------------------------------- conclusions
    add("conclusions 1", "\\pmtf{} follows from midday, by",
        [A(r.boot("Dhaka", PM, AFT)), A(r.boot("Dhaka", PM, EVE)), A(r.boot("national", PM, EVE))])
    add("conclusions 2", "In Dhaka the daily mean changes by",
        [A(r.boot("Dhaka", PM, DAY)), A(r.bootaqi("Dhaka", "Fri"))])
    add("conclusions 3", "around the clock, lowered daily \\pmtf{} by",
        [A(r.eid("main")), A(r.eid("main", "AQI"))])
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
    add("S3 BTV level", "The host\nlisting settles its identity", [btv, btv / dmed, dmed])
    eid = pd.read_csv(os.path.join(ROOT, "Traffic", "data_availability", "02_event_calendars", "eid_dates.csv"),
                      parse_dates=["date"])
    ns = q[q.station == "Narsingdi"].set_index("time").PM25
    cov = []
    for d in eid.date:
        if d < ns.index.min():
            continue
        w = pd.date_range(d - pd.Timedelta(days=1), d + pd.Timedelta(days=3, hours=23), freq="h")
        cov.append((int(ns.reindex(w).notna().sum()), len(w)))
    ev = r.eev[(r.eev.station == "Narsingdi") & (r.eev.tracer == "PM25")].pct
    add("S Eid Narsingdi", "which turns the Narsingdi effect positive (",
        [r.event_station("Eid", "Narsingdi"), 100 * sum(a for a, _ in cov) / sum(b for _, b in cov), len(cov),
         min(100 * a / b for a, b in cov), max(100 * a / b for a, b in cov), int((ev > 0).sum()), len(ev)])
    return C


HIGHLIGHT_LINES = 5


def highlight_values(r):
    """The numbers each line of highlights.txt must carry, in order."""
    nm = [r.block_share(s, range(0, 12)) for s in ("Winter", "Pre-monsoon", "Post-monsoon", "Monsoon")]
    k = int(r.boot("Dhaka", "PM25", "daily", field="k") + r.boot("national", "PM25", "daily", field="k"))
    return [[k],
            [-r.boot("Dhaka", "PM25", "afternoon 12-17"), -r.boot("Dhaka", "PM25", "evening 18-23")],
            [min(nm), max(nm)],
            [-r.eid("main")],
            [-float(r.scen.loc["rest_day", "d_mean_aqi"])]]


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
        want = highlight_values(r)
        lines = open(hl_path, encoding="utf-8").read().splitlines()
        for i, (line, exp) in enumerate(zip(lines, want)):
            nums = re.findall(r"\d+\.\d+|\d+", line.replace("PM2.5", ""))
            ok = len(nums) == len(exp) and all(close(e, float(n), dec(n)) for e, n in zip(exp, nums))
            if not ok:
                fails.append("HIGHLIGHT line %d: printed %s, derived %s" %
                             (i + 1, nums, ", ".join("%.4g" % e for e in exp)))
        if len(lines) > HIGHLIGHT_LINES or any(len(x) > 85 for x in lines):
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
