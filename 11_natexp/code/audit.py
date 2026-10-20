"""Data audits: QC, coverage, missingness, analyzer faults, sub-index, gap fill, strike days.

  qc_station_year.csv      QC removals by rule, station, year and pollutant, re-applying the
                           rules of cams_qc.py to the parsed table (nothing is rewritten)
  coverage_station_year.csv  valid hours (%) by station, year and pollutant after QC
  missingness_dow.csv      share of valid hours by day of week and hour, Friday minus the
                           Sunday-Wednesday mean, per station and pollutant, 2019-2024
  flatline_diagnostics.csv per gas analyzer: median 30-day CV of daily means, diurnal range of
                           the mean hourly profile relative to its mean, share of hours equal
                           to the previous hour, and the correlation of daily means with the
                           co-located PM2.5, with the screening outcome
  aqi_dominance.csv        how often the PM2.5 sub-index is not the largest one that can be
                           checked: PM10 at the CAMS stations, ozone at the US Embassy
  aqi_annualization.csv    Embassy valid-day coverage by season, days capped at 500, and days
                           per year above 150, 200 and 300 with and without seasonal balancing
  gapfill_errors.csv       blocked cross-validation error of the boundary-layer gap fill by
                           season, by hour of day and for shallow layers (< 200 m)
  strike_local.csv         strike-day estimates with Dhaka-district dates only
Also writes the derived calendar Traffic/.../hartal_event_days_DHAKA_DISTRICT_DERIVED.csv
(dates and counts only, as ACLED's terms allow).
Usage: python -I audit.py <FYDP_D root>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_qc import RANGES, STATION, stuck_mask  # noqa: E402
from cams_shocks import calendars, embassy_hourly, met_daily, tropomi_daily  # noqa: E402
from events_hourly import aqi_pm25  # noqa: E402
from screens import flatline_table, screens  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
GASES = ("NOX", "CO")
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
          6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}
PM10_BP = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200),
           (355, 424, 201, 300), (425, 604, 301, 500)]
O3_8H_BP = [(0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
            (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300)]


def subindex(c, bps, trunc):
    c = np.floor(np.asarray(c, float) / trunc) * trunc
    out = np.full_like(c, np.nan)
    for lo, hi, ilo, ihi in bps:
        m = (c >= lo) & (c <= hi + trunc - 1e-12)
        out[m] = ilo + (ihi - ilo) / (hi - lo) * (np.minimum(c[m], hi) - lo)
    return out


def qc_by_station_year():
    df = pd.read_csv(os.path.join(DATA, "cams_hourly_raw.csv.gz"), low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="mixed")
    df = df[df["time"].notna()].copy()
    df["station"] = df["station"].map(lambda s: STATION.get(str(s).strip(), str(s).strip()))
    df["time"] = df["time"] - pd.Timedelta(hours=1)
    df = df.sort_values(["station", "time"])
    dup = df.duplicated(["station", "time"])
    df["year"] = df.time.dt.year
    rows = [dict(rule="Q4 duplicate", station=st, year=yr, pollutant="all", n=int(n))
            for (st, yr), n in df[dup].groupby(["station", "year"]).size().items()]
    df = df[~dup]
    for p in ("PM25", "PM10") + GASES:
        lo, hi = RANGES[p]
        bad = df[p].notna() & ((df[p] <= lo if p in ("PM25", "PM10") else df[p] < lo) | (df[p] > hi))
        for (st, yr), n in df[bad].groupby(["station", "year"]).size().items():
            rows.append(dict(rule="Q1 range", station=st, year=yr, pollutant=p, n=int(n)))
        df.loc[bad, p] = np.nan
        m = np.zeros(len(df), bool)
        for st, idx in df.groupby("station").indices.items():
            m[idx] = stuck_mask(df[p].iloc[idx])
        for (st, yr), n in df[m].groupby(["station", "year"]).size().items():
            rows.append(dict(rule="Q2 stuck run", station=st, year=yr, pollutant=p, n=int(n)))
        df.loc[m, p] = np.nan
    bad = df["PM25"].notna() & df["PM10"].notna() & (df["PM25"] > 1.1 * df["PM10"])
    for (st, yr), n in df[bad].groupby(["station", "year"]).size().items():
        rows.append(dict(rule="Q3 PM2.5 > 1.1 PM10", station=st, year=yr, pollutant="PM25+PM10", n=int(n)))
    zeros = {g: int(((df[g] == 0) & (df.time >= "2019-01-01")).sum()) for g in GASES}
    return pd.DataFrame(rows), zeros


def main(root):
    # ---------------------------------------------------------------- QC and coverage
    Q, zeros = qc_by_station_year()
    Q.to_csv(os.path.join(OUT, "qc_station_year.csv"), index=False)
    c = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    c = c[(c.time >= "2019-01-01") & (c.time < "2025-01-01")]
    c["year"] = c.time.dt.year
    hours = {y: (pd.Timestamp("%d-01-01" % (y + 1)) - pd.Timestamp("%d-01-01" % y)).days * 24
             for y in range(2019, 2025)}
    cov = c.groupby(["station", "year"])[["PM25"] + list(GASES)].count()
    cov = cov.div([hours[y] for y in cov.index.get_level_values("year")], axis=0) * 100
    cov.round(1).to_csv(os.path.join(OUT, "coverage_station_year.csv"))
    # ---------------------------------------------------------------- missingness
    c["dow"] = c.time.dt.dayofweek
    c["hour"] = c.time.dt.hour
    c["date"] = c.time.dt.normalize()
    rows = []
    for st, h in c.groupby("station"):
        days = pd.date_range(h.date.min(), h.date.max())
        grid = pd.MultiIndex.from_product([days, range(24)], names=["date", "hour"])
        for p in ("PM25",) + GASES:
            v = h.set_index(["date", "hour"])[p].notna()
            v = v[~v.index.duplicated()].reindex(grid, fill_value=False).reset_index(name="ok")
            v["dow"] = v.date.dt.dayofweek
            sh = v.groupby("dow").ok.mean()
            if sh.mean() < 0.2:
                continue
            for blk, hs in (("all", range(24)), ("night 00-05", range(0, 6)), ("afternoon 12-17", range(12, 18)),
                            ("evening 18-23", range(18, 24))):
                vv = v[v.hour.isin(list(hs))]
                s2 = vv.groupby("dow").ok.mean()
                rows.append(dict(station=st, pollutant=p, hours=blk, valid_fri=s2.get(4, np.nan),
                                 valid_sun_wed=s2.reindex([6, 0, 1, 2]).mean(),
                                 diff_pp=100 * (s2.get(4, np.nan) - s2.reindex([6, 0, 1, 2]).mean())))
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(OUT, "missingness_dow.csv"), index=False)
    # ---------------------------------------------------------------- flat-line diagnostics
    fl = flatline_table()
    ok_pm, gas_ok, flat = screens()
    drows = []
    for st, h in c.groupby("station"):
        dpm = h.groupby("date").PM25.mean()
        for p in GASES:
            x = h[["time", "date", "hour", p]].dropna()
            if len(x) < 2000:
                continue
            prof = x.groupby("hour")[p].mean()
            rep = (x[p].values[1:] == x[p].values[:-1]).mean()
            d = x.groupby("date")[p].mean()
            j = pd.concat([np.log(d.clip(lower=1e-3)), np.log(dpm)], axis=1, join="inner").dropna()
            cv = fl[(fl.station == st) & (fl.tracer == p)].daily_cv30
            drows.append(dict(station=st, pollutant=p, hours=len(x),
                              cv30=float(cv.iloc[0]) if len(cv) else np.nan,
                              diurnal_range=float((prof.max() - prof.min()) / prof.mean()),
                              repeat_share=float(rep), r_daily_pm25=float(j.corr().iloc[0, 1]),
                              flat=(st, p) in flat))
    pd.DataFrame(drows).to_csv(os.path.join(OUT, "flatline_diagnostics.csv"), index=False)
    # ---------------------------------------------------------------- AQI checks
    arows = []
    for st, h in c.groupby("station"):
        g = h.groupby("date")
        both = pd.DataFrame({"pm25": g.PM25.mean().where(g.PM25.count() >= 18),
                             "pm10": g.PM10.mean().where(g.PM10.count() >= 18)}).dropna()
        if len(both) < 200:
            continue
        a25 = aqi_pm25(both.pm25.values)
        a10 = subindex(both.pm10.values, PM10_BP, 1.0)
        arows.append(dict(source="CAMS " + st, comparison="PM10 vs PM2.5", days=len(both),
                          share_other_higher=float(np.mean(a10 > a25)),
                          share_other_higher_above150=float(np.mean((a10 > a25) & (np.maximum(a10, a25) > 150)))))
    import glob
    fs = sorted(glob.glob(os.path.join(root, "02_data", "raw", "ground", "Dhaka_OZONE_*_YTD.csv")))
    o = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    o = o.rename(columns={"Date (LT)": "time", "Raw Conc.": "o3", "QC Name": "qc"})
    o["time"] = pd.to_datetime(o["time"], format="%Y-%m-%d %I:%M %p")
    o = o[(o.qc == "Valid") & (o.o3 >= 0)].set_index("time").sort_index()
    o3 = o.o3.resample("h").mean()
    roll = o3.rolling(8, min_periods=6).mean()
    o3d = roll.groupby(roll.index.normalize()).max()
    emb = embassy_hourly(root)
    eg = emb.assign(date=emb.time.dt.normalize()).groupby("date").PM25
    ed = eg.mean().where(eg.count() >= 18).dropna()
    j = pd.concat([ed.rename("pm25"), o3d.rename("o3")], axis=1, join="inner").dropna()
    unit = str(o["Conc. Unit"].iloc[0]) if "Conc. Unit" in o else "PPB"
    scale = 1e-3 if "PPB" in unit.upper() else 1.0
    ao = subindex(j.o3.values * scale, O3_8H_BP, 0.001)
    ap = aqi_pm25(j.pm25.values)
    arows.append(dict(source="US Embassy", comparison="ozone 8-h vs PM2.5", days=len(j),
                      share_other_higher=float(np.nanmean(ao > ap)),
                      share_other_higher_above150=float(np.nanmean((ao > ap) & (np.fmax(ao, ap) > 150))),
                      o3_unit=unit))
    pd.DataFrame(arows).to_csv(os.path.join(OUT, "aqi_dominance.csv"), index=False)
    # annualization at the Embassy, 2019 to the end of the record
    ed19 = ed[ed.index >= "2019-01-01"]
    a = pd.Series(aqi_pm25(ed19.values), index=ed19.index)
    span = pd.date_range("2019-01-01", ed19.index.max())
    srows = []
    for sname in ("Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"):
        days = span[span.month.map(SEASON) == sname]
        have = a[a.index.month.map(SEASON) == sname]
        srows.append(dict(season=sname, calendar_days=len(days), valid_days=len(have),
                          coverage=len(have) / len(days),
                          **{"share_gt%d" % t: float((have > t).mean()) for t in (150, 200, 300)}))
    S = pd.DataFrame(srows)
    yrs = len(span) / 365.25
    tot = {}
    for t in (150, 200, 300):
        simple = float((a > t).sum() / (len(a) / 365.25))
        balanced = float(sum(r["share_gt%d" % t] * r["calendar_days"] for r in srows) / yrs)
        tot[t] = (simple, balanced)
    S["capped_days_total"] = int((ed19 > 325.4).sum())
    for t, (s1, s2) in tot.items():
        S["days_gt%d_simple" % t] = s1
        S["days_gt%d_balanced" % t] = s2
    S.to_csv(os.path.join(OUT, "aqi_annualization.csv"), index=False)
    # ---------------------------------------------------------------- gap-fill errors
    sys.path.insert(0, os.path.join(root, "02_data"))
    import gapfill
    raw = pd.read_csv(os.path.join(root, "02_data", "raw", "meteo_dhaka_hourly.csv"), parse_dates=["time"])
    P = gapfill.blocked_cv_predictions(raw)
    P["time"] = pd.to_datetime(P.time)
    P["err"] = P.pred - P.obs
    P["season"] = P.time.dt.month.map(SEASON)
    P["hour"] = P.time.dt.hour
    erows = []
    groups = [("all", P)] + [("season " + s, g) for s, g in P.groupby("season")] + \
        [("hours 00-05", P[P.hour < 6]), ("hours 12-17", P[(P.hour >= 12) & (P.hour < 18)]),
         ("hours 18-23", P[P.hour >= 18]), ("observed < 200 m", P[P.obs < 200]),
         ("observed < 100 m", P[P.obs < 100])]
    for lab, g in groups:
        erows.append(dict(subset=lab, n=len(g), rmse=float(np.sqrt((g.err ** 2).mean())),
                          mae=float(g.err.abs().mean()), bias=float(g.err.mean()),
                          median_obs=float(g.obs.median()),
                          mape_median=float((g.err.abs() / g.obs.clip(lower=10)).median())))
    pd.DataFrame(erows).to_csv(os.path.join(OUT, "gapfill_errors.csv"), index=False)
    # ---------------------------------------------------------------- local strike days
    cal = os.path.join(root, "Traffic", "data_availability", "02_event_calendars",
                       "hartal_event_days_DHAKA_DISTRICT_DERIVED.csv")
    acled = os.path.join(root, "Traffic", "gates", "out", "gate1_acled_events.json")
    if os.path.exists(acled):
        acl = pd.DataFrame(json.load(open(acled, encoding="utf-8")))
        acl["date"] = pd.to_datetime(acl.event_date)
        loc = acl[acl.admin2 == "Dhaka"].groupby("date").size().rename("n_events_dhaka_district")
        loc.reset_index().to_csv(cal, index=False)
    else:
        # the public deposit holds only the derived calendar, not the ACLED records
        loc = pd.read_csv(cal, parse_dates=["date"]).set_index("date").n_events_dhaka_district
    import tropomi_events as te
    metd, _ = met_daily(root)
    hart, eid, eid_days = calendars(root)
    local = set(loc.index)
    tr = tropomi_daily(root).loc["2019-01-01":"2025-12-31"]
    ly = lambda s: np.log(s.y.values)
    srows = []
    for lab, days in (("all strike days", hart), ("Dhaka district only", local)):
        s = te.frame(tr, metd, days, eid_days)
        r = te.strike_effect(s, ly)
        if r:
            srows.append(dict(calendar=lab, outcome="TROPOMI NO2", n_days=int(s.hart.sum()), b=r[0], se=r[1]))
        for st in ("Darussalam", "Gazipur", "Savar", "US Embassy", "BARC"):
            if st == "US Embassy":
                h = embassy_hourly(root)
                p = "PM25"
            else:
                h = c[c.station == st]
                p = "NOX" if st in ("Darussalam", "Gazipur", "Savar") else "PM25"
            g = h.assign(date=h.time.dt.normalize()).groupby("date")[p]
            d = g.mean().where(g.count() >= 18).dropna()
            s = te.frame(d, metd, days, eid_days)
            s = s[s.index >= "2019-01-01"]
            r = te.strike_effect(s, ly)
            if r:
                srows.append(dict(calendar=lab, outcome=p + " " + st, n_days=int(s.hart.sum()), b=r[0], se=r[1]))
    T = pd.DataFrame(srows)
    T["pct"] = 100 * (np.exp(T.b) - 1)
    T["lo"] = 100 * (np.exp(T.b - 1.96 * T.se) - 1)
    T["hi"] = 100 * (np.exp(T.b + 1.96 * T.se) - 1)
    T.to_csv(os.path.join(OUT, "strike_local.csv"), index=False)
    pd.set_option("display.width", 220)
    print("gas zeros since 2019:", zeros)
    print(M[M.hours == "all"].round(3).to_string(index=False))
    print(pd.DataFrame(drows).round(3).to_string(index=False))
    print(pd.DataFrame(arows).round(4).to_string(index=False))
    print(S.round(3).to_string(index=False))
    print(pd.DataFrame(erows).round(1).to_string(index=False))
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
