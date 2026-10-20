"""Write every supplementary table as LaTeX from the result files (never by hand).

  supp_qc.tex          S1  QC rule counts              out/cams_qc_report.csv
  supp_coverage.tex    S2  valid-hour coverage 2019-24  out/cams_coverage_2019_2024.csv
  supp_screening.tex   S3  station screening + flat-line check
                           out/cams_validation.csv, out/cams_flatline_check.csv
  supp_station.tex     S4  station Friday PM2.5 effects by block   out/robust_station_estimates.csv
  supp_thusat.tex      S5  pooled Thursday and Saturday effects    out/robust_pooled.csv
  supp_robust.tex      S6  robustness variants, Friday PM2.5       out/robust_pooled.csv, robust_bootstrap.csv
  supp_vent.tex        S7  Friday effect by ventilation tercile    out/friday_by_block_vc.csv
  supp_eid_event.tex   S8  Eid effect by event and station         out/eid_by_event.csv
  supp_mobility.tex    S9  mobility by event                       out/mobility_shocks.csv
  supp_activity.tex    S10 shock responses, present designs        out/shock_activity.csv
Usage: python -I make_supp_tables.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
TAB = os.path.join(HERE, "..", "paper", "tables")
os.makedirs(TAB, exist_ok=True)
BLOCKS = ["night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23", "daily"]
BLAB = ["Night", "Morning", "Afternoon", "Evening", "Daily"]
TR = {"PM25": "PM$_{2.5}$", "NOX": "NO$_x$", "CO": "CO", "TROPOMI NO2": "TROPOMI NO$_2$",
      "TROPOMI NO2 any coverage": "TROPOMI NO$_2$, every observed day"}


def s(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return ("%+." + str(nd) + "f") % v


def m(x):
    return str(x).replace("-", "$-$").replace("+", "$+$") if x != "--" else x


def ci(p, lo, hi):
    return "%s {\\scriptsize[%s, %s]}" % (m(s(p)), m(s(lo)), m(s(hi)))


def ci2(p, lo, hi):
    return "%s {\\scriptsize[%s, %s]}" % (m(s(p, 2)), m(s(lo, 2)), m(s(hi, 2)))


def longtable(name, spec, caption, label, header, rows, size=r"\small"):
    h = " & ".join(r"\textbf{%s}" % c for c in header) + r" \\"
    lines = [r"\begingroup" + size, r"\begin{longtable}{%s}" % spec,
             r"\caption{%s}\label{%s}\\" % (caption, label), r"\toprule", h, r"\midrule\endfirsthead",
             r"\toprule", h, r"\midrule\endhead", r"\bottomrule\endfoot"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\end{longtable}", r"\endgroup"]
    open(os.path.join(TAB, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def qc():
    q = pd.read_csv(os.path.join(OUT, "cams_qc_report.csv"))
    rows = [(r.rule.replace(">", "$>$"), "{:,}".format(int(r.n))) for r in q.itertuples()]
    longtable("supp_qc.tex", "L{9cm} R{3cm}".replace("R{3cm}", r">{\raggedleft\arraybackslash}p{3cm}"),
              r"Quality-control rules applied to the CAMS-BD hourly record and the number of station-hours "
              r"each removed or set missing, all years. Q0 timestamps that could not be parsed; Q1 values "
              r"outside physical limits; Q2 runs of six or more identical consecutive hourly values; Q3 hours "
              r"with PM$_{2.5}$ above 1.1 times PM$_{10}$; Q4 duplicate station-hours.",
              "tab:s-qc", ["Rule", "Station-hours"], rows)


def coverage():
    c = pd.read_csv(os.path.join(OUT, "cams_coverage_2019_2024.csv"))
    cols = ["PM25_hours", "PM10_hours", "NO2_hours", "NOX_hours", "CO_hours"]
    rows = [[r["station"]] + ["%.0f" % r[k] for k in cols] for _, r in c.iterrows()]
    longtable("supp_coverage.tex", "L{4.2cm}" + " C{1.8cm}" * 5,
              r"Share of 2019--2024 hours with a valid value after quality control, percent; by year in "
              r"Table~\ref{tab:s-coverage-year}.",
              "tab:s-coverage", ["Station", "PM$_{2.5}$", "PM$_{10}$", "NO$_2$", "NO$_x$", "CO"], rows)


def screening():
    import sys
    sys.path.insert(0, HERE)
    from screens import gas_allowed, screens
    ok_pm, gas_ok, flat = screens()
    v = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    f = pd.read_csv(os.path.join(OUT, "cams_flatline_check.csv"))
    cv = f.pivot(index="station", columns="tracer", values="daily_cv30")
    fmt = lambda st, g: ("%.2f" % cv.loc[st, g] + ("$^{\\dagger}$" if (st, g) in flat else "")) \
        if st in cv.index and g in cv and np.isfinite(cv.loc[st, g]) else "--"
    rows = []
    for r in v.itertuples():
        use = (["PM$_{2.5}$"] if r.station in ok_pm else []) + \
              [lab for g, lab in (("NOX", "NO$_x$"), ("CO", "CO")) if gas_allowed(r.station, g, "Dhaka", gas_ok, flat)]
        rows.append([r.station, "%s--%s" % (str(r.first)[:4], str(r.last)[:4]), "%.2f" % r.daily_r_embassy,
                     "%.2f" % r.ratio_to_embassy, "%+d" % r.best_lag_h,
                     m(s(r.NOX_rho_tropomi, 2)), fmt(r.station, "NOX"), m(s(r.CO_rho_tropomi, 2)), fmt(r.station, "CO"),
                     ", ".join(use) or "none"])
    for st in sorted(set(f.station) - set(v.station)):
        use = ["PM$_{2.5}$"] + [lab for g, lab in (("NOX", "NO$_x$"), ("CO", "CO"))
                                if st in cv.index and g in cv and np.isfinite(cv.loc[st, g])
                                and gas_allowed(st, g, "national", gas_ok, flat)]
        rows.append([st + "$^{*}$", "--", "--", "--", "--", "--", fmt(st, "NOX"), "--", fmt(st, "CO"), ", ".join(use)])
    longtable("supp_screening.tex", "L{2.9cm} C{1.4cm} C{1.0cm} C{1.0cm} C{0.9cm} C{1.2cm} C{1.1cm} C{1.2cm} C{1.1cm} L{2.2cm}",
              r"Station screening. Dhaka-region stations: daily PM$_{2.5}$ correlation (log) and median "
              r"ratio with the US Embassy monitor; the hourly lag of best correlation with the Embassy; "
              r"the deseasonalized daily Spearman correlation of each gas analyzer with TROPOMI NO$_2$ "
              r"($\rho$); and the median 30-day coefficient of variation of daily means (CV). Inclusion: "
              r"PM$_{2.5}$ $r\geq0.6$ and lag within one hour; gases $\rho\geq0.2$ (Dhaka region only) and not "
              r"failing the suitability screen (all stations). $^{\dagger}$Failing the suitability screen: CV below 0.15 together with no co-variation "
              r"with the co-located PM$_{2.5}$ (Table~\ref{tab:s-flatline}). Correlation with the Embassy shows that a station "
              r"tracks the reference day to day; it does not establish equal accuracy or calibration. "
              r"$^{*}$Stations outside the Dhaka region, which have no reference monitor; only the fault rule "
              r"applies to them.",
              "tab:s-screening", ["Station", "Record", "$r$", "Ratio", "Lag", "NO$_x$ $\\rho$", "NO$_x$ CV",
                                  "CO $\\rho$", "CO CV", "Used for"], rows, size=r"\footnotesize")


def station():
    e = pd.read_csv(os.path.join(OUT, "robust_station_estimates.csv"))
    e = e[(e.variant == "main") & (e.day == "Fri") & (e.tracer == "PM25")]
    rows = []
    for (reg, st), g in e.groupby(["region", "station"]):
        g = g.set_index("block")
        cells = []
        for b in BLOCKS:
            if b in g.index:
                bb, se = g.loc[b, "b"], g.loc[b, "se"]
                f = lambda v: 100 * (np.exp(v) - 1)
                cells.append(ci(f(bb), f(bb - 1.96 * se), f(bb + 1.96 * se)))
            else:
                cells.append("--")
        n = int(g.loc["daily", "n"]) if "daily" in g.index else ""
        rows.append([st, reg.replace("national", "National")] + cells + [n])
    longtable("supp_station.tex", "L{2.8cm} L{1.4cm}" + " C{2.0cm}" * 5 + " C{0.9cm}",
              r"Friday effect on PM$_{2.5}$ by station and diurnal block, percent against Sunday to "
              r"Wednesday (95\% interval, Newey--West). $n$: days in the daily-mean regression.",
              "tab:s-station", ["Station", "Region"] + BLAB + ["$n$"], rows, size=r"\footnotesize")


def robust():
    r = pd.read_csv(os.path.join(OUT, "robust_pooled.csv"))
    b = pd.read_csv(os.path.join(OUT, "robust_bootstrap.csv"))
    names = {"main": "Main (random effects)", "cluster": "Week-clustered errors", "nomet": "No meteorology",
             "y2019_21": "2019--2021 only", "y2022_25": "2022--2025 only", "dry": "Dry season (Nov--Apr)",
             "wet": "Wet season (May--Oct)", "noembassy": "Without the Embassy",
             "no_imputed": "Without the gap-filled months",
             "no_pandemic": "Without 26 Mar 2020 -- Dec 2021", "balanced": "Balanced sample of dates",
             "screen_none": "No station screens", "screen_strict": "Stricter screens"}
    rows = []
    for reg in ("Dhaka", "national"):
        q = r[(r.region == reg) & (r.tracer == "PM25") & (r.day == "Fri")]
        bb = b[(b.region == reg) & (b.tracer == "PM25")].set_index("block")
        rows.append([r"\multicolumn{6}{l}{\textit{%s}}" % ("Dhaka region" if reg == "Dhaka" else "National")])
        rows[-1] = rows[-1] + []
        rows.append(["Joint-date bootstrap"] + [ci(bb.loc[x, "pct"], bb.loc[x, "lo_pct"], bb.loc[x, "hi_pct"])
                                                for x in BLOCKS])
        variants = list(names) + sorted(v for v in q.variant.unique() if v.startswith("loo_"))
        for v in variants:
            g = q[q.variant == v].set_index("block")
            if g.empty or (reg == "national" and v.startswith("screen")):
                continue        # national PM2.5 has no reference screen, so these equal Main
            lab = names.get(v, "Leave out " + v[4:].replace("USEmbassy", "US Embassy"))
            rows.append([lab] + ["%s {\\scriptsize(%s)}" % (m(s(g.loc[x, "pct"])), m(s(g.loc[x, "z"], 1)))
                                 if x in g.index else "--" for x in BLOCKS])
    fixed = []
    for row in rows:
        fixed.append(row if len(row) == 6 else [row[0]] if row[0].startswith(r"\multicolumn") else row)
    lines = []
    longtable("supp_robust.tex", "L{3.8cm}" + " C{2.2cm}" * 5,
              r"Robustness of the pooled Friday PM$_{2.5}$ effect by diurnal block, percent. Bootstrap rows: "
              r"95\% interval from 400 joint-date moving-block resamples (14-day blocks). Other rows: "
              r"$z$-statistic in parentheses, random-effects pooling. No station screens: every Dhaka-region "
              r"CAMS station, adding Savar. Stricter screens: daily correlation with the Embassy of at least "
              r"0.85 instead of 0.6 (the Embassy and four CAMS stations). Without the gap-filled months: January--June "
              r"2024 dropped. Balanced sample: only dates on which at least 70 percent of the region's stations "
              r"report. National \pmtf{} is not screened, "
              r"so these two rows are given for the Dhaka region only.", "tab:s-robust",
              ["Variant"] + BLAB, fixed, size=r"\footnotesize")


def vent():
    v = pd.read_csv(os.path.join(OUT, "friday_by_block_vc.csv"))
    rows = []
    for (reg, blk), g in v.groupby(["region", "block"]):
        g = g.set_index("vc_tercile")
        rows.append([reg.replace("national", "National"), blk.split()[0].capitalize()] +
                    [ci(g.loc[t, "pct"], g.loc[t, "lo"], g.loc[t, "hi"]) for t in ("low", "mid", "high")] +
                    ["%.0f / %.0f" % (g.loc["low", "vc_mean"], g.loc["high", "vc_mean"])])
    longtable("supp_vent.tex", "L{1.6cm} L{1.8cm} C{2.6cm} C{2.6cm} C{2.6cm} C{2.6cm}",
              r"Friday effect on PM$_{2.5}$ within each diurnal block by tercile of that block's ventilation "
              r"coefficient, percent (95\% interval, random-effects pooling). Last column: mean ventilation "
              r"coefficient (m$^2$\,s$^{-1}$) of the lowest and highest tercile.", "tab:s-vent",
              ["Region", "Block", "Low", "Middle", "High", "VC low / high"], rows, size=r"\footnotesize")


def eid_event():
    e = pd.read_csv(os.path.join(OUT, "eid_by_event.csv"))
    e = e[e.tracer == "PM25"]
    events = sorted(e.eid.unique())
    rows = []
    for st, g in e.groupby("station"):
        g = g.set_index("eid")
        rows.append([st] + [m(s(g.loc[x, "pct"], 0)) if x in g.index else "--" for x in events])
    longtable("supp_eid_event.tex", "L{2.0cm}" + " C{1.0cm}" * len(events),
              r"Eid effect on PM$_{2.5}$ by event and station, percent, days $-1$ to $+3$ against the "
              r"event's own surrounding days, from a stacked regression with shared meteorology and "
              r"day-of-week effects.", "tab:s-eid",
              ["Station"] + [x[2:].replace("-", "--", 0) for x in events], rows, size=r"\scriptsize")


def mobility():
    mo = pd.read_csv(os.path.join(OUT, "mobility_shocks.csv"))
    rows = [[r.shock.replace("..", "--").replace("_", " "), int(r.n_days)] +
            [m(s(r[c])) for c in ("transit", "workplaces", "retail", "grocery", "residential")]
            for _, r in mo.iterrows()]
    longtable("supp_mobility.tex", "L{4.4cm} C{1.0cm}" + " C{1.8cm}" * 5,
              r"Google Community Mobility Reports, Bangladesh, by shock: percent change against the "
              r"same-weekday baseline (lockdown) or against the surrounding days (each Eid, strike days).",
              "tab:s-mobility",
              ["Shock", "Days", "Transit", "Workplaces", "Retail", "Grocery", "Residential"], rows)


def activity():
    a = pd.read_csv(os.path.join(OUT, "shock_activity.csv"))
    # dependence-aware intervals: the joint-date bootstrap for the weekly rows, the event
    # bootstrap for Eid and the placebo-widened interval for the lockdown NO2 column
    B = pd.read_csv(os.path.join(OUT, "robust_bootstrap_days.csv"))
    B = B[B.block == "daily"].set_index(["day", "region", "tracer"])
    T = pd.read_csv(os.path.join(OUT, "events_tracers.csv")).set_index(["shock", "outcome", "region"])
    E = pd.read_csv(os.path.join(OUT, "eid_checks.csv"))
    e = E[(E.analysis == "event_boot") & (E.outcome == "PM25")].iloc[0]
    L = pd.read_csv(os.path.join(OUT, "lockdown_checks.csv"))
    lk = L[(L.analysis == "main_with_placebo_spread") & (L.outcome == "PM25")].iloc[0]

    def dep(r):
        day = {"Friday": "Fri", "Saturday": "Sat"}.get(r.shock)
        if day and (day, r.region, r.outcome) in B.index:
            x = B.loc[(day, r.region, r.outcome)]
            return "[%s, %s]" % (m(s(x.lo_pct)), m(s(x.hi_pct)))
        if (r.shock, r.outcome, r.region) in T.index:
            x = T.loc[(r.shock, r.outcome, r.region)]
            return "[%s, %s]" % (m(s(x.lo_pct)), m(s(x.hi_pct)))
        if (r.shock, r.outcome, r.region) == ("Eid", "PM25", "Dhaka"):
            return "[%s, %s]" % (m(s(e.lo_pct)), m(s(e.hi_pct)))
        if (r.shock, r.outcome, r.region) == ("Lockdown", "PM25", "Dhaka"):
            return "[%s, %s]" % (m(s(lk.lo_pct)), m(s(lk.hi_pct)))
        return "--"

    order = ["Friday", "Saturday", "Eid", "Lockdown", "Strike days"]
    rows = []
    for shock in order:
        for _, r in a[a.shock == shock].sort_values(["outcome", "region"]).iterrows():
            rows.append([shock, TR.get(r.outcome, r.outcome), r.region.replace("national", "National"),
                         int(r.k), ci(r.pct, r.lo, r.hi), dep(r)])
    # The national PM2.5 lockdown estimate is withheld (tropomi_events.py); the caption gives
    # the level shift that makes it unreliable, computed here from the QC'd hourly table.
    q = pd.read_csv(os.path.join(HERE, "..", "data", "cams_hourly_qc.csv.gz"),
                    usecols=["station", "time", "PM25"], parse_dates=["time"])
    jf = lambda st, yr: q[(q.station == st) & (q.time.dt.year == yr) & q.time.dt.month.isin([1, 2])].PM25.mean()
    shift = (jf("Agrabad CDA Chattogram", 2019), jf("Agrabad CDA Chattogram", 2020), jf("BARC", 2019), jf("BARC", 2020))
    longtable("supp_activity.tex", "L{2.0cm} L{3.4cm} L{1.5cm} C{0.6cm} C{3.3cm} C{2.4cm}",
              r"Daily-mean response to each shock in the designs of Section~3 of the main text: "
              r"TROPOMI NO$_2$ over the domain (days with at least half of the domain observed, and every "
              r"observed day), and PM$_{2.5}$, NO$_x$ and CO pooled over screened stations; percent with 95\% "
              r"interval, first treating stations as independent (Newey--West, random effects), then "
              r"dependence-aware: the joint-date bootstrap for Friday and Saturday, the event bootstrap over the "
              r"11 usable events for Eid, and for the lockdown the interval widened by the spread of four "
              r"pseudo-lockdowns. TROPOMI is one series, so its weekly and strike-day intervals are Newey--West "
              r"only, as are all strike-day intervals. "
              r"Gas analyzers are not compared across years, so no gas lockdown estimate is given. Nor "
              r"is a national PM$_{2.5}$ lockdown estimate: the design reads March--May 2020 against each "
              r"station's own 2020 level, the national stations have no reference monitor against which a "
              r"level shift could be detected, and the two with 2020 data, both in Chattogram, show one "
              + (r"(at Agrabad, January--February PM$_{2.5}$ averaged %.1f\,$\mu$g\,m$^{-3}$ in 2019 and "
                 r"%.1f in 2020, while at BARC in Dhaka it rose from %.1f to %.1f)." % shift),
              "tab:s-activity", ["Shock", "Outcome", "Region", "$k$", "Change (\\%)", "Dependence-aware"], rows,
              size=r"\footnotesize")


def qc_station():
    Q = pd.read_csv(os.path.join(OUT, "qc_station_year.csv"))
    piv = Q.pivot_table(index="station", columns="rule", values="n", aggfunc="sum", fill_value=0)
    gas = Q[Q.pollutant.isin(["NOX", "CO"])].pivot_table(index="station", columns="pollutant", values="n",
                                                          aggfunc="sum", fill_value=0)
    cols = [c for c in ["Q1 range", "Q2 stuck run", "Q3 PM2.5 > 1.1 PM10", "Q4 duplicate"] if c in piv]
    rows = []
    for st in piv.index:
        rows.append([st] + ["{:,}".format(int(piv.loc[st, c])) for c in cols]
                    + ["{:,}".format(int(gas.loc[st, g])) if st in gas.index and g in gas else "0" for g in ("NOX", "CO")])
    longtable("supp_qc_station.tex", "L{3.4cm}" + " C{1.7cm}" * (len(cols) + 2),
              r"Station-hours removed by each quality-control rule, all years and pollutants, by station; the last "
              r"two columns give the NO$_x$ and CO hours among them. The rules are those of Table~\ref{tab:s-qc}. Units as "
              r"supplied: PM in $\mu$g\,m$^{-3}$, NO$_x$ in ppb, CO in ppm. Zero gas readings are kept (they "
              r"are valid at low concentrations) and drop out of the log-scale analyses; of duplicate "
              r"station-hours the first is kept. CAMS-BD carries no calibration or detection-limit metadata.",
              "tab:s-qc-station", ["Station"] + [c.replace(">", "$>$") for c in cols] + ["of which NO$_x$", "of which CO"],
              rows, size=r"\footnotesize")


def coverage_year():
    C = pd.read_csv(os.path.join(OUT, "coverage_station_year.csv"))
    rows = []
    for st, g in C.groupby("station"):
        g = g.set_index("year")
        for p, lab in (("PM25", "PM$_{2.5}$"), ("NOX", "NO$_x$"), ("CO", "CO")):
            rows.append([st if p == "PM25" else "", lab] + ["%d" % round(g.loc[y, p]) if y in g.index else "0"
                                                           for y in range(2019, 2025)])
    longtable("supp_coverage_year.tex", "L{3.4cm} L{1.2cm}" + " C{1.0cm}" * 6,
              r"Valid hours by station, pollutant and year, percent of the year, after quality control.",
              "tab:s-coverage-year", ["Station", "Tracer"] + [str(y) for y in range(2019, 2025)], rows,
              size=r"\footnotesize")


def missingness():
    M = pd.read_csv(os.path.join(OUT, "missingness_dow.csv"))
    rows = []
    for (st, p), g in M.groupby(["station", "pollutant"], sort=True):
        g = g.set_index("hours")
        rows.append([st, TR.get(p, p), "%.1f" % (100 * g.loc["all", "valid_sun_wed"])]
                    + [m(s(g.loc[h, "diff_pp"])) if h in g.index else "--"
                       for h in ("all", "night 00-05", "afternoon 12-17", "evening 18-23")])
    longtable("supp_missing.tex", "L{3.2cm} L{1.2cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm}",
              r"Missing data by day of week, 2019--2024: the share of valid hours on Sunday to Wednesday, and "
              r"the Friday share minus it (percentage points), for all hours and three diurnal blocks. A "
              r"Friday-specific gap in the record would show here.", "tab:s-missing",
              ["Station", "Tracer", "Valid, Sun--Wed (\\%)", "Fri $-$ Sun--Wed", "Night", "Afternoon", "Evening"],
              rows, size=r"\footnotesize")


def flatline():
    D = pd.read_csv(os.path.join(OUT, "flatline_diagnostics.csv"))
    rows = []
    for r in D.sort_values(["flat", "cv30"], ascending=[False, True]).itertuples():
        rows.append([r.station, TR.get(r.pollutant, r.pollutant), "%.3f" % r.cv30, "%.2f" % r.diurnal_range,
                     m("%+.2f" % r.r_daily_pm25), "%.3f" % r.repeat_share, "excluded" if r.flat else "kept"])
    longtable("supp_flatline.tex", "L{3.2cm} L{1.1cm} C{1.4cm} C{1.6cm} C{1.8cm} C{1.6cm} C{1.6cm}",
              r"Fault diagnostics for every gas analyzer, 2019--2024: the median 30-day coefficient of variation "
              r"of daily means (CV), the range of the mean hourly profile relative to its mean, the daily log "
              r"correlation with the co-located PM$_{2.5}$, and the share of hours equal to the previous hour. "
              r"An analyzer fails the empirical suitability screen, and is excluded, when it both barely varies "
              r"(CV $<$ 0.15) and does not co-vary with PM$_{2.5}$ ($r<0.1$); neither condition alone shows a "
              r"fault, and the tracer results under a behaviour-only screen (CV alone) are in "
              r"Table~\ref{tab:s-gas-screens}.", "tab:s-flatline",
              ["Station", "Tracer", "CV", "Diurnal range", "$r$ with PM$_{2.5}$", "Repeats", "Outcome"],
              rows, size=r"\footnotesize")


def bootdays():
    B = pd.read_csv(os.path.join(OUT, "robust_bootstrap_days.csv"))
    rows = []
    for day, lab in (("Thu", "Thursday"), ("Sat", "Saturday")):
        for (reg, tr), g in B[(B.day == day) & B.tracer.isin(["PM25", "NOX", "CO"])].groupby(["region", "tracer"]):
            g = g.set_index("block")
            rows.append([lab, reg.replace("national", "National"), TR[tr]]
                        + [ci(g.loc[b, "pct"], g.loc[b, "lo_pct"], g.loc[b, "hi_pct"]) for b in BLOCKS])
    longtable("supp_thusat.tex", "L{1.6cm} L{1.4cm} L{1.1cm}" + " C{2.1cm}" * 5,
              r"Pooled Thursday and Saturday contrasts by diurnal block, percent against Sunday to Wednesday, "
              r"with 95\% intervals from the joint-date moving-block bootstrap (the same draws as the Friday "
              r"results).", "tab:s-thusat", ["Day", "Region", "Tracer"] + BLAB, rows, size=r"\scriptsize")


def bh():
    F = pd.read_csv(os.path.join(OUT, "robust_bh_bootstrap.csv")).sort_values("p_boot")
    rows = [[{"Fri": "Friday", "Sat": "Saturday"}[r.day], r.region.replace("national", "National"), TR[r.tracer],
             r.block.split(" ")[0].capitalize(), m(s(r.pct)), m(s(r.z_boot, 2)), "%.4f" % r.p_boot, "%.4f" % r.q_boot]
            for r in F.itertuples()]
    longtable("supp_bh.tex", "L{1.5cm} L{1.4cm} L{1.1cm} L{1.6cm} C{1.3cm} C{1.3cm} C{1.4cm} C{1.4cm}",
              r"The multiple-testing family: every pooled Friday and Saturday contrast of PM$_{2.5}$, NO$_x$ and CO, "
              r"in both regions, for the four diurnal blocks and the daily mean (%d tests). Each $p$-value comes "
              r"from the joint-date bootstrap ($z$ = estimate / bootstrap standard error), and $q$ is the "
              r"Benjamini--Hochberg adjustment over the whole family." % len(F), "tab:s-bh",
              ["Day", "Region", "Tracer", "Block", "\\%", "$z$", "$p$", "$q$"], rows, size=r"\scriptsize")


def blocklen():
    BL = pd.read_csv(os.path.join(OUT, "robust_bootstrap_blocklen.csv"))
    rows = []
    for (reg, blen), g in BL.groupby(["region", "block_days"]):
        g = g.set_index("block")
        rows.append([reg.replace("national", "National"), str(int(blen))]
                    + [ci(g.loc[b, "pct"], g.loc[b, "lo_pct"], g.loc[b, "hi_pct"]) for b in BLOCKS])
    longtable("supp_blocklen.tex", "L{1.6cm} C{1.4cm}" + " C{2.2cm}" * 5,
              r"Friday PM$_{2.5}$ contrasts with joint-date bootstrap intervals for three block lengths (days).",
              "tab:s-blocklen", ["Region", "Block (days)"] + BLAB, rows, size=r"\scriptsize")


def eid_checks():
    E = pd.read_csv(os.path.join(OUT, "eid_checks.csv"))
    lab = {"main": "Main (stations independent)", "event_boot": "Main, event bootstrap",
           "fitr": "Eid-ul-Fitr only", "adha": "Eid-ul-Adha only", "treat_0_4": "Treatment days 0 to $+4$",
           "treat_0_8": "Treatment days 0 to $+8$", "comp_clean": "Comparison days $-30$..$-8$, $+12$..$+30$",
           "treat_0_4_comp_clean": "Both of the above", "excl_2020_2021": "Without 2020 and 2021",
           "fitr_post_only": "Eid-ul-Fitr, comparison after Eid only"}
    rows = []
    for a, g in E.groupby("analysis", sort=False):
        pmr = g[g.outcome == "PM25"].iloc[0]
        aq = g[g.outcome == "AQI"].iloc[0]
        name = lab.get(a, "Without " + a[4:] if a.startswith("loo_") else a)
        rows.append([name, str(int(pmr.k)), str(int(pmr.n_events)), ci(pmr.b_pct, pmr.lo_pct, pmr.hi_pct),
                     ci(aq.b, aq.lo, aq.hi)])
    longtable("supp_eid_checks.tex", "L{6.0cm} C{0.7cm} C{1.1cm} C{3.0cm} C{3.0cm}",
              r"Checks on the Eid estimate (daily mean, Dhaka region): percent change in PM$_{2.5}$ and change in "
              r"the PM$_{2.5}$ AQI sub-index (points), 95\% intervals. Eid-ul-Fitr 2020 fell inside the lockdown "
              r"and is not usable, leaving 11 events. $k$: stations.", "tab:s-eid-checks",
              ["Analysis", "$k$", "Events", "PM$_{2.5}$ (\\%)", "Sub-index (points)"], rows, size=r"\footnotesize")


def lockdown_checks():
    L = pd.read_csv(os.path.join(OUT, "lockdown_checks.csv"))
    lab = {"main": "Main: year and month effects", "woy": "Week-of-year harmonics instead of months",
           "excl_later": "Without 31 May 2020 -- 31 Dec 2021", "nomet": "Without meteorology",
           "main_with_placebo_spread": "Main, interval widened by the placebo spread"}
    rows = []
    for a, g in L.groupby("analysis", sort=False):
        pmr = g[g.outcome == "PM25"].iloc[0]
        aq = g[g.outcome == "AQI"].iloc[0]
        name = lab.get(a, "Placebo window, " + a[-4:] if a.startswith("placebo_") else a)
        rows.append([name, str(int(pmr.k)), ci(pmr.b_pct, pmr.lo_pct, pmr.hi_pct), ci(aq.b, aq.lo, aq.hi)])
    longtable("supp_lockdown_checks.tex", "L{6.4cm} C{0.7cm} C{3.0cm} C{3.0cm}",
              r"Checks on the lockdown estimate (daily mean, Dhaka region). Placebo rows apply the same design to "
              r"26 March -- 30 May of other years with the true lockdown removed (2021 is excluded because a "
              r"nationwide lockdown began on 14 April 2021). The last row adds the variance of the placebo "
              r"estimates to the sampling variance.", "tab:s-lockdown-checks",
              ["Analysis", "$k$", "PM$_{2.5}$ (\\%)", "Sub-index (points)"], rows, size=r"\footnotesize")


def reconstruct():
    R = pd.read_csv(os.path.join(OUT, "reconstruct.csv"))
    samp = {"all": "All", "dry": "Dry", "wet": "Wet", "complete": "Complete days"}
    rows = [[r.region.replace("national", "National"), samp[r.season], r.scale.capitalize(), str(int(r.k)),
             str(int(r.n_fridays)), ci(r.recon_pct, r.recon_lo, r.recon_hi), ci(r.direct_pct, r.direct_lo, r.direct_hi),
             ci(r.diff_pp, r.diff_lo, r.diff_hi)] for r in R.itertuples()]
    longtable("supp_reconstruct.tex", "L{1.4cm} L{1.5cm} L{1.5cm} C{0.4cm} C{1.0cm} C{2.4cm} C{2.4cm} C{2.4cm}",
              r"The daily Friday change in the equal-block mean PM$_{2.5}$ (the mean of the four block means) "
              r"reconstructed from the four block contrasts, against the change estimated directly for the same daily "
              r"outcome on the same dates, percent, and their difference (percentage points); 95\% intervals from the "
              r"joint-date bootstrap carried through every step. Log: block and daily models fitted to log "
              r"concentrations; the reconstruction divides each Friday's observed block means by $e^{\beta_b}$, a "
              r"residual-preserving multiplicative counterfactual, and the direct value $e^{\beta}-1$ is a "
              r"geometric-mean contrast that, under the same construction, is also the ratio of the Friday total to "
              r"its counterfactual. Arithmetic: the same designs fitted by Poisson pseudo-maximum likelihood to "
              r"concentrations, so $e^{\beta}$ is a ratio of conditional arithmetic means and the fitted Friday total "
              r"equals the observed one; block and daily models then estimate the same ratio of Friday to working-day "
              r"totals under different mean models. Complete days: dates with all 24 hours valid, on which the "
              r"equal-block mean is the mean of all hours.", "tab:s-reconstruct",
              ["Region", "Sample", "Scale", "$k$", "Fridays", "Reconstructed", "Direct", "Difference"], rows,
              size=r"\scriptsize")


def decompose():
    D = pd.read_csv(os.path.join(OUT, "decompose.csv"))
    samp = {"all": "All", "dry": "Dry", "wet": "Wet", "complete": "Complete days"}
    rows = [[r.region.replace("national", "National"), samp[r.season], r.scale.capitalize(),
             ci(r.ae_pct, r.ae_lo, r.ae_hi), ci(r.partial_pct, r.partial_lo, r.partial_hi),
             ci(r.equal_pct, r.equal_lo, r.equal_hi), ci(r.conc_pct, r.conc_lo, r.conc_hi),
             ci2(r.cancel_pp, r.cancel_lo, r.cancel_hi), ci2(r.weight_pp, r.weight_lo, r.weight_hi)]
            for r in D.itertuples()]
    longtable("supp_decompose.tex", "L{1.2cm} L{1.1cm} L{1.4cm}" + " C{1.75cm}" * 6,
              r"What makes the daily Friday change small, step by step (percent; differences in percentage "
              r"points; 95\% joint-date bootstrap intervals). Affected: the mean of the afternoon and evening "
              r"contrasts. Partial day: the same with the night and morning contrasts set to zero and equal weights "
              r"for the four blocks (half the affected value). Equal weights: all four block contrasts, equal "
              r"weights. Concentration weights: each block weighted by its share of the counterfactual Friday total "
              r"(the reconstruction of Table~\ref{tab:s-reconstruct}). Cancellation: equal weights minus partial "
              r"day, the effect of the night and morning contrasts. Weighting: concentration minus equal weights, the "
              r"effect of the unequal baseline concentrations of the blocks. Scales as in "
              r"Table~\ref{tab:s-reconstruct}.", "tab:s-decompose",
              ["Region", "Sample", "Scale", "Affected", "Partial day", "Equal weights", "Conc.\\ weights",
               "Cancel\\-lation", "Weighting"], rows, size=r"\scriptsize")


def mechanism():
    M = pd.read_csv(os.path.join(OUT, "mechanism.csv"))
    lab = {"vc_interaction_per_doubling": "Friday $\\times$ ventilation, per doubling, log scale ($100\\delta\\ln 2$)",
           "vc_change_per_doubling": "Friday $\\times$ ventilation, per doubling, exact (pp)",
           "blh_interaction_per_doubling": "Friday $\\times$ boundary-layer height, per doubling, log scale ($100\\delta\\ln 2$)",
           "blh_change_per_doubling": "Friday $\\times$ boundary-layer height, per doubling, exact (pp)",
           "tercile_low_minus_high": "Lowest minus highest ventilation tercile (pp)",
           "friday_with_lagged_met": "Friday contrast, lagged ventilation added (\\%)",
           "friday_short_run_carryover": "Friday contrast, previous block's concentration added (\\%)",
           "carryover_rho": "Carry-over coefficient $\\rho$"}
    rows = []
    for r in M.itertuples():
        fmt = (lambda v: "%.2f" % v) if r.test == "carryover_rho" else (lambda v: s(v))
        val = m(fmt(r.est)) + (" {\\scriptsize[%s, %s]}" % (m(fmt(r.lo)), m(fmt(r.hi))) if np.isfinite(r.lo) else "")
        rows.append([lab.get(r.test, r.test), r.region.replace("national", "National"),
                     r.block.split(" ")[0].capitalize(), str(int(r.k)), val])
    longtable("supp_mechanism.tex", "L{7.0cm} L{1.4cm} L{1.6cm} C{0.6cm} C{3.2cm}",
              r"Direct tests of the boundary-layer interpretation, PM$_{2.5}$. Interactions: the Friday "
              r"indicator times the station-centred log ventilation coefficient (whose main effect is in the "
              r"controls) or the centred log boundary-layer height (added with its own main effect); per doubling of "
              r"the variable, as $100\delta\ln 2$ on the log scale (approximately percentage points) and exactly on "
              r"the response scale, $100e^{\beta}(2^{\delta}-1)$ at the centred reference. A positive value means a "
              r"smaller Friday reduction in better-ventilated conditions. Interactions and the tercile difference: "
              r"fixed-weight pooling with joint-date bootstrap intervals. Lagged-meteorology and carry-over models: "
              r"random-effects pooling; the carry-over model is a sensitivity check, not a model of a sustained "
              r"change.", "tab:s-mechanism", ["Test", "Region", "Block", "$k$", "Estimate"], rows,
              size=r"\footnotesize")


def gapfill():
    G = pd.read_csv(os.path.join(OUT, "gapfill_errors.csv"))
    rows = [[r.subset.replace("<", "$<$"), "{:,}".format(int(r.n)), "%.0f" % r.median_obs, "%.1f" % r.rmse, "%.1f" % r.mae,
             m(s(r.bias))] for r in G.itertuples()]
    longtable("supp_gapfill.tex", "L{4.4cm} C{1.4cm} C{2.0cm} C{1.4cm} C{1.4cm} C{1.4cm}",
              r"Error of the boundary-layer gap fill in blocked cross-validation (eight held-out 181-day blocks), "
              r"metres, by season, hour of day and depth of the observed layer, pooled over all held-out hours (the "
              r"fold-mean RMSE in the main text averages the eight folds).", "tab:s-gapfill",
              ["Subset", "Hours", "Median observed", "RMSE", "MAE", "Bias"], rows, size=r"\footnotesize")


def aqi_checks():
    A = pd.read_csv(os.path.join(OUT, "aqi_dominance.csv"))
    rows = [[r.source.replace("CAMS ", ""), r.comparison.replace("PM2.5", "PM$_{2.5}$").replace("PM10", "PM$_{10}$"),
             str(int(r.days)), "%.1f" % (100 * r.share_other_higher), "%.1f" % (100 * r.share_other_higher_above150)]
            for r in A.itertuples()]
    longtable("supp_aqi_checks.tex", "L{3.6cm} L{3.4cm} C{1.2cm} C{2.4cm} C{2.6cm}",
              r"How often another sub-index exceeds the PM$_{2.5}$ sub-index on the same day: PM$_{10}$ at each CAMS "
              r"station (2019--2024) and the 8-hour ozone sub-index at the US Embassy (2023--2025), percent of days, "
              r"and percent of days on which that happens with the larger sub-index above 150.", "tab:s-aqi",
              ["Station", "Comparison", "Days", "Other higher (\\%)", "and above 150 (\\%)"], rows,
              size=r"\footnotesize")


def aqi_annual():
    A = pd.read_csv(os.path.join(OUT, "aqi_annualization.csv"))
    rows = [[r.season, str(int(r.calendar_days)), str(int(r.valid_days)), "%.1f" % (100 * r.coverage),
             "%.1f" % (100 * r.share_gt150), "%.1f" % (100 * r.share_gt200), "%.1f" % (100 * r.share_gt300)]
            for r in A.itertuples()]
    a = A.iloc[0]
    longtable("supp_aqi_annual.tex", "L{2.6cm} C{1.6cm} C{1.4cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm}",
              r"US Embassy daily PM$_{2.5}$ sub-index by season, January 2019 -- March 2025: calendar days, days "
              r"with at least 18 valid hours, their share, and the share of valid days above 150, 200 and 300. "
              r"Days per year above each threshold (share of all valid days $\times$ 365.25): %.1f, %.1f and %.1f; "
              r"with the four seasons weighted by their calendar length instead, %.1f, %.1f and %.1f. %d days "
              r"exceeded 325.4\,$\mu$g\,m$^{-3}$ and were set to 500."
              % (a.days_gt150_simple, a.days_gt200_simple, a.days_gt300_simple, a.days_gt150_balanced,
                 a.days_gt200_balanced, a.days_gt300_balanced, int(a.capped_days_total)),
              "tab:s-aqi-annual", ["Season", "Days", "Valid", "Coverage (\\%)", "$>150$ (\\%)", "$>200$ (\\%)",
                                   "$>300$ (\\%)"], rows, size=r"\footnotesize")


def gas_screens():
    P = pd.read_csv(os.path.join(OUT, "robust_pooled.csv"))
    lab = {"main": "Main screens", "screen_behaviour": "Behaviour only", "screen_none": "No screens",
           "screen_strict": "Stricter screens"}
    rows = []
    for reg in ("Dhaka", "national"):
        for tr in ("NOX", "CO"):
            for v in ("main", "screen_behaviour", "screen_none", "screen_strict"):
                g = P[(P.variant == v) & (P.region == reg) & (P.tracer == tr) & (P.day == "Fri")].set_index("block")
                if g.empty:
                    rows.append([reg.replace("national", "National"), TR[tr], lab[v], "0"] + ["--"] * 5)
                    continue
                rows.append([reg.replace("national", "National"), TR[tr], lab[v], str(int(g.k.max()))]
                            + [ci(g.loc[b, "pct"], g.loc[b, "lo"], g.loc[b, "hi"]) if b in g.index else "--"
                               for b in BLOCKS])
    longtable("supp_gas_screens.tex", "L{1.3cm} L{1.0cm} L{2.0cm} C{0.4cm}" + " C{1.9cm}" * 5,
              r"Friday NO$_x$ and CO contrasts under the main, behaviour-only, no and stricter screens, percent "
              r"(95\% interval, stations treated as independent). Behaviour only: an analyzer is dropped when its "
              r"daily means barely vary (CV below 0.15), without asking whether it co-varies with PM$_{2.5}$. No "
              r"screens: every analyzer, including those failing the suitability screen and, in the Dhaka region, "
              r"those that fail the TROPOMI comparison. Stricter: TROPOMI correlation of at least 0.3 in the Dhaka "
              r"region and every analyzer with a CV below 0.2 dropped, which leaves no Dhaka-region CO analyzer.",
              "tab:s-gas-screens", ["Region", "Tracer", "Screens", "$k$"] + BLAB, rows, size=r"\scriptsize")


def aqi_cap():
    A = pd.read_csv(os.path.join(OUT, "aqi_cap.csv"))
    rows = [[r.shock, r.version.replace("capped", "capped at 500"), str(int(r.k)), "{:,}".format(int(r.station_days)),
             str(int(r.days_above_325)), ci(r.b, r.lo, r.hi)] for r in A.itertuples()]
    longtable("supp_aqi_cap.tex", "L{1.8cm} L{3.2cm} C{0.5cm} C{1.8cm} C{1.8cm} C{3.2cm}",
              r"Sensitivity of the daily PM$_{2.5}$ sub-index effects to the treatment of days above "
              r"325.4\,$\mu$g\,m$^{-3}$, Dhaka region: set to 500 (the main convention), or with the top breakpoint "
              r"segment (301--500 over 225.5--325.4\,$\mu$g\,m$^{-3}$) extended linearly. Station-days in each "
              r"analysis sample and those above 325.4; change in index points with 95\% interval (Newey--West, "
              r"random effects).", "tab:s-aqi-cap",
              ["Shock", "Version", "$k$", "Station-days", "Above 325.4", "Change (points)"], rows,
              size=r"\footnotesize")


def clock():
    K = pd.read_csv(os.path.join(OUT, "clock_check.csv"))
    rows = []
    for r in K.itertuples():
        lag = lambda y: ("%+d {\\scriptsize(%.2f)}" % (getattr(r, "lag_" + y), getattr(r, "r_" + y))).replace(
            "-", "$-$").replace("+", "$+$") if np.isfinite(getattr(r, "lag_" + y)) else "--"
        rows.append([r.station, "{:,}".format(int(r.sr_hours)),
                     "%.2f" % r.sr_centroid_h if np.isfinite(r.sr_centroid_h) else "--",
                     lag("PM25"), lag("NOX"), lag("CO")])
    longtable("supp_clock.tex", "L{3.6cm} C{1.6cm} C{1.8cm} C{2.0cm} C{2.0cm} C{2.0cm}",
              r"Station clocks, 2019--2024. Solar radiation: hours with a value, and the radiation-weighted mean "
              r"hour of the day (local, hour-beginning; not given below 5,000 hours). Lags: for the Dhaka-region "
              r"stations, the hourly lag at which the within-day log anomalies of each pollutant correlate best "
              r"with those at Darussalam (positive: the station follows Darussalam), with that correlation. A "
              r"clock offset would move every instrument at a station together.", "tab:s-clock",
              ["Station", "Radiation hours", "Radiation centre (h)", "Lag PM$_{2.5}$", "Lag NO$_x$", "Lag CO"],
              rows, size=r"\footnotesize")


def strikes():
    T = pd.read_csv(os.path.join(OUT, "strike_local.csv"))
    rows = [[r.calendar, r.outcome.replace("NOX", "NO$_x$").replace("PM25", "PM$_{2.5}$").replace("NO2", "NO$_2$"),
             str(int(r.n_days)), ci(r.pct, r.lo, r.hi)] for r in T.itertuples()]
    longtable("supp_strike_local.tex", "L{3.6cm} L{4.0cm} C{1.4cm} C{3.2cm}",
              r"Strike days: all qualifying dates in Dhaka division, and only the dates with an event in Dhaka "
              r"district itself; percent change in the daily mean with 95\% interval.", "tab:s-strikes",
              ["Calendar", "Outcome", "Days", "Change (\\%)"], rows, size=r"\footnotesize")


RUNLAB = {"published": "Published", "B2000": "2,000 resamples", "seed2": "Seed 2", "seed3": "Seed 3",
          "seed4": "Seed 4", "seed5": "Seed 5"}


def hour_runs(hours):
    """[16, 21, 22, 23] -> '16, 21--23'; [] -> 'none'."""
    runs = []
    for h in sorted(hours):
        if runs and h == runs[-1][1] + 1:
            runs[-1][1] = h
        else:
            runs.append([h, h])
    return ", ".join("%02d" % a if a == b else "%02d--%02d" % (a, b) for a, b in runs) or "none"


def stability_weekly():
    W = pd.read_csv(os.path.join(OUT, "boot_stability_weekly.csv"))
    rows = []
    for reg in ("Dhaka", "national"):
        for run in RUNLAB:
            g = W[(W.region == reg) & (W.run == run)].set_index("block")
            r0 = g.iloc[0]
            row = [reg.replace("national", "National"), RUNLAB[run], "{:,}".format(int(r0.B))]
            for b in ("afternoon 12-17", "evening 18-23", "daily"):
                q = g.loc[b, "q_bh"]
                row += [ci(g.loc[b, "pct"], g.loc[b, "lo_pct"], g.loc[b, "hi_pct"]),
                        "$<$0.001" if q < 0.001 else "%.3f" % q]
            rows.append(row)
    seeds = W.groupby("run", sort=False).seed.first()
    longtable("supp_stability_weekly.tex", "L{1.2cm} L{2.2cm} C{0.7cm}" + " C{2.5cm} C{0.9cm}" * 3,
              r"Monte Carlo stability of the joint-date bootstrap: Friday PM$_{2.5}$ contrasts (percent, 95\% "
              r"percentile interval) and Benjamini--Hochberg $q$-values, recomputed with the published seed and "
              r"400 resamples (which reproduce the published values), with 2,000 resamples, and with four further "
              r"seeds of 400. Every resample gave a pooled estimate. Each $q$-value uses the run's $p$-values for "
              r"the ten Friday PM$_{2.5}$ tests and the published ones for the other 50 tests of the family. "
              r"Seeds: " + "; ".join("%s %d" % (RUNLAB[k].lower(), v) for k, v in seeds.items()) + ".",
              "tab:s-stability-weekly",
              ["Region", "Run", "$B$", "Afternoon", "$q$", "Evening", "$q$", "Daily", "$q$"], rows,
              size=r"\scriptsize\setlength{\tabcolsep}{3pt}")


def stability_hourly():
    T = pd.read_csv(os.path.join(OUT, "boot_stability_hourly.csv"))
    curves = [("Dhaka", "PM25"), ("national", "PM25"), ("Dhaka", "NOX"), ("national", "NOX"),
              ("Dhaka", "CO"), ("national", "CO"), ("Dhaka", "NOX_CO"), ("national", "NOX_CO")]
    lab = {"PM25": "PM$_{2.5}$", "NOX": "NO$_x$", "CO": "CO", "NOX_CO": "NO$_x$/CO"}
    rows = []
    for reg, y in curves:
        for run in RUNLAB:
            g = T[(T.region == reg) & (T.tracer == y) & (T.run == run)]
            if g.empty:
                continue
            out = {d: hour_runs(g[(g.day == d) & g.sim_excludes_zero].hour) for d in ("Thu", "Fri", "Sat")}
            rows.append([reg.replace("national", "National") + " " + lab[y], RUNLAB[run],
                         "%.2f" % g.crit.iloc[0], out["Thu"], out["Fri"], out["Sat"]])
    seeds = T.groupby("run", sort=False).seed.first()
    longtable("supp_stability_hourly.tex", "L{2.6cm} L{2.4cm} C{0.9cm} L{2.0cm} L{3.6cm} L{2.0cm}",
              r"Monte Carlo stability of the simultaneous 95\% bands for the 72-hour curves (Thursday 00~h to "
              r"Saturday 23~h): the critical value $c$ and the clock hours at which the band excludes zero, for the "
              r"published seed and 400 resamples (which reproduce the published bands), 2,000 resamples, and four "
              r"further seeds of 400. Every resample gave a pooled estimate at every hour. Seeds: "
              + "; ".join("%s %d" % (RUNLAB[k].lower(), v) for k, v in seeds.items()) + ".",
              "tab:s-stability-hourly", ["Curve", "Run", "$c$", "Thursday", "Friday", "Saturday"], rows,
              size=r"\scriptsize")


if __name__ == "__main__":
    for f in (qc, coverage, screening, station, robust, vent, eid_event, mobility, activity,
              qc_station, coverage_year, missingness, flatline, bootdays, bh, blocklen, eid_checks,
              lockdown_checks, reconstruct, decompose, mechanism, gapfill, aqi_checks, aqi_annual, aqi_cap,
              clock, gas_screens, strikes, stability_weekly, stability_hourly):
        f()
    print("wrote", sorted(x for x in os.listdir(TAB) if x.startswith("supp_")))
