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
TR = {"PM25": "PM$_{2.5}$", "NOX": "NO$_x$", "CO": "CO", "TROPOMI NO2": "TROPOMI NO$_2$"}


def s(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return ("%+." + str(nd) + "f") % v


def m(x):
    return str(x).replace("-", "$-$").replace("+", "$+$") if x != "--" else x


def ci(p, lo, hi):
    return "%s {\\scriptsize[%s, %s]}" % (m(s(p)), m(s(lo)), m(s(hi)))


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
              r"Share of 2019--2024 hours with a valid value after quality control, percent. 2021 is absent "
              r"from CAMS-BD, so no station can exceed about 83 percent.",
              "tab:s-coverage", ["Station", "PM$_{2.5}$", "PM$_{10}$", "NO$_2$", "NO$_x$", "CO"], rows)


def screening():
    v = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    f = pd.read_csv(os.path.join(OUT, "cams_flatline_check.csv"))
    cv = f.pivot(index="station", columns="tracer", values="daily_cv30")
    rows = []
    for r in v.itertuples():
        nox_cv = cv.loc[r.station, "NOX"] if r.station in cv.index and "NOX" in cv else np.nan
        co_cv = cv.loc[r.station, "CO"] if r.station in cv.index and "CO" in cv else np.nan
        use = []
        if r.daily_r_embassy >= 0.6 and abs(r.best_lag_h) <= 1:
            use.append("PM$_{2.5}$")
        if r.NOX_rho_tropomi >= 0.2 and not (nox_cv < 0.15):
            use.append("NO$_x$")
        if r.CO_rho_tropomi >= 0.2 and not (co_cv < 0.15):
            use.append("CO")
        rows.append([r.station, "%s--%s" % (str(r.first)[:4], str(r.last)[:4]), "%.2f" % r.daily_r_embassy,
                     "%.2f" % r.ratio_to_embassy, "%+d" % r.best_lag_h,
                     m(s(r.NOX_rho_tropomi, 2)), "%.2f" % nox_cv, m(s(r.CO_rho_tropomi, 2)), "%.2f" % co_cv,
                     ", ".join(use) or "none"])
    others = sorted(set(f.station) - set(v.station))
    for st in others:
        nox_cv = cv.loc[st, "NOX"] if "NOX" in cv and st in cv.index else np.nan
        co_cv = cv.loc[st, "CO"] if "CO" in cv and st in cv.index else np.nan
        use = ["PM$_{2.5}$"] + (["NO$_x$"] if np.isfinite(nox_cv) and nox_cv >= 0.15 else []) + \
              (["CO"] if np.isfinite(co_cv) and co_cv >= 0.15 else [])
        rows.append([st + "$^{*}$", "--", "--", "--", "--", "--", "%.2f" % nox_cv if np.isfinite(nox_cv) else "--",
                     "--", "%.2f" % co_cv if np.isfinite(co_cv) else "--", ", ".join(use)])
    longtable("supp_screening.tex", "L{2.9cm} C{1.4cm} C{1.0cm} C{1.0cm} C{0.9cm} C{1.2cm} C{1.0cm} C{1.2cm} C{1.0cm} L{2.2cm}",
              r"Station screening. Dhaka-region stations: daily PM$_{2.5}$ correlation (log) and median "
              r"ratio with the US Embassy monitor; the hourly lag of best correlation with the Embassy; "
              r"the deseasonalized daily Spearman correlation of each gas analyzer with TROPOMI NO$_2$ "
              r"($\rho$); and the median 30-day coefficient of variation of daily means (CV), which "
              r"flags flat-lined analyzers. Inclusion: PM$_{2.5}$ $r\geq0.6$ and lag within one hour; "
              r"gases $\rho\geq0.2$ (Dhaka region only) and CV $\geq0.15$ (all stations). "
              r"$^{*}$Stations outside the Dhaka region, which have no reference monitor; only the "
              r"flat-line rule applies to them.",
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


def thusat():
    r = pd.read_csv(os.path.join(OUT, "robust_pooled.csv"))
    r = r[(r.variant == "main") & r.day.isin(["Thu", "Sat"])]
    rows = []
    for (day, reg, tr), g in r.groupby(["day", "region", "tracer"]):
        g = g.set_index("block")
        rows.append([{"Thu": "Thursday", "Sat": "Saturday"}[day], reg.replace("national", "National"), TR[tr]] +
                    [ci(g.loc[b, "pct"], g.loc[b, "lo"], g.loc[b, "hi"]) if b in g.index else "--" for b in BLOCKS])
    longtable("supp_thusat.tex", "L{1.6cm} L{1.4cm} L{1.0cm}" + " C{2.15cm}" * 5,
              r"Pooled Thursday and Saturday effects by diurnal block, percent against Sunday to Wednesday "
              r"(random-effects pooling; 95\% interval).", "tab:s-thusat",
              ["Day", "Region", "Tracer"] + BLAB, rows, size=r"\footnotesize")


def robust():
    r = pd.read_csv(os.path.join(OUT, "robust_pooled.csv"))
    b = pd.read_csv(os.path.join(OUT, "robust_bootstrap.csv"))
    names = {"main": "Main (random effects)", "cluster": "Week-clustered errors", "nomet": "No meteorology",
             "y2019_21": "2019--2021 only", "y2022_25": "2022--2025 only", "dry": "Dry season (Nov--Apr)",
             "wet": "Wet season (May--Oct)", "noembassy": "Without the Embassy",
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
            lab = names.get(v, "Leave out " + v[4:])
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
              r"0.85 instead of 0.6 (the Embassy and four CAMS stations). National \pmtf{} is not screened, "
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
    order = ["Friday", "Saturday", "Eid", "Lockdown", "Strike days"]
    rows = []
    for shock in order:
        for _, r in a[a.shock == shock].sort_values(["outcome", "region"]).iterrows():
            rows.append([shock, TR.get(r.outcome, r.outcome), r.region.replace("national", "National"),
                         int(r.k), ci(r.pct, r.lo, r.hi)])
    # The national PM2.5 lockdown estimate is withheld (tropomi_events.py); the caption gives
    # the level shift that makes it unreliable, computed here from the QC'd hourly table.
    q = pd.read_csv(os.path.join(HERE, "..", "data", "cams_hourly_qc.csv.gz"),
                    usecols=["station", "time", "PM25"], parse_dates=["time"])
    jf = lambda st, yr: q[(q.station == st) & (q.time.dt.year == yr) & q.time.dt.month.isin([1, 2])].PM25.mean()
    shift = (jf("Agrabad CDA Chattogram", 2019), jf("Agrabad CDA Chattogram", 2020), jf("BARC", 2019), jf("BARC", 2020))
    longtable("supp_activity.tex", "L{2.0cm} L{2.6cm} L{1.6cm} C{0.8cm} C{3.4cm}",
              r"Daily-mean response to each shock in the present designs (Section 3 of the main text): "
              r"TROPOMI NO$_2$ over the domain, "
              r"and PM$_{2.5}$, NO$_x$ and CO pooled over screened stations; percent with 95\% interval. "
              r"Gas analyzers are not compared across years, so no gas lockdown estimate is given. Nor "
              r"is a national PM$_{2.5}$ lockdown estimate: the design reads March--May 2020 against each "
              r"station's own 2020 level, the national stations have no reference monitor against which a "
              r"level shift could be detected, and the two with 2020 data, both in Chattogram, show one "
              + (r"(at Agrabad, January--February PM$_{2.5}$ averaged %.1f\,$\mu$g\,m$^{-3}$ in 2019 and "
                 r"%.1f in 2020, while at BARC in Dhaka it rose from %.1f to %.1f)." % shift),
              "tab:s-activity", ["Shock", "Outcome", "Region", "$k$", "Change (\\%)"], rows)


if __name__ == "__main__":
    for f in (qc, coverage, screening, station, thusat, robust, vent, eid_event, mobility, activity):
        f()
    print("wrote", sorted(x for x in os.listdir(TAB) if x.startswith("supp_")))
