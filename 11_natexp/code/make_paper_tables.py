"""Write the data-bearing manuscript tables as LaTeX, straight from the result files, so no
number in them is transcribed by hand.

  paper/working/tables/tab_effects.tex     Table 3, from out/table3.csv
  paper/working/tables/tab_scenarios.tex   Table 6, from out/aqi_scenarios.csv and
                                   out/aqi_scenarios_stations.csv
  paper/working/tables/tab_stations.tex    Table 2, stations and PM2.5 coverage by year
  paper/working/tables/tab_sensitivity.tex Table 5, robustness of the Dhaka Friday contrast
  paper/working/tables/tab_screening.tex   Table S2, from out/cams_validation.csv
Usage: python -I make_paper_tables.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
TAB = os.path.join(HERE, "..", "paper", "working", "tables")
NUM = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
os.makedirs(TAB, exist_ok=True)


def tex(s):
    return str(s).replace("−", "$-$").replace("+", "$+$").replace("%", r"\%")


def num(v, nd=1, sign=True):
    s = ("%+." + str(nd) + "f") % v if sign else ("%." + str(nd) + "f") % v
    return s.replace("-", "$-$").replace("+", "$+$") if sign else s


def effects():
    T = pd.read_csv(os.path.join(OUT, "table3.csv"))
    units = ["night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23", "daily"]
    lines = [r"\begin{longtable}{L{1.4cm} L{1.3cm} L{1.0cm} C{1.8cm} C{1.8cm} C{1.8cm} C{1.8cm} C{1.8cm} C{1.8cm}}",
             r"\caption{Pooled contrasts by diurnal block and shock, percent (95\% interval), and the change "
             r"in the daily PM$_{2.5}$ AQI sub-index in index points. Friday and Saturday: against the same "
             r"block on Sunday to Wednesday, joint-date moving-block bootstrap (14-day blocks, 400 "
             r"replicates). Eid: against the surrounding days, with intervals from resampling the 11 usable "
             r"events. Lockdown: against the year, month and weather expectation; first the model interval, "
             r"then a placebo-based sensitivity interval that adds the variance of the same design's estimates "
             r"in pseudo-lockdown windows of four other years to the model variance (a sensitivity check "
             r"with four placebo years, not a calibrated confidence interval). The night block is "
             r"00--05\,h of the calendar day, so a Friday's night block is Thursday night."
             r"}\label{tab:effects}\\",
             r"\toprule",
             r"\textbf{Shock} & \textbf{Region} & \textbf{Tracer} & \textbf{Night} & \textbf{Morning} & "
             r"\textbf{Afternoon} & \textbf{Evening} & \textbf{Daily} & \textbf{AQI (points)} \\",
             r"\midrule\endfirsthead",
             r"\toprule",
             r"\textbf{Shock} & \textbf{Region} & \textbf{Tracer} & \textbf{Night} & \textbf{Morning} & "
             r"\textbf{Afternoon} & \textbf{Evening} & \textbf{Daily} & \textbf{AQI (points)} \\",
             r"\midrule\endhead",
             r"\bottomrule\endfoot"]
    for _, r in T.iterrows():
        lab = r.shock
        if r.shock == "Lockdown":
            lab += r"\newline {\scriptsize(%s)}" % ("model" if "model interval" in r.interval else "placebo-based")
        cells = [lab, r.region.replace("national", "National"),
                 {"PM2.5": r"PM$_{2.5}$", "NOx": r"NO$_x$"}.get(r.tracer, r.tracer)]
        cells += [tex(r[u]).replace(" [", r"\newline [") for u in units]
        a = r["daily AQI (points)"]
        cells.append(tex(a).replace(" [", r"\newline [") if isinstance(a, str) and a.strip() else "")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    lines = [r"\begingroup\footnotesize"] + lines + [r"\endgroup"]
    open(os.path.join(TAB, "tab_effects.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def scenarios():
    S = pd.read_csv(os.path.join(OUT, "aqi_scenarios.csv")).set_index("scenario")
    St = pd.read_csv(os.path.join(OUT, "aqi_scenarios_stations.csv"))
    rng = St.groupby("scenario").d_mean_aqi.agg(["min", "max"])
    names = [("rest_day", "Every day takes the rest-day profile (Friday morning to evening, then the night after Friday)"),
             ("rest_day_calendar", "The same with the calendar-Friday profile (Thursday night included)"),
             ("cut_allday", "Friday-afternoon-sized concentration cut at all 24 hours"),
             ("cut_day_only", "The same cut at 12--23\\,h only"),
             ("cut_night_only", "The same cut at 00--11\\,h only")]
    b = S.loc["baseline"]
    hdr = (r"\textbf{Scenario} & \textbf{PM$_{2.5}$ (\%)} & \textbf{Sub-index (points)} & "
           r"\textbf{Days $>$150} & \textbf{Days $>$200} & \textbf{Days $>$300} & \textbf{CAMS range} \\")
    lines = [r"\begin{longtable}{L{4.4cm} C{1.7cm} C{1.9cm} C{1.6cm} C{1.6cm} C{1.6cm} C{2.0cm}}",
             r"\caption{Illustrative scenarios: concentration reductions shaped like the observed weekly "
             r"pattern, applied to a working-day counterfactual of the US Embassy hourly record (Thursday, "
             r"Friday and Saturday hours converted to their Sunday--Wednesday equivalents; Eid, lockdown and "
             r"strike days left out), January 2019 to March 2025. Change in mean PM$_{2.5}$, in the mean daily "
             r"PM$_{2.5}$ AQI sub-index and in days per year above three category boundaries, with 95\% "
             r"intervals from the joint-date bootstrap draws of the block effects. These rescale "
             r"concentrations; they are not forecasts of a policy and not equivalent emissions cuts. "
             r"Baseline: mean PM$_{2.5}$ " + num(b.mean_pm, 1, False) + r"\,$\mu$g\,m$^{-3}$, mean sub-index "
             + num(b.mean_aqi, 1, False) + r", and " + num(b.days150, 0, False) + r", " + num(b.days200, 0, False)
             + r" and " + num(b.days300, 0, False) + r" days per year above 150, 200 and 300. Last column: "
             r"range of the sub-index change over the " + NUM[St.station.nunique()]
             + r" screened Dhaka CAMS stations, 2019--2024.}\label{tab:scenarios}\\",
             r"\toprule", hdr, r"\midrule\endfirsthead", r"\toprule", hdr, r"\midrule\endhead",
             r"\bottomrule\endfoot"]

    def iv(r, col):
        return num(r[col]) + r"\newline {\scriptsize[%s, %s]}" % (num(r[col + "_lo"]), num(r[col + "_hi"]))
    for key, lab in names:
        r = S.loc[key]
        cams = ("%s to %s" % (num(rng.loc[key, "min"]), num(rng.loc[key, "max"]))) if key in rng.index else "--"
        lines.append(" & ".join([lab, iv(r, "d_pm_pct"), iv(r, "d_mean_aqi"), iv(r, "d_days150"),
                                 iv(r, "d_days200"), iv(r, "d_days300"), cams]) + r" \\")
    lines.append(r"\end{longtable}")
    lines = [r"\begingroup\small"] + lines + [r"\endgroup"]
    open(os.path.join(TAB, "tab_scenarios.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def stations():
    """Compact station table: region, host, PM2.5 coverage by year, and what each is used for."""
    import sys
    sys.path.insert(0, HERE)
    from cams_shocks import embassy_hourly
    from screens import gas_allowed, screens
    cov = pd.read_csv(os.path.join(OUT, "coverage_station_year.csv"))
    loc = pd.read_csv(os.path.join(HERE, "..", "data", "station_locations.csv")).set_index("station")
    ok_pm, gas_ok, flat = screens()
    dh = ["BARC", "Darussalam", "DoE", "Savar", "Gazipur", "Narayanganj", "Narsingdi"]
    emb = embassy_hourly(os.path.join(HERE, "..", ".."))
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2025-01-01")].drop_duplicates("time")
    years = list(range(2019, 2025))
    hours = {y: (pd.Timestamp("%d-01-01" % (y + 1)) - pd.Timestamp("%d-01-01" % y)).days * 24 for y in years}
    ecov = emb.groupby(emb.time.dt.year).size()
    pos = lambda st: (r"%.3f\,N\newline %.3f\,E" % (loc.loc[st, "lat"], loc.loc[st, "lon"])) if st in loc.index else "--"
    rows = [["US Embassy", "Dhaka", loc.loc["US Embassy", "host"].split(",")[0], pos("US Embassy")]
            + ["%d" % round(100 * ecov.get(y, 0) / hours[y]) for y in years] + ["PM$_{2.5}$ (reference)"]]
    order = dh + sorted(s for s in cov.station.unique() if s not in dh)
    for st in order:
        c = cov[cov.station == st].set_index("year")
        reg = "Dhaka" if st in dh else "National"
        use = []
        if (reg == "National") or st in ok_pm:
            use.append("PM$_{2.5}$")
        for g, lab in (("NOX", "NO$_x$"), ("CO", "CO")):
            if gas_allowed(st, g, "Dhaka" if st in dh else "national", gas_ok, flat):
                use.append(lab)
        host = loc.loc[st, "host"].split(",")[0] if st in loc.index else ""
        rows.append([st, reg, host, pos(st)] + ["%d" % round(c.loc[y, "PM25"]) if y in c.index else "0" for y in years]
                    + [", ".join(use) if use else "none"])
    hdr = (r"\textbf{Station} & \textbf{Region} & \textbf{Host} & \textbf{Position} & "
           + " & ".join(r"\textbf{%d}" % y for y in years) + r" & \textbf{Used for} \\")
    lines = [r"\begin{longtable}{L{2.3cm} L{1.2cm} L{2.9cm} L{1.4cm} " + "C{0.6cm} " * 6 + r"L{2.2cm}}",
             r"\caption{Monitoring stations: host institution, position (World Air Quality Index listing; the "
             r"US Embassy from OpenStreetMap), valid PM$_{2.5}$ hours by year (percent of the year, after quality "
             r"control) and the analyses each enters. No siting classification of the CAMS stations is "
             r"published. Dhaka-region PM$_{2.5}$ is screened against the US Embassy reference; gas analyzers are "
             r"screened against TROPOMI NO$_2$ in the Dhaka region and for faults everywhere "
             r"(Section~\ref{sec:screening}). Exclusions by rule, station and year are in "
             r"Table~\ref{S-tab:s-qc-station}; coverage of the gases in Table~\ref{S-tab:s-coverage-year}."
             r"}\label{tab:stations}\\",
             r"\toprule", hdr, r"\midrule\endfirsthead", r"\toprule", hdr, r"\midrule\endhead", r"\bottomrule\endfoot"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines.append(r"\end{longtable}")
    lines = [r"\begingroup\footnotesize"] + lines + [r"\endgroup"]
    open(os.path.join(TAB, "tab_stations.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def sensitivity():
    """Main-paper summary of robustness and sensitivity for the Dhaka Friday PM2.5 contrast."""
    P = pd.read_csv(os.path.join(OUT, "robust_pooled.csv"))
    B = pd.read_csv(os.path.join(OUT, "robust_bootstrap_days.csv"))
    BL = pd.read_csv(os.path.join(OUT, "robust_bootstrap_blocklen.csv"))
    q = P[(P.region == "Dhaka") & (P.tracer == "PM25") & (P.day == "Fri")]
    blocks = ["afternoon 12-17", "evening 18-23", "daily"]

    def boot_row(lab, tab):
        g = tab.set_index("block")
        return [lab, str(int(g.k.max())) if "k" in g else "7"] + [
            num(g.loc[b, "pct"]) + r"\newline {\scriptsize[%s, %s]}" % (num(g.loc[b, "lo_pct"]), num(g.loc[b, "hi_pct"]))
            for b in blocks]

    rows = []
    fb = B[(B.day == "Fri") & (B.region == "Dhaka") & (B.tracer == "PM25")]
    rows.append(boot_row("Main, joint-date bootstrap (14-day blocks)", fb))
    for blen in (7, 28):
        rows.append(boot_row("Bootstrap, %d-day blocks" % blen,
                             BL[(BL.block_days == blen) & (BL.region == "Dhaka")]))
    labs = [("cluster", "Week-clustered errors"), ("nomet", "No meteorology"), ("y2019_21", "2019--2021 only"),
            ("y2022_25", "2022--2025 only"), ("dry", "Dry season (Nov--Apr)"), ("wet", "Wet season (May--Oct)"),
            ("no_imputed", "Without the gap-filled boundary-layer months"),
            ("no_pandemic", "Without 26 March 2020 -- December 2021 (pandemic restrictions)"),
            ("balanced", "Balanced dates ($\\geq$70\\% of stations)"), ("noembassy", "Without the Embassy"),
            ("screen_none", "No station screens"), ("screen_strict", "Stricter screens")]
    for v, lab in labs:
        g = q[q.variant == v].set_index("block")
        rows.append([lab, str(int(g.k.max()))] + ["%s {\\scriptsize($z$=%s)}" % (num(g.loc[b, "pct"]),
                                                                                  num(g.loc[b, "z"], 1))
                                                    for b in blocks])
    hdr = r"\textbf{Variant} & \textbf{$k$} & \textbf{Afternoon} & \textbf{Evening} & \textbf{Daily} \\"
    lines = [r"\begin{longtable}{L{6.0cm} C{0.6cm} C{2.6cm} C{2.6cm} C{2.6cm}}",
             r"\caption{Robustness of the Dhaka-region Friday PM$_{2.5}$ contrast, percent. Bootstrap rows: "
             r"95\% interval from 400 joint-date resamples with the block length shown. Other rows: "
             r"random-effects pooling, $z$-statistic in parentheses. $k$: stations. The full set, with the "
             r"national stations and each station left out, is in Table~\ref{S-tab:s-robust}; the gases under "
             r"each screen in Table~\ref{S-tab:s-gas-screens}.}\label{tab:sensitivity}\\",
             r"\toprule", hdr, r"\midrule\endfirsthead", r"\toprule", hdr, r"\midrule\endhead", r"\bottomrule\endfoot"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines.append(r"\end{longtable}")
    lines = [r"\begingroup\small"] + lines + [r"\endgroup"]
    open(os.path.join(TAB, "tab_sensitivity.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def screening():
    V = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    lines = [r"\begin{longtable}{L{2.2cm} C{1.8cm} C{1.6cm} C{1.4cm} C{1.4cm} C{1.6cm} C{1.6cm} C{2.4cm}}",
             r"\caption{Screening of the Dhaka-region CAMS stations. Daily PM$_{2.5}$ correlation "
             r"(log) and median ratio with the US Embassy monitor over the overlapping days; the lag in "
             r"hours at which hourly PM$_{2.5}$ correlates best with the Embassy record; and the "
             r"deseasonalized daily Spearman correlation of each gas analyzer with TROPOMI NO$_2$. "
             r"Inclusion rules: PM$_{2.5}$ correlation $\geq$0.6 and best lag within one hour; a gas "
             r"correlation $\geq$0.2.}\label{tab:screening}\\",
             r"\toprule",
             r"\textbf{Station} & \textbf{Record} & \textbf{$r$ with Embassy} & \textbf{Ratio} & "
             r"\textbf{Best lag (h)} & \textbf{NO$_x$ vs TROPOMI} & \textbf{CO vs TROPOMI} & \textbf{Used for} \\",
             r"\midrule\endfirsthead",
             r"\toprule",
             r"\textbf{Station} & \textbf{Record} & \textbf{$r$ with Embassy} & \textbf{Ratio} & "
             r"\textbf{Best lag (h)} & \textbf{NO$_x$ vs TROPOMI} & \textbf{CO vs TROPOMI} & \textbf{Used for} \\",
             r"\midrule\endhead",
             r"\bottomrule\endfoot"]
    for _, r in V.iterrows():
        use = []
        if r.daily_r_embassy >= 0.6 and abs(r.best_lag_h) <= 1:
            use.append("PM$_{2.5}$")
        if r.NOX_rho_tropomi >= 0.2:
            use.append("NO$_x$")
        if r.CO_rho_tropomi >= 0.2:
            use.append("CO")
        lines.append(" & ".join([r.station, "%s--%s" % (str(r["first"])[:4], str(r["last"])[:4]),
                                 "%.2f" % r.daily_r_embassy, "%.2f" % r.ratio_to_embassy,
                                 "%+d" % r.best_lag_h, num(r.NOX_rho_tropomi, 2), num(r.CO_rho_tropomi, 2),
                                 ", ".join(use) if use else "none"]) + r" \\")
    lines.append(r"\end{longtable}")
    lines = [r"\begingroup\small"] + lines + [r"\endgroup"]
    open(os.path.join(TAB, "tab_screening.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    effects()
    scenarios()
    stations()
    sensitivity()
    # screening() is superseded by Table S3 (make_supp_tables.py) and no longer called
    print("wrote", sorted(os.listdir(TAB)))
