"""Write the data-bearing manuscript tables as LaTeX, straight from the result files, so no
number in them is transcribed by hand.

  paper/tables/tab_effects.tex     Table 3, from out/table3.csv
  paper/tables/tab_scenarios.tex   Table 4, from out/aqi_scenarios.csv and
                                   out/aqi_scenarios_stations.csv
  paper/tables/tab_screening.tex   Table S2, from out/cams_validation.csv
Usage: python -I make_paper_tables.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
TAB = os.path.join(HERE, "..", "paper", "tables")
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
             r"\caption{Pooled effects by diurnal block and shock, percent (95\% interval), and the change "
             r"in the daily AQI in index points. Friday: against the same block on Sunday to Wednesday, "
             r"joint-date block bootstrap. Saturday: same contrast, random-effects pooling. Eid: against "
             r"the surrounding days. Lockdown: against the year, month and weather expectation. "
             r"Random-effects intervals treat stations as independent and are optimistic. The night "
             r"block is 00--05\,h of the calendar day, so a Friday's night block is Thursday night."
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
        cells = [r.shock, r.region.replace("national", "National"), r.tracer.replace("PM2.5", r"PM$_{2.5}$")]
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
    names = [("rest_day", "Every day a rest day (Friday 06\\,h to Saturday 05\\,h profile)"),
             ("cut_allday", "Friday-afternoon-sized cut at all 24 hours"),
             ("cut_day_only", "The same cut at 12--23\\,h only"),
             ("cut_night_only", "The same cut at 00--11\\,h only"),
             ("eid_like", "Eid hourly profile on every day"),
             ("lockdown_like", "Lockdown hourly profile on every day")]
    b = S.loc["baseline"]
    lines = [r"\begin{longtable}{L{4.6cm} C{1.5cm} C{1.9cm} C{1.4cm} C{1.4cm} C{1.4cm} C{2.4cm}}",
             r"\caption{What activity reductions of the observed kinds would buy, applied to the US "
             r"Embassy hourly record, January 2019 to March 2025: change in annual mean PM$_{2.5}$, in the mean daily AQI "
             r"and in days per year at or above three index thresholds. Baseline: mean PM$_{2.5}$ "
             + num(b.mean_pm, 1, False) + r"\,$\mu$g\,m$^{-3}$, mean AQI " + num(b.mean_aqi, 1, False)
             + r", " + num(b.days150, 0, False) + r", " + num(b.days200, 0, False) + r" and "
             + num(b.days300, 0, False) + r" days per year at or above 150, 200 and 300. The last "
             r"column gives the range of the mean-AQI change over the " + NUM[St.station.nunique()]
             + r" screened Dhaka CAMS "
             r"stations, 2019--2024.}\label{tab:scenarios}\\",
             r"\toprule",
             r"\textbf{Scenario} & \textbf{PM$_{2.5}$ (\%)} & \textbf{AQI (points)} & \textbf{Days $\geq$150} & "
             r"\textbf{Days $\geq$200} & \textbf{Days $\geq$300} & \textbf{AQI, CAMS range} \\",
             r"\midrule\endfirsthead",
             r"\toprule",
             r"\textbf{Scenario} & \textbf{PM$_{2.5}$ (\%)} & \textbf{AQI (points)} & \textbf{Days $\geq$150} & "
             r"\textbf{Days $\geq$200} & \textbf{Days $\geq$300} & \textbf{AQI, CAMS range} \\",
             r"\midrule\endhead",
             r"\bottomrule\endfoot"]
    for key, lab in names:
        r = S.loc[key]
        cams = ("%s to %s" % (num(rng.loc[key, "min"]), num(rng.loc[key, "max"]))) if key in rng.index else "--"
        extra = ""
        if key == "rest_day":
            extra = r"\newline (%s to %s)" % (num(S.loc["rest_day_lo", "d_mean_aqi"]),
                                              num(S.loc["rest_day_hi", "d_mean_aqi"]))
        lines.append(" & ".join([lab, num(r.d_pm_pct), num(r.d_mean_aqi) + extra, num(r.d_days150),
                                 num(r.d_days200), num(r.d_days300), cams]) + r" \\")
    lines.append(r"\end{longtable}")
    lines = [r"\begingroup\small"] + lines + [r"\endgroup"]
    open(os.path.join(TAB, "tab_scenarios.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


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
    # screening() is superseded by Table S3 (make_supp_tables.py) and no longer called
    print("wrote", sorted(os.listdir(TAB)))
