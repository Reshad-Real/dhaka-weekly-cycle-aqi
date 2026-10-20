"""Fig. 5: hour profiles of the three activity shocks, and the change in the daily PM2.5 sub-index.

(a) Pooled Dhaka-region percent change in PM2.5 by local hour for Friday (joint-date
    bootstrap, out/hourly_boot.csv, with the working-day placebo band of
    out/hourly_profiles.csv), Eid (event bootstrap) and the lockdown (placebo-based
    sensitivity interval), both from out/events_dependence.csv; pointwise 95% intervals.
(b) Daily PM2.5 sub-index change per shock in index points, pooled over Dhaka-region
    stations, with the intervals of Table 4: 95% intervals from the joint-date bootstrap for
    Friday and Saturday (out/robust_bootstrap_days.csv) and the event bootstrap for Eid
    (out/eid_checks.csv), and for the lockdown the placebo-based sensitivity interval
    (out/lockdown_checks.csv), drawn dashed because it is not a calibrated 95% interval;
    open circles: the US Embassy alone (out/aqi_by_shock.csv).
Usage: python -I fig5_events.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
# validated categorical slots 1-3 (dataviz reference palette; all-pairs safe)
COL = {"Friday": "#2a78d6", "Eid": "#eb6834", "Lockdown": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e9e8e4"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2})
    H = pd.read_csv(os.path.join(OUT, "hourly_boot.csv"))
    fri = H[(H.region == "Dhaka") & (H.tracer == "PM25") & (H.day == "Fri")].sort_values("hour")
    Pl = pd.read_csv(os.path.join(OUT, "hourly_profiles.csv"))
    Pl = Pl[(Pl.region == "Dhaka") & (Pl.tracer == "PM25") & (Pl.day == "Fri")]
    fri = fri.merge(Pl[["hour", "placebo_min", "placebo_max"]], on="hour")
    E = pd.read_csv(os.path.join(OUT, "events_dependence.csv"))
    E = E[E.unit.astype(str).str.isdigit()].copy()
    E["hour"] = E.unit.astype(int)
    E["pct"], E["lo"], E["hi"] = E.b_pct, E.lo_pct, E.hi_pct
    A = pd.read_csv(os.path.join(OUT, "aqi_by_shock.csv")).set_index("event")
    # the intervals of Table 4
    Bd = pd.read_csv(os.path.join(OUT, "robust_bootstrap_days.csv"))
    Bd = Bd[(Bd.region == "Dhaka") & (Bd.tracer == "AQI")].set_index("day")
    Ec = pd.read_csv(os.path.join(OUT, "eid_checks.csv"))
    Ec = Ec[(Ec.analysis == "event_boot") & (Ec.outcome == "AQI")].iloc[0]
    Lc = pd.read_csv(os.path.join(OUT, "lockdown_checks.csv"))
    Lc = Lc[(Lc.analysis == "main_with_placebo_spread") & (Lc.outcome == "AQI")].iloc[0]
    for ev, (b0, lo, hi) in {"Friday": Bd.loc["Fri", ["b_point", "lo_b", "hi_b"]].values,
                             "Saturday": Bd.loc["Sat", ["b_point", "lo_b", "hi_b"]].values,
                             "Eid": (Ec.b, Ec.lo, Ec.hi), "Lockdown": (Lc.b, Lc.lo, Lc.hi)}.items():
        A.loc[ev, ["pooled_aqi_pts", "lo", "hi"]] = [b0, lo, hi]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
    a = ax[0]
    a.fill_between(fri.hour, fri.placebo_min, fri.placebo_max, color=GRID, lw=0, zorder=0,
                   label="working-day placebos")
    series = [("Friday", fri)] + [(ev, E[E.event == ev].sort_values("hour")) for ev in ("Eid", "Lockdown")]
    for name, q in series:
        a.fill_between(q.hour, q.lo, q.hi, color=COL[name], alpha=0.15, lw=0)
        a.plot(q.hour, q.pct, color=COL[name], lw=2, label=name)
        a.text(23.4, q.pct.iloc[-1], name, color=INK, fontsize=7.5, va="center")
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlim(0, 26.5)
    a.set_xticks(range(0, 24, 3))
    a.set_xlabel("Local hour")
    a.set_ylabel("PM2.5 change, pooled Dhaka region (%)")
    a.set_title("(a) When each shock changes PM2.5", loc="left", fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.6)

    b = ax[1]
    order = ["Friday", "Saturday", "Eid", "Lockdown"]
    A = A.reindex(order)
    y = np.arange(len(order))
    cal = np.array([ev != "Lockdown" for ev in order])
    b.errorbar(A.pooled_aqi_pts[cal], y[cal],
               xerr=[(A.pooled_aqi_pts - A.lo)[cal], (A.hi - A.pooled_aqi_pts)[cal]],
               fmt="o", color="#2a78d6", ms=6, capsize=2, label="pooled, Dhaka region (95% interval)")
    eb = b.errorbar(A.pooled_aqi_pts[~cal], y[~cal],
                    xerr=[(A.pooled_aqi_pts - A.lo)[~cal], (A.hi - A.pooled_aqi_pts)[~cal]],
                    fmt="o", color="#2a78d6", ms=6, capsize=2, label="lockdown: sensitivity interval")
    eb[2][0].set_linestyle((0, (3, 2)))
    b.scatter(A.embassy_aqi_pts, y + 0.22, marker="o", s=28, facecolors="none",
              edgecolors="#2a78d6", label="US Embassy")
    for yi, v, lo in zip(y, A.pooled_aqi_pts, A.lo):
        b.text(lo - 1.0, yi, "%.1f" % v, ha="right", va="center", fontsize=7.5, color=INK)
    b.axvline(0, color=INK2, lw=0.8)
    b.set_yticks(y)
    b.set_yticklabels(order)
    b.invert_yaxis()
    b.set_xlabel("Change in the daily PM2.5 sub-index (points)")
    b.set_title("(b) Daily PM2.5 sub-index", loc="left", fontsize=9)
    from matplotlib.lines import Line2D
    b.legend(handles=[
        Line2D([], [], color="#2a78d6", marker="o", ms=5, lw=1.2, label="pooled, Dhaka region (95% interval)"),
        Line2D([], [], color="#2a78d6", marker="o", ms=5, lw=1.2, ls=(0, (3, 2)),
               label="lockdown: sensitivity interval"),
        Line2D([], [], color="#2a78d6", marker="o", ms=5, mfc="none", ls="none", label="US Embassy")],
        frameon=False, fontsize=7, loc="upper left")
    b.set_xlim(min(A.lo.min(), A.embassy_aqi_pts.min()) - 8, max(A.hi.max(), 0) + 3)
    b.spines[["top", "right"]].set_visible(False)
    b.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_events.png"), dpi=220)
    print("wrote fig5_events.png")


if __name__ == "__main__":
    main()
