"""Measured activity loss for each shock, from Google's Community Mobility Reports.

Bangladesh is reported at national level only, 2020-02-15 to 2022-10-15, as percent change
from a same-weekday baseline (median of 2020-01-03 to 2020-02-06).  Two consequences:

  * The weekly cycle is removed by construction, so Fridays cannot be measured here; the
    NO2/PM2.5 ratio test is used for them instead, which needs no activity figure.
  * For Eid, the pandemic moved the whole series, so the Eid effect is measured against the
    surrounding days (days -15..-3 and +6..+15 around Eid) rather than against the baseline.

Writes 11_natexp/out/mobility_shocks.csv.
Usage: python -I mobility.py <google_mobility dir> <event calendar dir>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
os.makedirs(OUT, exist_ok=True)
CATS = {"transit_stations_percent_change_from_baseline": "transit",
        "workplaces_percent_change_from_baseline": "workplaces",
        "retail_and_recreation_percent_change_from_baseline": "retail",
        "grocery_and_pharmacy_percent_change_from_baseline": "grocery",
        "residential_percent_change_from_baseline": "residential"}


def main(mdir, edir):
    m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(mdir, "*_BD_*.csv")))])
    m = m[m["sub_region_1"].isna()].copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.set_index("date")[list(CATS)].rename(columns=CATS).sort_index()
    eid = pd.read_csv(os.path.join(edir, "eid_dates.csv"), parse_dates=["date"])
    hart = pd.read_csv(os.path.join(edir, "hartal_event_days_DERIVED.csv"), parse_dates=["date"])

    rows = []
    # Lockdown ("general holiday"), against the baseline itself
    lk = m.loc["2020-03-26":"2020-05-30"]
    pre = m.loc["2020-02-15":"2020-03-15"]
    r = dict(shock="lockdown 2020-03-26..05-30", n_days=len(lk))
    for c in CATS.values():
        r[c] = lk[c].mean()
        r[c + "_vs_pre"] = lk[c].mean() - pre[c].mean()
    rows.append(r)

    # Eid windows (-1..+3), against the surrounding days
    for _, e in eid.iterrows():
        t = e["date"]
        if t < m.index.min() + pd.Timedelta(days=15) or t > m.index.max() - pd.Timedelta(days=15):
            continue
        win = m.loc[t - pd.Timedelta(days=1): t + pd.Timedelta(days=3)]
        ctl = pd.concat([m.loc[t - pd.Timedelta(days=15): t - pd.Timedelta(days=3)],
                         m.loc[t + pd.Timedelta(days=6): t + pd.Timedelta(days=15)]])
        r = dict(shock="%s %s" % (e["festival"], t.date()), n_days=len(win))
        for c in CATS.values():
            # convert to activity ratio relative to the surrounding days
            r[c] = 100 * ((1 + win[c].mean() / 100) / (1 + ctl[c].mean() / 100) - 1)
        rows.append(r)

    # Strike days inside the mobility window, against the same weekday in the 4 weeks around
    hd = hart[(hart["date"] >= m.index.min()) & (hart["date"] <= m.index.max())]["date"]
    diffs = {c: [] for c in CATS.values()}
    for t in hd:
        if t not in m.index:
            continue
        ref = [t + pd.Timedelta(weeks=k) for k in (-2, -1, 1, 2)]
        ref = [x for x in ref if x in m.index and x not in set(hd)]
        if not ref:
            continue
        for c in CATS.values():
            diffs[c].append(100 * ((1 + m.loc[t, c] / 100) / (1 + m.loc[ref, c].mean() / 100) - 1))
    r = dict(shock="strike days (n=%d)" % len(diffs["transit"]), n_days=len(diffs["transit"]))
    for c in CATS.values():
        r[c] = float(np.mean(diffs[c])) if diffs[c] else np.nan
        r[c + "_se"] = float(np.std(diffs[c], ddof=1) / np.sqrt(len(diffs[c]))) if len(diffs[c]) > 1 else np.nan
    rows.append(r)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "mobility_shocks.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["shock", "n_days", "transit", "workplaces", "retail", "grocery", "residential"]
    print(t[cols].round(1).to_string(index=False))
    eids = t[t.shock.str.startswith("Eid")]
    print("\nEid mean (n=%d): transit %.1f%%, workplaces %.1f%%, retail %.1f%%" % (
        len(eids), eids.transit.mean(), eids.workplaces.mean(), eids.retail.mean()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
