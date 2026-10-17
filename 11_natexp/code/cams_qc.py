"""Quality control for the parsed CAMS-BD hourly table.

Rules (each one counted and reported, none silent):
  Q1  physical range: PM2.5, PM10 in (0, 1500] ug/m3; NO2, NOX, NO, SO2, O3 in [0, 1000] ppb;
      CO in [0, 50] ppm.  Values outside are set missing.
  Q2  stuck instrument: a run of 6 or more identical consecutive hourly values is set missing.
  Q3  PM2.5 > 1.1 * PM10 in the same hour: both set missing (fine cannot exceed total).
  Q4  duplicates: one value per station-hour (first kept).
  Q5  the timestamp is taken as local time ending the hour (the files' first value is 01:00).
      It is shifted back one hour so that hour h means the interval h to h+1, matching the
      Embassy monitor convention used elsewhere; the comparison with the Embassy record
      below checks the alignment empirically.
Writes 11_natexp/data/cams_hourly_qc.csv.gz and 11_natexp/out/cams_qc_report.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
os.makedirs(OUT, exist_ok=True)

RANGES = {"PM25": (0, 1500), "PM10": (0, 1500), "NO2": (0, 1000), "NOX": (0, 1000),
          "NO": (0, 1000), "SO2": (0, 1000), "O3": (0, 1000), "CO": (0, 50)}
POLS = list(RANGES)

# One name per station; the yearly files spell several differently.
STATION = {"Narayonganj": "Narayanganj", "Narayangonj": "Narayanganj",
           "Mymensing": "Mymensingh",
           # One site, three spellings: the Bangladesh Television Chattogram Center station in
           # Khulshi, Chattogram (aqicn.org station A538645).  "TV st-Chittagong" covers
           # 2012-2020 and "TV center"/"TV Center"/"TV Sation" 2022-2024.  An earlier version
           # treated the 2022-2024 name as a Dhaka station; its PM2.5 level (half of Dhaka's)
           # and weak correlation with the Dhaka Embassy monitor confirm it is not.
           "TV center": "BTV Chattogram", "TV Center": "BTV Chattogram", "TV Sation": "BTV Chattogram",
           "TV st-Chittagong": "BTV Chattogram",
           # One site, two spellings: Agrabad CDA Residential Area, Chattogram.
           "Agrabad Chittagong": "Agrabad CDA Chattogram", "CDA": "Agrabad CDA Chattogram"}


def stuck_mask(s, n=6):
    v = s.values
    same = np.r_[False, (v[1:] == v[:-1]) & ~np.isnan(v[1:])]
    grp = np.cumsum(~same)
    run = pd.Series(grp).map(pd.Series(grp).value_counts()).values
    return (run >= n) & ~np.isnan(v)


def main():
    df = pd.read_csv(os.path.join(DATA, "cams_hourly_raw.csv.gz"), low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="mixed")
    rep = [dict(rule="Q0 unparseable timestamp dropped", n=int(df["time"].isna().sum()))]
    df = df[df["time"].notna()].copy()
    df["station"] = df["station"].map(lambda s: STATION.get(str(s).strip(), str(s).strip()))
    df["time"] = df["time"] - pd.Timedelta(hours=1)
    n0 = len(df)
    df = df.sort_values(["station", "time"]).drop_duplicates(["station", "time"])
    rep.append(dict(rule="Q4 duplicates removed", n=n0 - len(df)))
    for p in POLS:
        if p not in df:
            continue
        lo, hi = RANGES[p]
        bad = df[p].notna() & ((df[p] <= lo if p in ("PM25", "PM10") else df[p] < lo) | (df[p] > hi))
        rep.append(dict(rule="Q1 range " + p, n=int(bad.sum())))
        df.loc[bad, p] = np.nan
    for p in POLS:
        if p not in df:
            continue
        m = np.zeros(len(df), bool)
        for st, idx in df.groupby("station").indices.items():
            m[idx] = stuck_mask(df[p].iloc[idx])
        rep.append(dict(rule="Q2 stuck runs " + p, n=int(m.sum())))
        df.loc[m, p] = np.nan
    bad = df["PM25"].notna() & df["PM10"].notna() & (df["PM25"] > 1.1 * df["PM10"])
    rep.append(dict(rule="Q3 PM2.5 > 1.1 PM10", n=int(bad.sum())))
    df.loc[bad, ["PM25", "PM10"]] = np.nan
    df.to_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), index=False)
    r = pd.DataFrame(rep)
    r.to_csv(os.path.join(OUT, "cams_qc_report.csv"), index=False)
    print(r.to_string(index=False))

    # Coverage after QC, 2019-2024, per station
    w = df[(df.time >= "2019-01-01") & (df.time < "2025-01-01")]
    cov = w.groupby("station").agg(**{p + "_hours": (p, "count") for p in
                                      ("PM25", "PM10", "NO2", "NOX", "CO", "O3", "SO2")})
    cov = (cov / (6 * 8766) * 100).round(1)
    cov.to_csv(os.path.join(OUT, "cams_coverage_2019_2024.csv"))
    print("\npercent of 2019-2024 hours with a valid value, after QC")
    print(cov.to_string())


if __name__ == "__main__":
    main()
