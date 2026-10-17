"""Screen each Dhaka-region CAMS station before it is used.

  V1  timestamp alignment: hourly PM2.5 correlation with the US Embassy monitor at lags of
      -3 to +3 hours.  After the hour-ending shift in cams_qc.py the peak should sit at 0.
  V2  agreement: daily PM2.5 correlation and ratio with the Embassy (2019-2024 overlap).
  V3  gas tracer validity: daily NO2 and NOx against TROPOMI NO2 (Spearman, deseasonalized by
      year-month means).  A gas analyzer that does not track the satellite column at all
      is not used as a traffic tracer.
Writes 11_natexp/out/cams_validation.csv.
Usage: python -I cams_validate.py <FYDP_D root>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
DHAKA = ["BARC", "Darussalam", "DoE", "Savar", "Gazipur", "Narayanganj", "Narsingdi"]


def main(root):
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    fs = sorted(glob.glob(os.path.join(root, "02_data", "raw", "ground", "Dhaka_PM2.5_*_YTD.csv")))
    e = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    e = e.rename(columns={"Date (LT)": "time", "Raw Conc.": "pm", "QC Name": "qc"})
    e["time"] = pd.to_datetime(e["time"], format="%Y-%m-%d %I:%M %p")
    e = e[(e.qc == "Valid") & (e.pm > 0)].drop_duplicates("time").set_index("time")["pm"]
    tr = pd.read_csv(os.path.join(root, "02_data", "raw", "tropomi_no2_dhaka_daily.csv"),
                     parse_dates=["date"]).set_index("date")[
        "tropospheric_NO2_column_number_density_mean"]
    out = []
    for st in DHAKA:
        h = cams[cams.station == st].drop_duplicates("time").set_index("time").sort_index()
        if h.empty:
            continue
        r = dict(station=st, first=h.index.min().date(), last=h.index.max().date(),
                 pm25_hours=int(h["PM25"].notna().sum()))
        lp = np.log(h["PM25"].where(h["PM25"] > 0))
        le = np.log(e)
        best, bestlag = -9, None
        for lag in range(-3, 4):
            c = lp.corr(le.shift(lag, freq="h").reindex(lp.index))
            r["lag%+d" % lag] = round(c, 3) if np.isfinite(c) else np.nan
            if np.isfinite(c) and c > best:
                best, bestlag = c, lag
        r["best_lag_h"] = bestlag
        dd = h[["PM25", "PM10", "NO2", "NOX", "CO"]].resample("D").mean()
        cnt = h[["PM25"]].resample("D").count()["PM25"]
        dd = dd[cnt >= 18]
        ed = e.resample("D").mean()
        j = dd.join(ed.rename("emb"), how="inner").dropna(subset=["PM25", "emb"])
        r["daily_r_embassy"] = round(np.corrcoef(np.log(j.PM25), np.log(j.emb))[0, 1], 3) if len(j) > 50 else np.nan
        r["ratio_to_embassy"] = round((j.PM25 / j.emb).median(), 3) if len(j) > 50 else np.nan
        r["n_days_overlap"] = len(j)
        for g in ("NO2", "NOX", "CO"):
            k = dd[[g]].join(tr.rename("trop"), how="inner").dropna()
            k = k[(k[g] > 0) & (k.trop > 0)]
            if len(k) < 100:
                r[g + "_rho_tropomi"] = np.nan
                continue
            ym = k.index.to_period("M")
            a = np.log(k[g]) - np.log(k[g]).groupby(ym).transform("mean")
            b = np.log(k.trop) - np.log(k.trop).groupby(ym).transform("mean")
            r[g + "_rho_tropomi"] = round(a.corr(b, method="spearman"), 3)
            r[g + "_median"] = round(k[g].median(), 2)
        out.append(r)
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(OUT, "cams_validation.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
