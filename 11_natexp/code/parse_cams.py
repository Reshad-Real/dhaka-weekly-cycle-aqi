"""Parse the four CAMS-BD workbooks into one tidy hourly table.

The workbooks are raw exports: one sheet each, station blocks stacked vertically, a header
row ("Location", "Date & Time", pollutants, met) that can recur and change column order
between files, and a units row per block whose timestamp cell is empty.  This parser
tracks the current header, keeps only rows whose second cell is a timestamp, maps values
by header name, and writes:

    11_natexp/data/cams_hourly_raw.csv.gz  one row per station-hour
    11_natexp/data/cams_inventory.csv    per station: rows, first/last time, % non-missing

Usage:  python -I parse_cams.py <dir containing the four xlsx files>
Nothing is cleaned here beyond type coercion; QC is a separate, documented step.
"""
import datetime as dt
import glob
import os
import sys

import numpy as np
import openpyxl
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "..", "data")
os.makedirs(OUTD, exist_ok=True)

CANON = {"so2": "SO2", "no": "NO", "no2": "NO2", "nox": "NOX", "co": "CO", "o3": "O3",
         "pm10": "PM10", "pm2.5": "PM25", "pm25": "PM25", "ws": "WS", "wd": "WD",
         "temp": "TEMP", "rh": "RH", "bp": "BP", "sr": "SR", "rain": "RAIN", "vws": "VWS"}


def canon(h):
    if h is None:
        return None
    k = str(h).strip().lower().replace(" ", "")
    return CANON.get(k, str(h).strip())


def num(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except ValueError:
        return np.nan


EXCEL0 = pd.Timestamp("1899-12-30")


def hhmm(x):
    """Hours and minutes from the separate Time cell of the 2012-2021 layout."""
    if isinstance(x, dt.time):
        return x.hour, x.minute
    if isinstance(x, dt.datetime):
        return x.hour, x.minute
    if isinstance(x, (int, float)) and 0 <= x < 1:
        m = int(round(x * 1440))
        return m // 60, m % 60
    if isinstance(x, str):
        s = x.strip()
        try:
            h, m = s.split(":")[:2]
            return int(h), int(m[:2])
        except ValueError:
            return None
    return None


def parse(path):
    """Two layouts occur.  A (2022-2024): Location, 'Date & Time' as one timestamp, values.
    B (2012-2021): Location, Date, Time, values, where Date is an Excel serial number on most
    rows, a text date on some, and empty on whole station blocks; empty-date rows cannot be
    placed in time and are dropped and counted, never guessed."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows, header, layout, nhdr = [], None, None, 0
    textdates = []           # (row index in rows, station, raw text, hh, mm), resolved below
    dropped = {}
    for ws in wb.worksheets:
        for r in ws.iter_rows(values_only=True):
            if not r:
                continue
            cells = list(r)
            c1 = str(cells[1]).strip().lower() if len(cells) > 1 and cells[1] is not None else ""
            c2 = str(cells[2]).strip().lower() if len(cells) > 2 and cells[2] is not None else ""
            c0 = str(cells[0]).strip().lower() if cells[0] is not None else ""
            if c0 == "location" or c1 in ("date & time", "date") or (c1 == "" and c2 == "time"):
                if c1 == "date & time":
                    layout, header = "A", [canon(c) for c in cells]
                elif c1 == "date" and c2 == "time":
                    layout, header = "B", [canon(c) for c in cells]
                elif c1 == "" and c2 == "time" and header is not None:
                    pass       # a block header with its Date cell missing; layout B, no dates
                nhdr += 1
                continue
            if header is None:
                continue
            st = str(cells[0]).strip() if cells[0] is not None else None
            first = 2 if layout == "A" else 3
            if layout == "A":
                t = cells[1]
                if not isinstance(t, (dt.datetime, pd.Timestamp)):
                    dropped[st] = dropped.get(st, 0) + 1
                    continue
                t = pd.Timestamp(t)
            else:
                d, tm = cells[1], hhmm(cells[2]) if len(cells) > 2 else None
                if isinstance(d, (int, float)) and d > 20000:
                    if tm is None:
                        # Most 2021 rows (and some 2020 rows) carry the full date and time in
                        # the Date serial and leave Time empty; the fraction is the hour.
                        t = (EXCEL0 + pd.Timedelta(days=float(d))).round("min")
                    else:
                        t = EXCEL0 + pd.Timedelta(days=int(np.floor(d + 1e-9))) + pd.Timedelta(
                            hours=tm[0], minutes=tm[1])
                elif isinstance(d, dt.datetime) and tm is not None:
                    t = pd.Timestamp(d.date()) + pd.Timedelta(hours=tm[0], minutes=tm[1])
                elif isinstance(d, dt.datetime):
                    t = pd.Timestamp(d).round("min")       # date and time in one cell
                elif isinstance(d, str) and d.strip() and tm is not None:
                    t = None
                    textdates.append((len(rows), st, d.strip(), tm))
                else:
                    dropped[st] = dropped.get(st, 0) + 1
                    continue
            rec = {"station": st, "time": t}
            for j, h in enumerate(header[first:], start=first):
                if h and j < len(cells):
                    rec[h] = num(cells[j])
            rows.append(rec)
    wb.close()
    # Resolve text dates per station: day-first if any first field exceeds 12, month-first
    # if any second field does, otherwise reported as ambiguous and dropped.
    if textdates:
        tdf = pd.DataFrame(textdates, columns=["i", "station", "raw", "tm"])
        parts = tdf["raw"].str.extract(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})")
        tdf = tdf.join(parts.astype(float).rename(columns={0: "a", 1: "b", 2: "y"}))
        for st, g in tdf.groupby("station"):
            dayfirst = (g.a > 12).any()
            monthfirst = (g.b > 12).any()
            fmt = "dayfirst" if dayfirst and not monthfirst else (
                "monthfirst" if monthfirst and not dayfirst else "ambiguous")
            print("   text dates, %-20s %6d rows -> %s" % (st, len(g), fmt))
            for _, x in g.iterrows():
                if fmt == "ambiguous" or np.isnan(x.a):
                    rows[x.i]["time"] = None
                    continue
                dd, mm = (x.a, x.b) if fmt == "dayfirst" else (x.b, x.a)
                yy = int(x.y) + (2000 if x.y < 100 else 0)
                try:
                    rows[x.i]["time"] = pd.Timestamp(year=yy, month=int(mm), day=int(dd),
                                                     hour=x.tm[0], minute=x.tm[1])
                except ValueError:
                    rows[x.i]["time"] = None
    df = pd.DataFrame(rows)
    n_bad = int(df["time"].isna().sum())
    df = df[df["time"].notna()].copy()
    df["file"] = os.path.basename(path)
    print("%s: %d data rows kept, %d header rows, %d rows without a usable date dropped"
          % (os.path.basename(path), len(df), nhdr, n_bad + sum(dropped.values())))
    for st, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
        if n > 100:
            print("   dropped (no date): %-20s %d" % (st, n))
    REPORT.append(dict(file=os.path.basename(path), kept=len(df), header_rows=nhdr,
                       dropped_no_date=n_bad + sum(dropped.values())))
    for st, n in dropped.items():
        REPORT.append(dict(file=os.path.basename(path), station=st, dropped_no_date=n))
    return df


REPORT = []      # rows kept and dropped per file and station, written to cams_parse_report.csv


def main(src):
    fs = sorted(glob.glob(os.path.join(src, "*.xlsx")))
    parts = [parse(f) for f in fs]
    pd.DataFrame(REPORT).to_csv(os.path.join(HERE, "..", "out", "cams_parse_report.csv"), index=False)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(OUTD, "cams_hourly_raw.csv.gz"), index=False)
    pol = [c for c in ("PM25", "PM10", "NO2", "NOX", "NO", "CO", "O3", "SO2", "WS", "WD",
                       "TEMP", "RH") if c in df.columns]
    inv = df.groupby(["file", "station"]).agg(rows=("time", "size"), first=("time", "min"),
                                              last=("time", "max"))
    for p in pol:
        inv[p + "_pct"] = df.groupby(["file", "station"])[p].apply(
            lambda s: 100 * s.notna().mean()).round(1)
    inv.to_csv(os.path.join(OUTD, "cams_inventory.csv"))
    pd.set_option("display.width", 250)
    print(inv.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
