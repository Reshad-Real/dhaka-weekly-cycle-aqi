"""Assemble the public data and code deposit cited in the Data availability statement.

The deposit mirrors the project's folder layout, so the analysis runs from a clone exactly
as it runs here:

  11_natexp/code/            every analysis script
  11_natexp/data/            the parsed and the quality-controlled CAMS station-hour tables,
                             approximate station locations
  11_natexp/out/             every result file and figure the manuscript and supplement use
  02_data/raw/               derived inputs: gap-filled ERA5 meteorology, the TROPOMI daily
                             series and baseline field, the US Embassy hourly PM2.5 files
  02_data/gapfill.py         the boundary-layer gap fill, with its cross-validation result
  06_results/metrics/        the two earlier-analysis results the manuscript quotes
  Traffic/data_availability/ event calendars (Eid dates; DERIVED strike dates only, as
                             ACLED's terms require) and the road-length grid of Fig. 1
  Traffic/gates/data/        geoBoundaries ADM1 and ADM2 for Fig. 1
  README.md, LICENSE.md, MANIFEST.csv (path, bytes, SHA-256)

Not redistributed: the CAMS-BD workbooks (only needed to re-parse) and the Google mobility
CSVs; README.md gives their exact sources and checksums.

Usage: python -I make_deposit.py <FYDP_D root>
"""
import glob
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
DEP = os.path.join(BASE, "deposit")

REPO = "https://github.com/Reshad-Real/dhaka-weekly-cycle-aqi"

README = """# Dhaka's working week does not reach the night: data and code

Data and code for the manuscript

> Md. Reshad Al Muttaki, Sadia Afrin, Hasanat Fariha Noon, Alvi Ibn Amzad Anil,
> SK Tahmed Salim Rafid and Mirza Rasheduzzaman. *Dhaka's working week does not reach the
> night: why activity cuts barely move the daily Air Quality Index.* Prepared for submission
> to Urban Climate, 2026.

Department of Electrical and Electronic Engineering, BSRM School of Engineering, BRAC
University, Dhaka, Bangladesh. Correspondence: Mirza Rasheduzzaman,
mirza.rasheduzzaman@bracu.ac.bd.

## Layout

The folders mirror the project's own layout, so every script runs from a clone unchanged.

| Path | What it holds |
|---|---|
| `11_natexp/code/` | Every analysis script (Python). `run_all.py` runs them in order. |
| `11_natexp/data/cams_hourly_raw.csv.gz` | The DoE CAMS hourly record as parsed from CAMS-BD, one row per station-hour. |
| `11_natexp/data/cams_hourly_qc.csv.gz` | The same after the documented quality control (`cams_qc.py`). |
| `11_natexp/data/station_locations.csv` | Approximate station positions, geocoded from the host institutions with OpenStreetMap Nominatim. |
| `11_natexp/out/` | Every result file (CSV, JSON) and figure (PNG) the manuscript and supplement draw on. |
| `02_data/raw/meteo_dhaka_hourly_filled.csv` | ERA5 hourly meteorology at 23.8103 N, 90.4125 E from the Open-Meteo archive API (`models=era5`, `timezone=Asia/Dhaka`, retrieved 17 October 2026), local time, with the 4,368 gap-filled boundary-layer hours flagged in `blh_filled`. |
| `02_data/gapfill.py`, `06_results/metrics/blh_gapfill_cv.csv` | The boundary-layer gap fill and its blocked cross-validation (8 held-out 181-day blocks). |
| `02_data/raw/tropomi_no2_dhaka_daily.csv` | Daily TROPOMI tropospheric NO2 over the study domain. |
| `02_data/raw/no2_field_baseline.csv` | Mean TROPOMI NO2 field on non-event days (Fig. 1b). |
| `02_data/raw/ground/Dhaka_PM2.5_*_YTD.csv` | US Embassy Dhaka hourly PM2.5, 1 January 2019 to 24 March 2025, as published by AirNow. |
| `06_results/metrics/lockdown_test.json` | The earlier full-year lockdown estimate for the Embassy monitor that Section 4.4 compares against. |
| `Traffic/data_availability/02_event_calendars/` | Eid dates, and strike-day dates derived from ACLED (no ACLED event records). |
| `Traffic/data_availability/01_dose_field/dose_field_0p05deg.csv` | OpenStreetMap road length per 0.05 degree cell (Fig. 1a). |
| `Traffic/gates/data/bgd_ADM1.geojson`, `bgd_ADM2.geojson` | geoBoundaries administrative boundaries (Fig. 1). |
| `MANIFEST.csv` | Every file with its size and SHA-256 digest. |

## Inputs not redistributed

- **CAMS-BD workbooks** (needed only to re-run the parser): Ansari, N.W., Rafi, A.H., 2026.
  CAMS-BD: hourly air quality and meteorological measurements dataset. Mendeley Data, V1.
  https://doi.org/10.17632/hwmzyh5w4w.1 (CC BY 4.0). Unzip `CAMS-BD.zip` (SHA-256
  d9bec4d0d6dd7696f6f6b4c95ee03eaa04139b55481a1db6267d153991e04ecc) into
  `02_data/raw/cams_bd/`, then run `run_all.py --parse`.
- **Google COVID-19 Community Mobility Reports**: download
  https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip (SHA-256
  303a1d7ba631c1e66b5c0ff7e1f7b29a121c717d7e040b0e39664f5de2059447 on 16 October 2026)
  and place `2020_BD_Region_Mobility_Report.csv`, `2021_BD_...` and `2022_BD_...` in
  `02_data/raw/google_mobility/`. Without them `run_all.py` skips `mobility.py` and keeps
  the mobility results already in `11_natexp/out/`.

## Reproducing

Python 3.14 with numpy 2.4, pandas 3.0, scipy 1.17, statsmodels 0.15, matplotlib 3.10,
geopandas 1.1, shapely 2.1, scikit-learn 1.8 and openpyxl 3.1. From the repository root:

```
python -I 11_natexp/code/run_all.py
```

`run_all.py` rewrites every file in `11_natexp/out/` and logs each step to
`11_natexp/out/run_all.log`. `11_natexp/code/verify_claims.py` re-derives every number in
the manuscript from `11_natexp/out/`; it needs the manuscript source, which is not part of
this deposit until publication.

## Sources and attribution

- DoE CAMS data: Department of Environment, Bangladesh, via CAMS-BD (CC BY 4.0).
- US Embassy Dhaka PM2.5: U.S. Department of State Air Quality Monitoring Program, AirNow,
  https://files.airnowtech.org/airnow/EmbassyHistorical/Dhaka/
- TROPOMI NO2: contains modified Copernicus Sentinel data (2019-2025) processed by ESA,
  via Google Earth Engine collection COPERNICUS/S5P/OFFL/L3_NO2.
- ERA5: contains modified Copernicus Climate Change Service information (2019-2025),
  through the Open-Meteo historical weather API (CC BY 4.0).
- Strike dates: derived from ACLED (Armed Conflict Location & Event Data),
  https://acleddata.com. Only derived dates and counts are shared, as ACLED's terms of use
  require.
- Administrative boundaries: geoBoundaries (Runfola et al., 2020). ADM1
  (BGD-ADM1-32408957, from Wikimedia Commons) is CC0 1.0; ADM2 (BGD-ADM2-16705992, from the
  Bangladesh Bureau of Statistics and OCHA ROAP) is CC BY 3.0 IGO.
- Road network and station geocoding: (c) OpenStreetMap contributors, ODbL 1.0.
"""

LICENSE = """# Licence

## Code

The scripts in `11_natexp/code/` and `02_data/gapfill.py` are released under the MIT
Licence:

Copyright (c) 2026 Md. Reshad Al Muttaki, Sadia Afrin, Hasanat Fariha Noon, Alvi Ibn
Amzad Anil, SK Tahmed Salim Rafid and Mirza Rasheduzzaman

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

## Data and figures

The results, figures and derived tables in this deposit are released under the Creative
Commons Attribution 4.0 International licence (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/), with these inherited terms:

- the CAMS tables derive from CAMS-BD (CC BY 4.0): cite Ansari and Rafi (2026);
- the meteorology derives from ERA5 through Open-Meteo (CC BY 4.0);
- the TROPOMI series derive from Copernicus Sentinel-5P data, free and open under the
  Copernicus data policy;
- the US Embassy files are U.S. Government data published by AirNow;
- the geoBoundaries files keep their own licences: ADM1 CC0 1.0, ADM2 CC BY 3.0 IGO
  (Bangladesh Bureau of Statistics, OCHA ROAP);
- the road-length grid and the station positions derive from OpenStreetMap
  (ODbL 1.0, (c) OpenStreetMap contributors);
- the strike calendar contains derived dates only; ACLED's terms of use apply to the
  underlying event data.
"""

FILES = [
    # (source relative to the FYDP_D root, or a glob)
    "02_data/raw/meteo_dhaka_hourly_filled.csv",
    "02_data/raw/tropomi_no2_dhaka_daily.csv",
    "02_data/raw/no2_field_baseline.csv",
    "02_data/raw/ground/Dhaka_PM2.5_*_YTD.csv",
    "02_data/gapfill.py",
    "06_results/metrics/blh_gapfill_cv.csv",
    "06_results/metrics/lockdown_test.json",
    "Traffic/data_availability/02_event_calendars/eid_dates.csv",
    "Traffic/data_availability/02_event_calendars/hartal_event_days_DERIVED.csv",
    "Traffic/data_availability/01_dose_field/dose_field_0p05deg.csv",
    "Traffic/gates/data/bgd_ADM1.geojson",
    "Traffic/gates/data/bgd_ADM2.geojson",
    "11_natexp/data/cams_hourly_raw.csv.gz",
    "11_natexp/data/cams_hourly_qc.csv.gz",
    "11_natexp/data/station_locations.csv",
]


def copy(src, rel):
    dst = os.path.join(DEP, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main(root):
    # Rebuild from scratch but keep the git history of the published repository.
    os.makedirs(DEP, exist_ok=True)
    for name in os.listdir(DEP):
        if name == ".git":
            continue
        p = os.path.join(DEP, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for f in sorted(os.listdir(os.path.join(BASE, "code"))):
        if f.endswith(".py"):
            copy(os.path.join(BASE, "code", f), "11_natexp/code/" + f)
    for pat in FILES:
        hits = sorted(glob.glob(os.path.join(root, pat)))
        assert hits, pat
        for p in hits:
            copy(p, os.path.relpath(p, root).replace("\\", "/"))
    for f in sorted(os.listdir(os.path.join(BASE, "out"))):
        if f.endswith((".csv", ".json", ".png")):
            copy(os.path.join(BASE, "out", f), "11_natexp/out/" + f)
    open(os.path.join(DEP, "README.md"), "w", encoding="utf-8").write(README)
    open(os.path.join(DEP, "LICENSE.md"), "w", encoding="utf-8").write(LICENSE)
    # no line-ending conversion, so every file checks out byte-identical to MANIFEST.csv
    open(os.path.join(DEP, ".gitattributes"), "w", encoding="utf-8").write("* -text\n")
    rows = ["path,bytes,sha256"]
    for d, _, fs in os.walk(DEP):
        for f in sorted(fs):
            p = os.path.join(d, f)
            rel = os.path.relpath(p, DEP).replace("\\", "/")
            if rel == "MANIFEST.csv" or rel.startswith(".git/"):
                continue
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            rows.append("%s,%d,%s" % (rel, os.path.getsize(p), h))
    open(os.path.join(DEP, "MANIFEST.csv"), "w", encoding="utf-8").write("\n".join([rows[0]] + sorted(rows[1:])) + "\n")
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DEP) for f in fs
               if ".git" + os.sep not in os.path.join(d, f) + os.sep and not d.endswith(".git"))
    print("deposit: %d files, %.1f MB" % (len(rows) - 1, size / 1e6))


if __name__ == "__main__":
    main(sys.argv[1])
