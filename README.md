# Partial-day activity reductions and the daily air quality index (Dhaka): Data and code

Data and code for the manuscript

> Md. Reshad Al Muttaki, Sadia Afrin, Hasanat Fariha Noon, Alvi Ibn Amzad Anil,
> SK Tahmed Salim Rafid and Mirza Rasheduzzaman. *Partial-day activity reductions and the
> daily air quality index: Hour-resolved PM2.5 evidence from Dhaka's weekly rest day.*
> Manuscript prepared for submission to Urban Climate, 2026 (earlier drafts were titled
> "Dhaka's working week does not reach the night" and "Diurnal patterns of weekly activity
> and PM2.5 in Dhaka").
>
> Release v2.7 is the version the manuscript describes; it has the same analysis code and
> results as v2.3, with every figure drawn at its printed size.

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
| `11_natexp/data/station_locations.csv` | Station hosts and positions from the World Air Quality Index listing; the US Embassy from OpenStreetMap Nominatim. |
| `11_natexp/out/` | Every result file (CSV, JSON, the bootstrap draws as CSV.GZ) and figure (PNG) the manuscript and supplement draw on. |
| `11_natexp/paper/working/tables/supp_*.tex` | Every supplementary table, as written by `make_supp_tables.py` from the result files. |
| `02_data/raw/meteo_dhaka_hourly_filled.csv` | ERA5 hourly meteorology requested at 23.8103 N, 90.4125 E from the Open-Meteo archive API (`models=era5`, `timezone=Asia/Dhaka`, retrieved 17 October 2026), which returns the nearest 0.25 degree ERA5 cell (23.75 N, 90.50 E) without interpolation; local time, with the 4,368 gap-filled boundary-layer hours flagged in `blh_filled`. |
| `02_data/raw/meteo_dhaka_hourly.csv` | The same ERA5 download before the gap fill (the boundary-layer gap is empty here); used for the gap-fill error analysis. |
| `02_data/gapfill.py`, `06_results/metrics/blh_gapfill_cv.csv` | The boundary-layer gap fill and its blocked cross-validation (8 held-out 181-day blocks). |
| `02_data/raw/tropomi_no2_dhaka_daily.csv` | Daily TROPOMI tropospheric NO2 over the study domain. |
| `02_data/raw/no2_field_baseline.csv` | Mean TROPOMI NO2 field on non-event days (Fig. 1b). |
| `02_data/raw/ground/Dhaka_PM2.5_*_YTD.csv` | US Embassy Dhaka hourly PM2.5, 1 January 2019 to 24 March 2025, as published by AirNow. |
| `02_data/raw/ground/Dhaka_OZONE_*_YTD.csv` | US Embassy Dhaka hourly ozone, 2023 to March 2025, as published by AirNow; used only to check which sub-index sets the AQI. |
| `Traffic/data_availability/02_event_calendars/` | Eid dates, and strike-day dates and Dhaka-district strike-day counts derived from ACLED (no ACLED event records). |
| `Traffic/data_availability/01_dose_field/dose_field_0p05deg.csv` | OpenStreetMap road length per 0.05 degree cell (Fig. 1a). |
| `Traffic/gates/data/bgd_ADM1.geojson`, `bgd_ADM2.geojson` | geoBoundaries administrative boundaries (Fig. 1). |
| `MANIFEST.csv` | Every file with its size and SHA-256 digest. |

## Where the main results come from

| Result | Script (`11_natexp/code/`) | Result files (`11_natexp/out/`) |
|---|---|---|
| Weekly contrasts by diurnal block, joint-date bootstrap intervals, Benjamini-Hochberg q-values, block-length and sensitivity runs | `robustness.py` | `robust_bootstrap*.csv`, `robust_bh_bootstrap.csv`, `robust_pooled.csv` |
| Hour-by-hour profiles, with pointwise intervals and simultaneous bands for each 72-hour curve | `hourly_profiles.py`, `hourly_boot.py` | `hourly_profiles*.csv`, `hourly_boot.csv` |
| Reconstruction of the daily change on the log and arithmetic (PPML) scales, and its step-by-step decomposition | `reconstruct.py` | `reconstruct.csv`, `decompose.csv` |
| Monte Carlo stability of both bootstraps (2,000 resamples; four further seeds) | `boot_stability.py` | `boot_stability_weekly.csv`, `boot_stability_hourly.csv` |
| Eid and lockdown checks, event bootstrap and placebo-based sensitivity intervals | `events_checks.py`, `events_tracers.py` | `eid_checks.csv`, `lockdown_checks.csv`, `events_dependence.csv`, `events_tracers.csv` |
| Boundary-layer interaction tests | `mechanism.py` | `mechanism.csv` |
| AQI sub-index, scenarios and the cap at 500 | `aqi_scenarios.py`, `aqi_cap.py` | `aqi_scenarios.csv`, `aqi_cap.csv` |

The random seeds of every bootstrap are fixed in the scripts.

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
`11_natexp/out/run_all.log`; the full run takes about 1 hour 40 minutes on a 20-thread
laptop. Rerun from a fresh copy of release v2.3, whose analysis code this release shares, 64
of the 65 result files were byte-identical and the 65th, `robust_bootstrap_draws.csv.gz`, differed only in its gzip
header timestamp; `make_supp_tables.py` then rewrote all 30 supplementary tables
byte-identically, and the figure scripts of this release redrew all eight figures
byte-identically from those results.
`11_natexp/code/verify_claims.py` re-derives every number in
the manuscript from `11_natexp/out/`; it needs the manuscript source, which is not part of
this deposit until publication.

## Sources and attribution

- DoE CAMS data: Department of Environment, Bangladesh, via CAMS-BD (CC BY 4.0).
- US Embassy Dhaka PM2.5 and ozone: U.S. Department of State Air Quality Monitoring Program, AirNow,
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
