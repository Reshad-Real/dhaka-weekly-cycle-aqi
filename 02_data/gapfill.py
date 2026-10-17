"""
Boundary layer height gap fill.

The Open-Meteo archive has a single contiguous gap in boundary_layer_height covering
2024-01-02 to 2024-06-30 (181 days, 7.1% of the record).  No available model backs that
window, so the gap is upstream, not a request artefact.  Dropping it would remove a full
winter season and the 2024 political events, both of which the identification design needs.

BLH is reconstructed from the co-located variables that are present, using gradient
boosting with the diurnal and seasonal cycle as explicit features.  Skill is measured by
BLOCKED cross-validation: contiguous 181-day blocks are held out to mimic the real gap,
not random hours, because random-hour CV would be optimistic when neighbouring hours are
nearly identical.  Filled hours are flagged in the output so downstream results can be
recomputed without them.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
FEATURES = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
            "surface_pressure", "wind_speed_10m", "cloud_cover"]


def build_features(df):
    d = df.copy()
    t = d["time"]
    hr = t.dt.hour + t.dt.minute / 60.0
    doy = t.dt.dayofyear
    X = pd.DataFrame(index=d.index)
    for c in FEATURES:
        X[c] = d[c].astype(float)
    X["sin_h"] = np.sin(2 * np.pi * hr / 24)
    X["cos_h"] = np.cos(2 * np.pi * hr / 24)
    X["sin_d"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_d"] = np.cos(2 * np.pi * doy / 365.25)
    X["hour"] = hr
    # 24h rolling context, which carries the synoptic state
    X["t_24"] = d["temperature_2m"].rolling(24, min_periods=1, center=True).mean().values
    X["w_24"] = d["wind_speed_10m"].rolling(24, min_periods=1, center=True).mean().values
    X["dtr"] = (d["temperature_2m"].rolling(24, min_periods=1, center=True).max()
                - d["temperature_2m"].rolling(24, min_periods=1, center=True).min()).values
    return X


def blocked_cv(df, n_blocks=8, block_hours=181 * 24, seed=0):
    """Hold out contiguous blocks the size of the real gap."""
    obs = df["boundary_layer_height"].notna().values
    X = build_features(df)
    y = df["boundary_layer_height"].values
    idx = np.where(obs)[0]
    rng = np.random.default_rng(seed)
    scores = []
    for b in range(n_blocks):
        if len(idx) <= block_hours + 24:
            break
        s = rng.integers(0, len(idx) - block_hours)
        hold = idx[s:s + block_hours]
        tr = np.setdiff1d(idx, hold)
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                          max_depth=8, random_state=seed + b)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[hold])
        a = y[hold]
        ss = 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
        scores.append(dict(block=b, r2=float(ss),
                           rmse=float(np.sqrt(np.mean((a - p) ** 2))),
                           mae=float(np.mean(np.abs(a - p))),
                           bias=float(np.mean(p - a)),
                           obs_mean=float(a.mean()), pred_mean=float(p.mean())))
    return pd.DataFrame(scores)


def fill(df, seed=0):
    X = build_features(df)
    y = df["boundary_layer_height"].values
    obs = ~np.isnan(y)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                      max_depth=8, random_state=seed)
    m.fit(X[obs], y[obs])
    out = df.copy()
    pred = m.predict(X[~obs])
    out.loc[~obs, "boundary_layer_height"] = np.clip(pred, 20.0, 3000.0)
    out["blh_filled"] = (~obs).astype(int)
    return out


if __name__ == "__main__":
    p = os.path.join(RAW, "meteo_dhaka_hourly.csv")
    df = pd.read_csv(p, parse_dates=["time"])
    n_missing = int(df["boundary_layer_height"].isna().sum())
    print("hours missing BLH: %d of %d (%.1f%%)" % (n_missing, len(df), 100 * n_missing / len(df)))

    cv = blocked_cv(df)
    print("\nblocked cross-validation (contiguous 181-day holdouts):")
    print(cv.round(3).to_string(index=False))
    print("\nmean R2 %.3f | mean RMSE %.0f m | mean bias %+.0f m"
          % (cv.r2.mean(), cv.rmse.mean(), cv.bias.mean()))

    out = fill(df)
    out.to_csv(os.path.join(RAW, "meteo_dhaka_hourly_filled.csv"), index=False)
    cv.to_csv(os.path.join(HERE, "..", "06_results", "metrics", "blh_gapfill_cv.csv"),
              index=False)
    print("\nfilled %d hours -> meteo_dhaka_hourly_filled.csv" % n_missing)
    print("flag column blh_filled marks reconstructed hours")
