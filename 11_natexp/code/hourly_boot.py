"""Dependence-aware intervals for the hour-by-hour Thursday-Friday-Saturday profiles.

The pooled hourly profiles of hourly_profiles.py treat stations as independent.  Here every
station-tracer-hour fit of Thursday, Friday and Saturday against Sunday-Wednesday is
repeated on the same joint-date moving-block resamples (14-day calendar blocks, the same
blocks for every station, hour and day; B = 400), and pooled in each resample with the fixed
inverse-variance weights of the main fit, as for the diurnal blocks in robustness.py.

For each region, tracer, day and hour it reports
  pct, lo, hi          the pooled contrast and its pointwise 95% percentile interval
  sim_lo, sim_hi       a simultaneous 95% band for the whole 72-hour curve (Thursday 00 h to
                       Saturday 23 h): b +/- c * sd, where sd is the bootstrap standard
                       deviation at each hour and c the 95th percentile, over resamples, of the
                       largest standardized deviation along the curve (a sup-t band).  The
                       whole curve lies inside it in 95% of resamples, so an hour whose band
                       excludes zero is evidence about the curve, not one of 72 separate tests.
Writes out/hourly_boot.csv.
Usage: python -I hourly_boot.py <FYDP_D root>
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robustness as rb  # noqa: E402
from cams_pooled import dl_pool  # noqa: E402

OUT = os.path.join(HERE, "..", "out")
B = 400
SEED = 20261026
DAYS = {3: "Thu", 4: "Fri", 5: "Sat"}
WORKERS = max(1, min(10, (os.cpu_count() or 2) // 2))


def worker(args):
    """Point estimates of every (frame, day) on every resample of dates."""
    items, draws = args
    out = {}
    for key, s in items:
        for dw in DAYS:
            ss = s[s.dow.isin(rb.WORK + [dw])]
            bid = ((ss.index - rb.T0).days // rb.BLOCK_DAYS).values
            by = {b: np.where(bid == b)[0] for b in np.unique(bid)}
            res = np.full(len(draws), np.nan)
            for i, draw in enumerate(draws):
                idx = np.concatenate([by[b] for b in draw if b in by] or [np.array([], int)])
                if len(idx) < 150:
                    continue
                b = rb.fit(ss.iloc[idx], dw, point_only=True)
                if b is not None:
                    res[i] = b
            out[(key, dw)] = res
    return out


def main(root):
    from hourly_profiles import hourly_frames
    frames = hourly_frames(root)
    S = pd.read_csv(os.path.join(OUT, "hourly_profiles_station.csv"))
    S = S[S.day.isin(DAYS.values())]
    weights, point = {}, {}
    for (reg, y, hh, day), g in S.groupby(["region", "tracer", "hour", "day"]):
        pb, _, tau2, _, k = dl_pool(g.b, g.se)
        weights[(reg, y, hh, day)] = {r.station: 1 / (r.se ** 2 + tau2) for r in g.itertuples()}
        point[(reg, y, hh, day)] = (pb, k)
    nblk = int(np.ceil((pd.Timestamp("2025-12-31") - rb.T0).days / rb.BLOCK_DAYS)) + 1
    draws = np.random.default_rng(SEED).integers(0, nblk, (B, nblk))
    # balance the work by frame size
    items = sorted(frames.items(), key=lambda kv: -len(kv[1]))
    chunks = [[] for _ in range(WORKERS)]
    load = np.zeros(WORKERS)
    for kv in items:
        j = int(np.argmin(load))
        chunks[j].append(kv)
        load[j] += len(kv[1])
    res = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for part in ex.map(worker, [(c, draws) for c in chunks]):
            res.update(part)
    print("fitted %d station-hour-day series on %d resamples" % (len(res), B), flush=True)

    rows, pooled = [], {}
    for (reg, y, hh, day), w in weights.items():
        dw = {v: k for k, v in DAYS.items()}[day]
        mat = np.vstack([res[((st, reg, y, hh), dw)] for st in w])           # stations x draws
        wv = np.array([w[st] for st in w])[:, None]
        ok = np.isfinite(mat)
        pb_draws = np.where(ok.any(axis=0), np.nansum(np.where(ok, mat, 0) * wv, axis=0)
                            / np.where(ok, wv, 0).sum(axis=0), np.nan)
        pb, k = point[(reg, y, hh, day)]
        pooled[(reg, y, day, hh)] = (pb, pb_draws)
        rows.append(dict(region=reg, tracer=y, day=day, hour=hh, k=k, b=pb,
                         se_boot=float(np.nanstd(pb_draws, ddof=1)), n_draws=int(np.isfinite(pb_draws).sum()),
                         lo_b=float(np.nanpercentile(pb_draws, 2.5)), hi_b=float(np.nanpercentile(pb_draws, 97.5))))
    T = pd.DataFrame(rows)
    # simultaneous band over each 72-hour curve
    T["crit"] = np.nan
    order = {"Thu": 0, "Fri": 1, "Sat": 2}
    for (reg, y), g in T.groupby(["region", "tracer"]):
        g = g.assign(o=g.day.map(order) * 24 + g.hour).sort_values("o")
        keys = [(reg, y, r.day, r.hour) for r in g.itertuples()]
        b0 = np.array([pooled[kk][0] for kk in keys])[:, None]
        Dm = np.vstack([pooled[kk][1] for kk in keys])
        sd = np.nanstd(Dm, axis=1, ddof=1)[:, None]
        M = np.nanmax(np.abs(Dm - b0) / sd, axis=0)
        T.loc[g.index, "crit"] = float(np.nanpercentile(M, 95))
    f = lambda v: 100 * (np.exp(v) - 1)  # noqa: E731
    T["pct"], T["lo"], T["hi"] = f(T.b), f(T.lo_b), f(T.hi_b)
    T["sim_lo"] = f(T.b - T.crit * T.se_boot)
    T["sim_hi"] = f(T.b + T.crit * T.se_boot)
    T = T.drop(columns=["lo_b", "hi_b"])
    T = T.sort_values(["region", "tracer", "day", "hour"])
    T.to_csv(os.path.join(OUT, "hourly_boot.csv"), index=False)
    pd.set_option("display.width", 250)
    for (reg, y), g in T.groupby(["region", "tracer"]):
        q = g[g.day == "Fri"]
        print("%s %s Friday (k=%d, crit %.2f)" % (reg, y, q.k.max(), g.crit.iloc[0]))
        print(q[["hour", "pct", "lo", "hi", "sim_lo", "sim_hi"]].round(1).T.to_string(header=False))


if __name__ == "__main__":
    main(sys.argv[1])
