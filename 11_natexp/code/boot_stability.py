"""Monte Carlo stability of the joint-date bootstrap.

The published intervals use B = 400 resamples drawn with fixed seeds (robustness.py: 20261016,
for the diurnal blocks and the daily mean; hourly_boot.py: 20261026, for the hour-by-hour
curves), and every resample gave a pooled estimate for every contrast.  Here the Friday PM2.5
block and daily-mean intervals, their Benjamini-Hochberg q-values, and the simultaneous bands
of every hourly curve are recomputed in six runs:
  published   the published seed and B = 400; must reproduce the published numbers exactly
  B2000       a new seed and B = 2,000
  seed2-5     four further seeds, B = 400 each
The q-value of a Friday PM2.5 test is recomputed with the run's p-values for the ten Friday
PM2.5 tests and the published p-values for the other 50 tests of the family.
Writes out/boot_stability_weekly.csv and out/boot_stability_hourly.csv.
Usage: python -I boot_stability.py <FYDP_D root>
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robustness as rb  # noqa: E402
import hourly_boot as hb  # noqa: E402
from cams_pooled import dl_pool  # noqa: E402

OUT = os.path.join(HERE, "..", "out")
#        run          weekly seed  hourly seed  B
RUNS = [("published", 20261016, 20261026, 400),
        ("B2000", 20261030, 20261040, 2000),
        ("seed2", 20261031, 20261041, 400),
        ("seed3", 20261032, 20261042, 400),
        ("seed4", 20261033, 20261043, 400),
        ("seed5", 20261034, 20261044, 400)]
FRI = 4
WORKERS = hb.WORKERS


def nblocks():
    return int(np.ceil((pd.Timestamp("2025-12-31") - rb.T0).days / rb.BLOCK_DAYS)) + 1


def weekly_draws(seed, b):
    # robustness.py draws one resample per iteration from its generator; do the same, so the
    # published seed reproduces the published resamples
    rng = np.random.default_rng(seed)
    return np.vstack([rng.integers(0, nblocks(), nblocks()) for _ in range(b)])


def weekly_worker(args):
    """Friday point estimates of each station frame on every resample of dates."""
    items, draws = args
    out = {}
    for key, s in items:
        ss = s[s.dow.isin(rb.WORK + [FRI])]
        bid = ((ss.index - rb.T0).days // rb.BLOCK_DAYS).values
        by = {b: np.where(bid == b)[0] for b in np.unique(bid)}
        res = np.full(len(draws), np.nan)
        for i, draw in enumerate(draws):
            idx = np.concatenate([by[b] for b in draw if b in by] or [np.array([], int)])
            if len(idx) < 150:
                continue
            b = rb.fit(ss.iloc[idx], FRI, point_only=True)
            if b is not None:
                res[i] = b
        out[key] = res
    return out


def chunked(items):
    items = sorted(items, key=lambda kv: -len(kv[1]))
    chunks, load = [[] for _ in range(WORKERS)], np.zeros(WORKERS)
    for kv in items:
        j = int(np.argmin(load))
        chunks[j].append(kv)
        load[j] += len(kv[1])
    return chunks


def pool_draws(mat, wv):
    ok = np.isfinite(mat)
    return np.where(ok.any(axis=0), np.nansum(np.where(ok, mat, 0) * wv, axis=0)
                    / np.where(ok, wv, 0).sum(axis=0), np.nan)


def weekly(root):
    frames = rb.load(root)[0]
    E = pd.read_csv(os.path.join(OUT, "robust_station_estimates.csv"))
    E = E[(E.variant == "main") & (E.tracer == "PM25") & (E.day == "Fri")]
    weights, point = {}, {}
    for (reg, blk), g in E.groupby(["region", "block"]):
        tau2 = dl_pool(g.b, g.se)[2]
        w = {r.station: 1 / (r.se ** 2 + tau2) for r in g.itertuples()}
        weights[(reg, blk)] = w
        point[(reg, blk)] = sum(w[r.station] * r.b for r in g.itertuples()) / sum(w.values())
    items = [(k, s) for k, s in frames.items() if k[2] == "PM25" and k[0] in weights.get((k[1], k[3]), {})]
    sizes = [b for _, _, _, b in RUNS]
    draws = np.vstack([weekly_draws(ws, b) for _, ws, _, b in RUNS])
    res = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for part in ex.map(weekly_worker, [(c, draws) for c in chunked(items)]):
            res.update(part)
    print("weekly: %d station frames on %d resamples" % (len(res), len(draws)), flush=True)

    fam = pd.read_csv(os.path.join(OUT, "robust_bh_bootstrap.csv"))
    rows, start = [], 0
    for (run, ws, _, b), n in zip(RUNS, sizes):
        sl = slice(start, start + n)
        start += n
        R = []
        for (reg, blk), w in weights.items():
            st = sorted(w)
            mat = np.vstack([res[(s, reg, "PM25", blk)][sl] for s in st])
            pd_ = pool_draws(mat, np.array([w[s] for s in st])[:, None])
            se = float(np.nanstd(pd_, ddof=1))
            z = point[(reg, blk)] / se
            R.append(dict(run=run, seed=ws, B=b, region=reg, block=blk, n_draws=int(np.isfinite(pd_).sum()),
                          b_point=point[(reg, blk)], se_boot=se,
                          lo_b=float(np.nanpercentile(pd_, 2.5)), hi_b=float(np.nanpercentile(pd_, 97.5)),
                          z_boot=z, p_boot=2 * norm.sf(abs(z))))
        R = pd.DataFrame(R)
        F = fam.copy()
        for r in R.itertuples():
            F.loc[(F.region == r.region) & (F.tracer == "PM25") & (F.block == r.block) & (F.day == "Fri"),
                  "p_boot"] = r.p_boot
        F = F.sort_values("p_boot")
        nt = len(F)
        qb = (F.p_boot * nt / np.arange(1, nt + 1)).values
        F["q"] = np.minimum(np.minimum.accumulate(qb[::-1])[::-1], 1)
        q = F[(F.tracer == "PM25") & (F.day == "Fri")].set_index(["region", "block"]).q
        R["q_bh"] = [q[(r.region, r.block)] for r in R.itertuples()]
        rows.append(R)
    W = pd.concat(rows, ignore_index=True)
    for c in ("b_point", "lo_b", "hi_b"):
        W[c.replace("b_point", "pct").replace("_b", "_pct")] = 100 * (np.exp(W[c]) - 1)
    # the published seed must reproduce the published intervals and q-values
    P = pd.read_csv(os.path.join(OUT, "robust_bootstrap.csv"))
    P = P[P.tracer == "PM25"].set_index(["region", "block"])
    Q = fam[(fam.tracer == "PM25") & (fam.day == "Fri")].set_index(["region", "block"])
    for r in W[W.run == "published"].itertuples():
        p = P.loc[(r.region, r.block)]
        assert abs(r.lo_pct - p.lo_pct) < 1e-9 and abs(r.hi_pct - p.hi_pct) < 1e-9, (r.region, r.block)
        assert abs(r.q_bh - Q.loc[(r.region, r.block)].q_boot) < 1e-9, (r.region, r.block)
    W.to_csv(os.path.join(OUT, "boot_stability_weekly.csv"), index=False)
    pd.set_option("display.width", 250)
    print(W[["run", "B", "region", "block", "n_draws", "pct", "lo_pct", "hi_pct", "q_bh"]].round(3).to_string(index=False))


def hourly(root):
    from hourly_profiles import hourly_frames
    frames = hourly_frames(root)
    S = pd.read_csv(os.path.join(OUT, "hourly_profiles_station.csv"))
    S = S[S.day.isin(hb.DAYS.values())]
    weights, point = {}, {}
    for (reg, y, hh, day), g in S.groupby(["region", "tracer", "hour", "day"]):
        pb, _, tau2, _, k = dl_pool(g.b, g.se)
        weights[(reg, y, hh, day)] = {r.station: 1 / (r.se ** 2 + tau2) for r in g.itertuples()}
        point[(reg, y, hh, day)] = pb
    sizes = [b for _, _, _, b in RUNS]
    draws = np.vstack([np.random.default_rng(hs).integers(0, nblocks(), (b, nblocks())) for _, _, hs, b in RUNS])
    res = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for part in ex.map(hb.worker, [(c, draws) for c in chunked(list(frames.items()))]):
            res.update(part)
    print("hourly: %d station-hour-day series on %d resamples" % (len(res), len(draws)), flush=True)
    dwof = {v: k for k, v in hb.DAYS.items()}
    order = {"Thu": 0, "Fri": 1, "Sat": 2}
    rows, start = [], 0
    for (run, _, hs, b), n in zip(RUNS, sizes):
        sl = slice(start, start + n)
        start += n
        pooled = {}
        for (reg, y, hh, day), w in weights.items():
            st = sorted(w)
            mat = np.vstack([res[((s, reg, y, hh), dwof[day])][sl] for s in st])
            pooled[(reg, y, day, hh)] = pool_draws(mat, np.array([w[s] for s in st])[:, None])
        for (reg, y) in sorted({(k[0], k[1]) for k in pooled}):
            keys = sorted([k for k in pooled if k[0] == reg and k[1] == y], key=lambda k: order[k[2]] * 24 + k[3])
            b0 = np.array([point[(reg, y, k[3], k[2])] for k in keys])
            Dm = np.vstack([pooled[k] for k in keys])
            sd = np.nanstd(Dm, axis=1, ddof=1)
            M = np.nanmax(np.abs(Dm - b0[:, None]) / sd[:, None], axis=0)
            crit = float(np.nanpercentile(M, 95))
            for k, bb, s_, d in zip(keys, b0, sd, Dm):
                rows.append(dict(run=run, seed=hs, B=b, region=reg, tracer=y, day=k[2], hour=k[3],
                                 n_draws=int(np.isfinite(d).sum()), b=bb, se_boot=s_, crit=crit,
                                 lo_b=float(np.nanpercentile(d, 2.5)), hi_b=float(np.nanpercentile(d, 97.5))))
    T = pd.DataFrame(rows)
    f = lambda v: 100 * (np.exp(v) - 1)  # noqa: E731
    T["pct"], T["lo"], T["hi"] = f(T.b), f(T.lo_b), f(T.hi_b)
    T["sim_lo"], T["sim_hi"] = f(T.b - T.crit * T.se_boot), f(T.b + T.crit * T.se_boot)
    T["sim_excludes_zero"] = (T.sim_hi < 0) | (T.sim_lo > 0)
    T = T.drop(columns=["lo_b", "hi_b"])
    # the published seed must reproduce out/hourly_boot.csv
    H = pd.read_csv(os.path.join(OUT, "hourly_boot.csv")).set_index(["region", "tracer", "day", "hour"])
    for r in T[T.run == "published"].itertuples():
        h = H.loc[(r.region, r.tracer, r.day, r.hour)]
        assert abs(r.crit - h.crit) < 1e-9 and abs(r.sim_lo - h.sim_lo) < 1e-9 and abs(r.lo - h.lo) < 1e-9, \
            (r.region, r.tracer, r.day, r.hour)
    T.to_csv(os.path.join(OUT, "boot_stability_hourly.csv"), index=False)
    for (reg, y), g in T.groupby(["region", "tracer"]):
        for run, q in g.groupby("run", sort=False):
            fr = q[(q.day == "Fri") & q.sim_excludes_zero].hour.tolist()
            print("%-8s %-6s %-9s crit %.3f  Friday hours outside the band: %s" % (reg, y, run, q.crit.iloc[0], fr))


if __name__ == "__main__":
    weekly(sys.argv[1])
    hourly(sys.argv[1])
