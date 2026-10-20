"""The daily Friday change reconstructed from the block effects, on a common sample.

Eq. (1) of the manuscript weights log-scale block effects by concentration shares, mixing
two estimands and four different samples.  This script replaces it with a reconstruction
that keeps one estimand throughout: the ratio of daily arithmetic-mean concentrations on
Fridays to their working-day counterfactual.

For each station and season:
  common sample  non-event dates on which all four block means (each from >= 4 valid hours)
                 and the daily meteorology are available; the daily mean is the mean of the
                 four block means
  block effects  the main weekly design (Friday against Sunday-Wednesday, year-month fixed
                 effects, block meteorology) fitted on that sample, one per block
  reconstruction counterfactual block means on Fridays = observed x exp(-beta_block), which
                 retransforms without a smearing assumption because each Friday keeps its
                 own residual; the station change is
                 sum_Fridays(daily mean) / sum_Fridays(counterfactual daily mean) - 1
  direct         the same design fitted to the log of the same daily mean
Both are pooled after aggregation, across stations, with the same fixed inverse-variance
weights (from the direct fit), and every step is repeated inside a joint-date moving-block
bootstrap (B = 400, 14-day blocks), which also gives an interval for their difference.

Writes out/reconstruct.csv.
Usage: python -I reconstruct.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robustness as rb  # noqa: E402
from cams_pooled import dl_pool  # noqa: E402

OUT = os.path.join(HERE, "..", "out")
BLOCK_NAMES = ["night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23"]
SEASONS = {"all": list(range(1, 13)), "dry": [11, 12, 1, 2, 3, 4], "wet": [5, 6, 7, 8, 9, 10]}
B = 400
RNG = np.random.default_rng(20261021)


def station_data(frames, st, reg):
    """Common-sample block frames and the daily frame for one station's PM2.5."""
    blocks = {b: frames.get((st, reg, "PM25", b)) for b in BLOCK_NAMES}
    daily = frames.get((st, reg, "PM25", "daily"))
    if any(v is None for v in blocks.values()) or daily is None:
        return None
    common = set(daily.index)
    for v in blocks.values():
        common &= set(v.index)
    common = sorted(common)
    blk = {b: v.loc[common] for b, v in blocks.items()}
    d = daily.loc[common].copy()
    d["PM25"] = np.mean([blk[b]["PM25"].values for b in BLOCK_NAMES], axis=0)
    d["ly"] = np.log(d["PM25"])
    return blk, d


def recon_once(blk, d, rows=None):
    """Reconstructed and direct log-ratios for one station (rows: positional resample)."""
    if rows is not None:
        blk = {b: v.iloc[rows] for b, v in blk.items()}
        d = d.iloc[rows]
    betas = {}
    for b, v in blk.items():
        r = rb.fit(v, 4, point_only=True)
        if r is None:
            return None
        betas[b] = r
    rd = rb.fit(d, 4, point_only=True)
    if rd is None:
        return None
    fri = (d.dow == 4).values
    obs = np.mean([blk[b]["PM25"].values[fri] for b in BLOCK_NAMES], axis=0)
    cf = np.mean([blk[b]["PM25"].values[fri] * np.exp(-betas[b]) for b in BLOCK_NAMES], axis=0)
    return np.log(obs.sum() / cf.sum()), rd


def main(root):
    frames = rb.load(root)[0]
    rows_out = []
    for reg in ("Dhaka", "national"):
        stations = sorted({k[0] for k in frames if k[1] == reg and k[2] == "PM25"})
        for season, months in SEASONS.items():
            data, fits = {}, {}
            for st in stations:
                sd = station_data(frames, st, reg)
                if sd is None:
                    continue
                blk, d = sd
                keep = d.index.month.isin(months) & d.dow.isin(rb.WORK + [4])
                blk = {b: v[keep] for b, v in blk.items()}
                d = d[keep]
                full = rb.fit(d, 4)
                pr = recon_once(blk, d)
                if full is None or pr is None:
                    continue
                data[st] = (blk, d)
                fits[st] = (full[0], full[1], pr[0])
            if len(fits) < 2:
                continue
            g = pd.DataFrame([(st, b, se) for st, (b, se, _) in fits.items()], columns=["st", "b", "se"])
            _, _, tau2, _, _ = dl_pool(g.b, g.se)
            w = {st: 1 / (se ** 2 + tau2) for st, (_, se, _) in fits.items()}
            W = sum(w.values())
            p_rec = sum(w[s] * fits[s][2] for s in fits) / W
            p_dir = sum(w[s] * fits[s][0] for s in fits) / W
            # joint-date bootstrap over the same 14-day calendar blocks for every station
            T0 = rb.T0
            nblk = int(np.ceil((pd.Timestamp("2025-12-31") - T0).days / rb.BLOCK_DAYS)) + 1
            byblock = {}
            for st, (blk, d) in data.items():
                bid = ((d.index - T0).days // rb.BLOCK_DAYS).values
                byblock[st] = {b: np.where(bid == b)[0] for b in np.unique(bid)}
            draws = []
            for _ in range(B):
                draw = RNG.integers(0, nblk, nblk)
                rec, dirr = {}, {}
                for st, (blk, d) in data.items():
                    idx = np.concatenate([byblock[st][b] for b in draw if b in byblock[st]]
                                         or [np.array([], int)])
                    if len(idx) < 150:
                        continue
                    r = recon_once(blk, d, idx)
                    if r is not None:
                        rec[st], dirr[st] = r
                if rec:
                    ws = sum(w[s] for s in rec)
                    draws.append((sum(w[s] * rec[s] for s in rec) / ws, sum(w[s] * dirr[s] for s in rec) / ws))
            D = np.array(draws)
            f = lambda v: 100 * (np.exp(v) - 1)
            diff = f(D[:, 0]) - f(D[:, 1])
            rows_out.append(dict(region=reg, season=season, k=len(fits),
                                 n_fridays=int(sum((d.dow == 4).sum() for _, d in data.values())),
                                 recon_pct=f(p_rec), recon_lo=np.percentile(f(D[:, 0]), 2.5),
                                 recon_hi=np.percentile(f(D[:, 0]), 97.5),
                                 direct_pct=f(p_dir), direct_lo=np.percentile(f(D[:, 1]), 2.5),
                                 direct_hi=np.percentile(f(D[:, 1]), 97.5),
                                 diff_pp=f(p_rec) - f(p_dir), diff_lo=np.percentile(diff, 2.5),
                                 diff_hi=np.percentile(diff, 97.5)))
            print(rows_out[-1], flush=True)
    R = pd.DataFrame(rows_out)
    R.to_csv(os.path.join(OUT, "reconstruct.csv"), index=False)
    pd.set_option("display.width", 220)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
