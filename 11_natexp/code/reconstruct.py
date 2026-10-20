"""The daily Friday change reconstructed from the block contrasts, on a common sample, and
what each step contributes.

For each station and sample:
  common sample  non-event dates on which all four block means (each from >= 4 valid hours)
                 and the daily meteorology are available.  The daily outcome is the
                 equal-block mean, the mean of the four block means; on the "complete"
                 sample (all 24 hours valid) it equals the mean of all hours.
  log scale      the main weekly design (log outcome, OLS, year-month fixed effects,
                 meteorology), one fit per block and one for the log of the daily outcome.
                 Reconstruction: the counterfactual working-day block mean on each Friday is
                 the observed one divided by exp(beta_block), a residual-preserving
                 multiplicative counterfactual (an assumption about how the counterfactual is
                 built, not an assumption-free retransformation); the station change is
                 sum_Fridays(daily outcome) / sum_Fridays(counterfactual daily outcome) - 1.
                 The direct comparator exp(beta_daily) - 1 is a geometric-mean contrast; under
                 the same residual-preserving construction it is also the ratio of the Friday
                 total to its counterfactual total, so the two are compared on that footing.
  arithmetic     the same designs fitted by Poisson pseudo-maximum likelihood (PPML) on the
  (ppml)         concentration itself, with the year-month effects profiled out: exp(beta)
                 is then a ratio of conditional arithmetic means.  Because Friday is a
                 regressor, the fitted Friday total equals the observed Friday total, so the
                 block reconstruction is the model-implied ratio of the Friday total to the
                 working-day total at the Fridays' own covariates, with no retransformation,
                 and the direct PPML contrast is the same ratio for the daily outcome.  Block
                 and daily models then estimate one quantity under two mean models.
Decomposition of the reconstruction (log scale and ppml; m_b = exp(beta_b)):
  affected       (m_afternoon + m_evening)/2 - 1     the change in the hours it occurs
  partial day    (m_afternoon + m_evening + 2)/4 - 1 the same with night and morning at zero
  equal weight   (sum_b m_b)/4 - 1                   all four block contrasts, equal weights
  concentration  sum_b w_b m_b - 1                   the reconstruction: w_b is the block's
                                                     share of the counterfactual Friday total
  cancellation = equal weight - partial day; weighting = concentration - equal weight.
Every quantity is a station-level log-ratio pooled with the fixed inverse-variance weights of
the direct log-scale fit, and every step is repeated inside a joint-date moving-block
bootstrap (B = 400, 14-day blocks), which also gives intervals for the differences.

Writes out/reconstruct.csv and out/decompose.csv.
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
RNG_COMPLETE = np.random.default_rng(20261025)
QUANT = ["ols_direct", "ols_conc", "ols_equal", "ols_ae", "ols_partial",
         "ppml_direct", "ppml_conc", "ppml_equal", "ppml_ae", "ppml_partial"]


def ppml(s, dw, y="PM25"):
    """Friday (dw) coefficient of a Poisson pseudo-likelihood fit of the concentration on the
    day indicator and meteorology, year-month effects profiled out (Newton on the profiled
    likelihood).  Same comparison days and minimum sizes as robustness.fit."""
    comp = [w for w in rb.WORK if w != dw]
    s = s[s.dow.isin(comp + [dw])]
    if (s.dow == dw).sum() < 30 or len(s) < 150:
        return None
    codes, uniq = pd.factorize(s["ym"])
    ng = len(uniq)
    yv = s[y].values.astype(float)
    X = np.column_stack([(s.dow == dw).astype(float).values] + [s[c].values for c in rb.METC])
    X = X - X.mean(axis=0) * np.r_[0.0, np.ones(X.shape[1] - 1)]       # centre the controls
    sy = np.bincount(codes, yv, ng)
    beta = np.zeros(X.shape[1])

    def profile(b):
        e = np.exp(X @ b)
        a = np.log(sy / np.bincount(codes, e, ng))
        mu = e * np.exp(a)[codes]
        return mu, float(np.sum(yv * np.log(mu) - mu))

    mu, ll = profile(beta)
    for _ in range(100):
        sw = np.bincount(codes, mu, ng)
        xm = np.vstack([np.bincount(codes, mu * X[:, j], ng) / sw for j in range(X.shape[1])]).T
        Xc = X - xm[codes]
        H = Xc.T @ (Xc * mu[:, None])
        g = X.T @ (yv - mu)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            mu_new, ll_new = profile(beta + t * step)
            if ll_new >= ll - 1e-10 or t < 1e-6:
                break
            t /= 2
        beta = beta + t * step
        mu, ll = mu_new, ll_new
        if np.max(np.abs(t * step)) < 1e-10:
            break
    return float(beta[0])


def station_data(frames, st, reg, dates=None):
    """Common-sample block frames and the daily frame for one station's PM2.5."""
    blocks = {b: frames.get((st, reg, "PM25", b)) for b in BLOCK_NAMES}
    daily = frames.get((st, reg, "PM25", "daily"))
    if any(v is None for v in blocks.values()) or daily is None:
        return None
    common = set(daily.index)
    for v in blocks.values():
        common &= set(v.index)
    if dates is not None:
        common &= dates
    common = sorted(common)
    blk = {b: v.loc[common] for b, v in blocks.items()}
    d = daily.loc[common].copy()
    d["PM25"] = np.mean([blk[b]["PM25"].values for b in BLOCK_NAMES], axis=0)
    d["ly"] = np.log(d["PM25"])
    return blk, d


def recon_once(blk, d, rows=None):
    """Station-level log-ratios of every quantity in QUANT (rows: positional resample).  None
    when a log-scale fit fails; arithmetic-scale entries are NaN when a PPML fit fails."""
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
    yb = {b: blk[b]["PM25"].values[fri] for b in BLOCK_NAMES}
    out = {}
    for scale, bet, direct in (("ols", betas, rd), ("ppml", None, None)):
        if scale == "ppml":
            bet = {b: ppml(v, 4) for b, v in blk.items()}
            direct = ppml(d, 4)
            if direct is None or any(v is None for v in bet.values()):
                out.update({k: np.nan for k in QUANT if k.startswith("ppml")})
                continue
        m = {b: np.exp(bet[b]) for b in BLOCK_NAMES}
        cf = {b: yb[b].sum() / m[b] for b in BLOCK_NAMES}            # counterfactual block totals
        ma, me = m["afternoon 12-17"], m["evening 18-23"]
        out[scale + "_direct"] = direct
        out[scale + "_conc"] = np.log(sum(yb[b].sum() for b in BLOCK_NAMES) / sum(cf.values()))
        out[scale + "_equal"] = np.log(np.mean(list(m.values())))
        out[scale + "_ae"] = np.log((ma + me) / 2)
        out[scale + "_partial"] = np.log((ma + me + 2) / 4)
    return out


def complete_dates(allh):
    """Per station, the dates with all 24 hours of PM2.5 valid."""
    h = allh[["station", "time", "PM25"]].dropna()
    h = h[h.PM25 > 0].drop_duplicates(["station", "time"])
    n = h.assign(date=h.time.dt.normalize()).groupby(["station", "date"]).size()
    return {st: set(g.index.get_level_values(1)) for st, g in n[n == 24].groupby(level=0)}


def run_sample(frames, reg, months, rng, dates_of=None):
    stations = sorted({k[0] for k in frames if k[1] == reg and k[2] == "PM25"})
    data, pts = {}, {}
    for st in stations:
        sd = station_data(frames, st, reg, None if dates_of is None else dates_of.get(st, set()))
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
        pts[st] = (full[0], full[1], pr)
    if len(pts) < 2:
        return None
    g = pd.DataFrame([(st, b, se) for st, (b, se, _) in pts.items()], columns=["st", "b", "se"])
    _, _, tau2, _, _ = dl_pool(g.b, g.se)
    w = {st: 1 / (se ** 2 + tau2) for st, (_, se, _) in pts.items()}

    def pool(vals):
        ok = {s: v for s, v in vals.items() if np.isfinite(v)}
        return sum(w[s] * v for s, v in ok.items()) / sum(w[s] for s in ok) if ok else np.nan

    point = {q: pool({s: pts[s][2][q] for s in pts}) for q in QUANT}
    # joint-date bootstrap over the same 14-day calendar blocks for every station
    nblk = int(np.ceil((pd.Timestamp("2025-12-31") - rb.T0).days / rb.BLOCK_DAYS)) + 1
    byblock = {}
    for st, (blk, d) in data.items():
        bid = ((d.index - rb.T0).days // rb.BLOCK_DAYS).values
        byblock[st] = {b: np.where(bid == b)[0] for b in np.unique(bid)}
    draws = []
    for _ in range(B):
        draw = rng.integers(0, nblk, nblk)
        res = {}
        for st, (blk, d) in data.items():
            idx = np.concatenate([byblock[st][b] for b in draw if b in byblock[st]] or [np.array([], int)])
            if len(idx) < 150:
                continue
            r = recon_once(blk, d, idx)
            if r is not None:
                res[st] = r
        if res:
            draws.append({q: pool({s: res[s][q] for s in res}) for q in QUANT})
    D = pd.DataFrame(draws)
    n_fri = int(sum((d.dow == 4).sum() for _, d in data.values()))
    return dict(k=len(pts), n_fridays=n_fri, point=point, draws=D)


def pct(v):
    return 100 * (np.exp(v) - 1)


def rows_for(reg, sample, res):
    P, D = res["point"], res["draws"]
    ci = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))  # noqa: E731
    rec, dec = [], []
    for scale in ("ols", "ppml"):
        r, dr = pct(P[scale + "_conc"]), pct(D[scale + "_conc"])
        c, dc = pct(P[scale + "_direct"]), pct(D[scale + "_direct"])
        rec.append(dict(region=reg, season=sample, scale={"ols": "log", "ppml": "arithmetic"}[scale],
                        k=res["k"], n_fridays=res["n_fridays"],
                        recon_pct=r, recon_lo=ci(dr)[0], recon_hi=ci(dr)[1],
                        direct_pct=c, direct_lo=ci(dc)[0], direct_hi=ci(dc)[1],
                        diff_pp=r - c, diff_lo=ci(dr - dc)[0], diff_hi=ci(dr - dc)[1]))
        row = dict(region=reg, season=sample, scale={"ols": "log", "ppml": "arithmetic"}[scale], k=res["k"])
        vals = {}
        for q in ("ae", "partial", "equal", "conc"):
            vals[q] = (pct(P[scale + "_" + q]), pct(D[scale + "_" + q]))
            row[q + "_pct"] = vals[q][0]
            row[q + "_lo"], row[q + "_hi"] = ci(vals[q][1])
        for name, a, b in (("cancel", "equal", "partial"), ("weight", "conc", "equal"),
                           ("partday", "partial", "ae")):
            row[name + "_pp"] = vals[a][0] - vals[b][0]
            row[name + "_lo"], row[name + "_hi"] = ci(vals[a][1] - vals[b][1])
        dec.append(row)
    return rec, dec


def main(root):
    frames, allh = rb.load(root)[:2]
    rec, dec = [], []
    # the seasonal samples first, in their original order and with their original random
    # stream, so those results are unchanged; the complete-hours sample after them
    for reg in ("Dhaka", "national"):
        for season, months in SEASONS.items():
            res = run_sample(frames, reg, months, RNG)
            if res is None:
                continue
            a, b = rows_for(reg, season, res)
            rec += a
            dec += b
            print(pd.DataFrame(a).round(2).to_string(index=False), flush=True)
    cd = complete_dates(allh)
    for reg in ("Dhaka", "national"):
        res = run_sample(frames, reg, SEASONS["all"], RNG_COMPLETE, cd)
        if res is None:
            continue
        a, b = rows_for(reg, "complete", res)
        rec += a
        dec += b
        print(pd.DataFrame(a).round(2).to_string(index=False), flush=True)
    R = pd.DataFrame(rec)
    R.to_csv(os.path.join(OUT, "reconstruct.csv"), index=False)
    Dc = pd.DataFrame(dec)
    Dc.to_csv(os.path.join(OUT, "decompose.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(2).to_string(index=False))
    print(Dc.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
