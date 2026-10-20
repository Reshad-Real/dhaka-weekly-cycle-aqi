"""Direct tests of the boundary-layer interpretation.

For the afternoon and evening blocks of PM2.5, Dhaka region and national, per station:
  vc_interaction   Friday x (log ventilation coefficient of the block, centred) added to the
                   main weekly design, whose controls already contain the log ventilation
                   coefficient and its square (so its main effect is in the model)
  blh_interaction  Friday x (log boundary-layer height of the block, centred), with the centred
                   log boundary-layer height also added as a main effect
                   Both are reported per doubling of the variable in two ways: 100 delta ln2,
                   the change of the Friday contrast on the log scale (approximately a change
                   in percentage points), and on the response scale, the exact change of the
                   percent contrast at the centred reference, 100 exp(beta) (2^delta - 1)
  tercile_diff     the Friday effect in the lowest minus the highest tercile of the block's
                   ventilation coefficient (station-specific terciles)
  lagged_met       the Friday effect with the previous block's and the previous day's log
                   ventilation coefficient added as controls
  carryover        the Friday effect with the previous block's log concentration added: a
                   sensitivity check on carry-over from earlier hours, not a model of a
                   sustained change (the previous block can itself reflect Friday activity),
                   so no long-run effect is derived
Interactions and the tercile difference are pooled with fixed inverse-variance weights and
given joint-date moving-block bootstrap intervals (B = 400, 14-day blocks); the lagged and
carryover models are pooled by DerSimonian-Laird.  A positive interaction means the Friday
reduction is smaller when the air is better ventilated.

Writes out/mechanism.csv.
Usage: python -I mechanism.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robustness as rb  # noqa: E402
from cams_pooled import dl_pool  # noqa: E402
from cams_shocks import BLOCKS, met_daily  # noqa: E402

OUT = os.path.join(HERE, "..", "out")
ORD = {b: i for i, b in enumerate(BLOCKS)}
TEST_BLOCKS = ["afternoon 12-17", "evening 18-23"]
B = 400
RNG = np.random.default_rng(20261022)


def fit_terms(s, dw, extra=(), inter=(), point=False):
    """Main weekly design plus extra controls and treatment interactions.  Returns the
    coefficients of the treatment and of each interaction, with HAC standard errors."""
    comp = [w for w in rb.WORK if w != dw]
    s = s[s.dow.isin(comp + [dw])].dropna(subset=list(extra) + list(inter))
    if (s.dow == dw).sum() < 30 or len(s) < 150:
        return None
    codes, uniq = pd.factorize(s["ym"])
    ng = len(uniq)
    t = (s.dow == dw).astype(float).values
    cols = [t] + [t * s[c].values for c in inter] + [s[c].values for c in rb.METC] + \
        [s[c].values for c in extra]
    X = np.column_stack(cols)
    yd = rb.demean(s["ly"].values, codes, ng)
    Xd = rb.demean(X, codes, ng)
    nterm = 1 + len(inter)
    if point:
        return np.linalg.lstsq(Xd, yd, rcond=None)[0][:nterm]
    r = sm.OLS(yd, Xd).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    n, k = Xd.shape
    corr = np.sqrt((n - k) / max(n - k - ng, 1))
    out = [(float(r.params[j]), float(r.bse[j] * corr)) for j in range(nterm)]
    if extra and "ly_prev" in extra:
        j = nterm + len(rb.METC) + list(extra).index("ly_prev")
        out.append((float(r.params[j]), float(r.bse[j] * corr)))
    return out


def add_columns(frames, root):
    """Block BLH, lagged ventilation and the previous block's concentration."""
    metd, metb = met_daily(root)
    m = pd.read_csv(os.path.join(root, "02_data", "raw", "meteo_dhaka_hourly_filled.csv"), parse_dates=["time"])
    m["date"] = m["time"].dt.normalize()
    m["block"] = m["time"].dt.hour.map({h: b for b, hs in BLOCKS.items() for h in hs})
    blh = m.groupby(["date", "block"]).boundary_layer_height.mean()
    seq = metb[["lvc"]].reset_index()
    seq["ord"] = seq.block.map(ORD)
    seq = seq.sort_values(["date", "ord"])
    seq["lvc_prev"] = seq.lvc.shift(1)
    lvc_prev = seq.set_index(["date", "block"]).lvc_prev
    lvc_prevday = metd.lvc.shift(1, freq="D")
    out = {}
    # sorted: a set's order changes with Python's string hashing from run to run, and the
    # order of summation then changes the last digits of the pooled results
    stations = sorted({(k[0], k[1]) for k in frames if k[2] == "PM25"})
    for st, reg in stations:
        conc = []
        for b in BLOCKS:
            f = frames.get((st, reg, "PM25", b))
            if f is not None:
                conc.append(pd.DataFrame({"date": f.index, "ord": ORD[b], "ly": f.ly.values}))
        if not conc:
            continue
        C = pd.concat(conc).sort_values(["date", "ord"])
        C["slot"] = (C.date - rb.T0).dt.days * 4 + C.ord
        prev = dict(zip(C.slot + 1, C.ly))
        for b in TEST_BLOCKS:
            f = frames.get((st, reg, "PM25", b))
            if f is None:
                continue
            f = f.copy()
            keys = list(zip(f.index, [b] * len(f)))
            f["blh"] = np.log(blh.reindex(keys).values)
            f["lvc_prev"] = lvc_prev.reindex(keys).values
            f["lvc_prevday"] = lvc_prevday.reindex(f.index).values
            slot = (f.index - rb.T0).days * 4 + ORD[b]
            f["ly_prev"] = [prev.get(x, np.nan) for x in slot]
            f["c_vc"] = f.lvc - f.lvc.mean()
            f["c_blh"] = f.blh - f.blh.mean()
            q = f.lvc.quantile([1 / 3, 2 / 3]).values
            f["terc"] = np.where(f.lvc <= q[0], "low", np.where(f.lvc > q[1], "high", "mid"))
            out[(st, reg, b)] = f
    return out


def main(root):
    frames = rb.load(root)[0]
    F = add_columns(frames, root)
    rows = []
    for reg in ("Dhaka", "national"):
        for b in TEST_BLOCKS:
            keys = [k for k in F if k[1] == reg and k[2] == b]
            res = {}
            for k in keys:
                s = F[k]
                st = k[0]
                vi = fit_terms(s, 4, inter=("c_vc",))
                bi = fit_terms(s, 4, extra=("c_blh",), inter=("c_blh",))
                lo = fit_terms(s[s.terc == "low"], 4)
                hi = fit_terms(s[s.terc == "high"], 4)
                lm = fit_terms(s, 4, extra=("lvc_prev", "lvc_prevday"))
                co = fit_terms(s, 4, extra=("ly_prev",))
                if None in (vi, bi, lo, hi, lm, co):
                    continue
                res[st] = dict(vi=vi[1], bi=bi[1], vb=vi[0], bb=bi[0], lo=lo[0], hi=hi[0], lm=lm[0],
                               co=co[0], rho=co[1])
            if len(res) < 2:
                continue

            def pool(name):
                g = pd.DataFrame([(st, v[name][0], v[name][1]) for st, v in res.items()],
                                 columns=["st", "b", "se"])
                pb, pse, tau2, i2, k = dl_pool(g.b, g.se)
                return pb, pse, {r.st: 1 / (r.se ** 2 + tau2) for r in g.itertuples()}, k

            P = {n: pool(n) for n in ("vi", "bi", "vb", "bb", "lo", "hi", "lm", "co", "rho")}
            # bootstrap the interactions and the tercile difference
            nblk = int(np.ceil((pd.Timestamp("2025-12-31") - rb.T0).days / rb.BLOCK_DAYS)) + 1
            byb = {}
            for k in keys:
                if k[0] not in res:
                    continue
                s = F[k][F[k].dow.isin(rb.WORK + [4])]
                bid = ((s.index - rb.T0).days // rb.BLOCK_DAYS).values
                byb[k[0]] = (s, {x: np.where(bid == x)[0] for x in np.unique(bid)})
            D = {"vi": [], "bi": [], "vb": [], "bb": [], "diff": []}
            for _ in range(B):
                draw = RNG.integers(0, nblk, nblk)
                acc = {"vi": {}, "bi": {}, "vb": {}, "bb": {}, "lo": {}, "hi": {}}
                for st, (s, by) in byb.items():
                    idx = np.concatenate([by[x] for x in draw if x in by] or [np.array([], int)])
                    if len(idx) < 150:
                        continue
                    ss = s.iloc[idx]
                    for name, main_, kw in (("vi", "vb", dict(inter=("c_vc",))),
                                            ("bi", "bb", dict(extra=("c_blh",), inter=("c_blh",)))):
                        r = fit_terms(ss, 4, point=True, **kw)
                        if r is not None:
                            acc[name][st] = r[1]
                            acc[main_][st] = r[0]
                    for tn in ("lo", "hi"):
                        r = fit_terms(ss[ss.terc == ("low" if tn == "lo" else "high")], 4, point=True)
                        if r is not None:
                            acc[tn][st] = r[0]
                for name in ("vi", "bi", "vb", "bb"):
                    w = P[name][2]
                    if acc[name]:
                        D[name].append(sum(w[s] * v for s, v in acc[name].items()) / sum(w[s] for s in acc[name]))
                common = set(acc["lo"]) & set(acc["hi"])
                if common:
                    wl, wh = P["lo"][2], P["hi"][2]
                    lo_b = sum(wl[s] * acc["lo"][s] for s in common) / sum(wl[s] for s in common)
                    hi_b = sum(wh[s] * acc["hi"][s] for s in common) / sum(wh[s] for s in common)
                    D["diff"].append(100 * (np.exp(lo_b) - 1) - 100 * (np.exp(hi_b) - 1))
            f = lambda v: 100 * (np.exp(v) - 1)
            ln2 = np.log(2)
            for name, main_, lab in (("vi", "vb", "vc"), ("bi", "bb", "blh")):
                d = np.array(D[name]) * ln2 * 100
                rows.append(dict(test=lab + "_interaction_per_doubling", region=reg, block=b, k=P[name][3],
                                 est=P[name][0] * ln2 * 100, lo=np.percentile(d, 2.5),
                                 hi=np.percentile(d, 97.5), unit="log-scale x 100 per doubling"))
                # the exact change of the percent contrast per doubling, at the centred reference
                ex = lambda bb, dd: 100 * np.exp(bb) * (np.exp(dd * ln2) - 1)  # noqa: E731
                de = ex(np.array(D[main_]), np.array(D[name]))
                rows.append(dict(test=lab + "_change_per_doubling", region=reg, block=b, k=P[name][3],
                                 est=ex(P[main_][0], P[name][0]), lo=np.percentile(de, 2.5),
                                 hi=np.percentile(de, 97.5), unit="pp per doubling",
                                 low=f(P[main_][0])))
            d = np.array(D["diff"])
            rows.append(dict(test="tercile_low_minus_high", region=reg, block=b, k=P["lo"][3],
                             est=f(P["lo"][0]) - f(P["hi"][0]), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                             unit="pp", low=f(P["lo"][0]), high=f(P["hi"][0])))
            for name, lab in (("lm", "friday_with_lagged_met"), ("co", "friday_short_run_carryover")):
                pb, pse, _, k = P[name]
                rows.append(dict(test=lab, region=reg, block=b, k=k, est=f(pb), lo=f(pb - 1.96 * pse),
                                 hi=f(pb + 1.96 * pse), unit="pct"))
            rho = P["rho"][0]
            rows.append(dict(test="carryover_rho", region=reg, block=b, k=P["rho"][3], est=rho,
                             lo=rho - 1.96 * P["rho"][1], hi=rho + 1.96 * P["rho"][1], unit="coef"))
            print(pd.DataFrame(rows[-10:]).round(2).to_string(index=False), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "mechanism.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1])
