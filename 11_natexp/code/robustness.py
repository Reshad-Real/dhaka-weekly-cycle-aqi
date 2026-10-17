"""Robustness of the pooled diurnal Friday result (OUTLINE.md, next steps 1).

The first pooled run (cams_pooled.py) combined station estimates as if they were
independent.  They are not: the Dhaka stations share the same Fridays and the same
weather, so their errors are correlated and the random-effects z-scores are too large.
This script re-estimates the result several ways:

  main        as cams_pooled.py (HAC lag 7), Friday and Saturday vs Sunday-Wednesday
  cluster     standard errors clustered by calendar week instead of HAC
  nomet       year-month fixed effects only, no meteorology
  y2019_21    2019-2021 only            y2022_25   2022-2025 only
  dry         November-April            wet        May-October
  noembassy   Dhaka pool without the US Embassy monitor
  loo_<site>  Dhaka pool leaving one station out
  bootstrap   joint-date moving-block bootstrap (14-day blocks resampled identically for
              every station, B = 400), which keeps the cross-station dependence; the
              pooled estimate uses fixed inverse-variance weights from the main fit
  hourly      single-hour pooled Friday profile with the four working-day placebos
  BH          Benjamini-Hochberg q-values over the main pooled tests

Writes out/robust_pooled.csv, out/robust_bootstrap.csv, out/robust_hourly.csv.
Usage: python -I robustness.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_shocks import BLOCKS, calendars, embassy_hourly, frame, met_daily  # noqa: E402
from cams_pooled import DHAKA, dl_pool  # noqa: E402
from screens import gas_allowed, screens  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
METC = ["lvc", "lvc2", "T", "rh", "lpr", "wd_s", "wd_c"]
WORK = [6, 0, 1, 2]
DAYN = {6: "Sun", 0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat"}
ORDER = list(BLOCKS) + ["daily"]
RNG = np.random.default_rng(20261016)
B = 400
BLOCK_DAYS = 14
T0 = pd.Timestamp("2019-01-01")


def prep(d, y):
    s = d[(~d.event) & d[y].notna() & (d[y] > 0)].copy()
    s["ly"] = np.log(s[y])
    s["lvc2"] = s["lvc"] ** 2
    s["lpr"] = np.log1p(s["pr"])
    s = s.dropna(subset=["ly"] + METC)
    return s.sort_index()


def demean(a, codes, ng):
    a = np.asarray(a, float)
    if a.ndim == 1:
        m = np.bincount(codes, a, ng) / np.bincount(codes, minlength=ng)
        return a - m[codes]
    out = np.empty_like(a)
    cnt = np.bincount(codes, minlength=ng)
    for j in range(a.shape[1]):
        out[:, j] = a[:, j] - (np.bincount(codes, a[:, j], ng) / cnt)[codes]
    return out


def fit(s, dw, met=True, cov="HAC", point_only=False):
    """Effect of day-of-week dw against Sun-Wed (minus dw), year-month FE absorbed."""
    comp = [w for w in WORK if w != dw]
    s = s[s.dow.isin(comp + [dw])]
    if (s.dow == dw).sum() < 30 or len(s) < 150:
        return None
    codes, uniq = pd.factorize(s["ym"])
    ng = len(uniq)
    t = (s.dow == dw).astype(float).values
    X = np.column_stack([t] + ([s[c].values for c in METC] if met else []))
    yd = demean(s["ly"].values, codes, ng)
    Xd = demean(X, codes, ng)
    if point_only:
        b = np.linalg.lstsq(Xd, yd, rcond=None)[0]
        return b[0]
    if cov == "HAC":
        r = sm.OLS(yd, Xd).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    else:
        wk = s.index.to_period("W").astype(str)
        r = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(wk)[0]})
    n, k = Xd.shape
    corr = np.sqrt((n - k) / max(n - k - ng, 1))       # absorbed fixed effects
    return float(r.params[0]), float(r.bse[0] * corr), int(n)


def load(root):
    metd, metb = met_daily(root)
    hart, eid, eid_days = calendars(root)
    ok_pm, gas_ok, flat = screens()
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[(cams.time >= "2019-01-01") & (cams.time < "2025-01-01")]
    emb = embassy_hourly(root)
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2026-01-01")]
    allh = pd.concat([cams, emb], ignore_index=True)
    frames = {}
    for st, h in allh.groupby("station"):
        reg = "Dhaka" if st in DHAKA else "national"
        if reg == "Dhaka" and st != "US Embassy" and st not in ok_pm:
            continue
        db = frame(h, metd, hart, eid_days, how="blocks", metb=metb)
        dd = frame(h, metd, hart, eid_days)
        for y in ("PM25", "NOX", "CO"):
            if y != "PM25" and not gas_allowed(st, y, reg, gas_ok, flat):
                continue
            for blk in ORDER:
                src = dd if blk == "daily" else db[db.block == blk]
                if y not in src or src[y].notna().sum() < 300:
                    continue
                frames[(st, reg, y, blk)] = prep(src, y)
    return frames, allh, metb, (hart, eid_days)


def pooled_table(est, label):
    rows = []
    for (reg, y, blk, day), g in est.groupby(["region", "tracer", "block", "day"]):
        if len(g) < 2:
            continue
        b, se, tau2, i2, k = dl_pool(g.b, g.se)
        rows.append(dict(variant=label, region=reg, tracer=y, block=blk, day=day, b=b, se=se,
                         pct=100 * (np.exp(b) - 1), lo=100 * (np.exp(b - 1.96 * se) - 1),
                         hi=100 * (np.exp(b + 1.96 * se) - 1), z=b / se, tau2=tau2, I2=i2, k=k))
    return pd.DataFrame(rows)


def main(root):
    frames, allh, metb, (hart, eid_days) = load(root)
    print("station-block-tracer frames:", len(frames))

    # ---------------------------------------------------------------- variants
    variants = {
        "main": dict(),
        "cluster": dict(cov="cluster"),
        "nomet": dict(met=False),
        "y2019_21": dict(sel=lambda s: s[s.index.year <= 2021]),
        "y2022_25": dict(sel=lambda s: s[s.index.year >= 2022]),
        "dry": dict(sel=lambda s: s[s.index.month.isin([11, 12, 1, 2, 3, 4])]),
        "wet": dict(sel=lambda s: s[s.index.month.isin([5, 6, 7, 8, 9, 10])]),
    }
    est_all = []
    for name, v in variants.items():
        rows = []
        for (st, reg, y, blk), s in frames.items():
            ss = v["sel"](s) if "sel" in v else s
            days = (4, 5) + ((6, 0, 1, 2, 3) if name == "main" else ())
            for dw in days:
                r = fit(ss, dw, met=v.get("met", True), cov=v.get("cov", "HAC"))
                if r:
                    rows.append(dict(station=st, region=reg, tracer=y, block=blk,
                                     day=DAYN[dw], b=r[0], se=r[1], n=r[2]))
        est = pd.DataFrame(rows)
        est["variant"] = name
        est_all.append(est)
        print("variant %-9s %d station estimates" % (name, len(est)))
    EST = pd.concat(est_all, ignore_index=True)
    EST.to_csv(os.path.join(OUT, "robust_station_estimates.csv"), index=False)

    tabs = []
    for name in variants:
        tabs.append(pooled_table(EST[EST.variant == name], name))
    main = EST[EST.variant == "main"]
    tabs.append(pooled_table(main[main.station != "US Embassy"], "noembassy"))
    dh = sorted(main[(main.region == "Dhaka") & (main.tracer == "PM25")].station.unique())
    for st in dh:
        tabs.append(pooled_table(main[(main.station != st) & (main.region == "Dhaka")],
                                 "loo_" + st.replace(" ", "")))
    P = pd.concat(tabs, ignore_index=True)

    # Benjamini-Hochberg over the main Friday and Saturday pooled tests
    from scipy.stats import norm
    m = P[(P.variant == "main") & P.day.isin(["Fri", "Sat"])].copy()
    m["p"] = 2 * norm.sf(np.abs(m.z))
    m = m.sort_values("p")
    nt = len(m)
    q = (m.p * nt / np.arange(1, nt + 1)).values
    q = np.minimum.accumulate(q[::-1])[::-1]
    m["q_BH"] = np.minimum(q, 1)
    P = P.merge(m[["variant", "region", "tracer", "block", "day", "p", "q_BH"]],
                on=["variant", "region", "tracer", "block", "day"], how="left")
    P.to_csv(os.path.join(OUT, "robust_pooled.csv"), index=False)

    pd.set_option("display.width", 250)
    for reg in ("Dhaka", "national"):
        q = P[(P.region == reg) & (P.tracer == "PM25") & (P.day == "Fri")]
        t = q.pivot(index="variant", columns="block", values="pct")[ORDER]
        print("\n%s PM2.5, Friday, pooled percent by variant" % reg)
        print(t.round(1).to_string())
        print("z:")
        print(q.pivot(index="variant", columns="block", values="z")[ORDER].round(2).to_string())
    print("\nBH q-values (main):")
    print(m[["region", "tracer", "block", "day", "pct", "z", "p", "q_BH"]].round(4).to_string(index=False))

    # ---------------------------------------------------------------- bootstrap
    w_main = {}
    for (reg, y, blk), g in main[main.day == "Fri"].groupby(["region", "tracer", "block"]):
        _, _, tau2, _, _ = dl_pool(g.b, g.se)
        w_main[(reg, y, blk)] = {r.station: 1 / (r.se ** 2 + tau2) for r in g.itertuples()}
    nblk = int(np.ceil((pd.Timestamp("2025-12-31") - T0).days / BLOCK_DAYS)) + 1
    cache = {}
    for key, s in frames.items():
        st, reg, y, blk = key
        if (reg, y, blk) not in w_main or st not in w_main[(reg, y, blk)]:
            continue
        s = s[s.dow.isin(WORK + [4])]
        bid = ((s.index - T0).days // BLOCK_DAYS).values
        rows_by = {b: np.where(bid == b)[0] for b in np.unique(bid)}
        cache[key] = (s, rows_by)
    boots = []
    for it in range(B):
        draw = RNG.integers(0, nblk, nblk)
        res = {}
        for (st, reg, y, blk), (s, rows_by) in cache.items():
            idx = np.concatenate([rows_by[b] for b in draw if b in rows_by] or [np.array([], int)])
            if len(idx) < 150:
                continue
            ss = s.iloc[idx]
            b = fit(ss, 4, point_only=True)
            if b is not None:
                res.setdefault((reg, y, blk), []).append((st, b))
        for k, lst in res.items():
            w = w_main[k]
            num = sum(w[st] * b for st, b in lst)
            den = sum(w[st] for st, b in lst)
            boots.append(dict(it=it, region=k[0], tracer=k[1], block=k[2], b=num / den))
        if (it + 1) % 100 == 0:
            print("bootstrap", it + 1)
    BT = pd.DataFrame(boots)
    point = []
    for (reg, y, blk), g in main[main.day == "Fri"].groupby(["region", "tracer", "block"]):
        w = w_main[(reg, y, blk)]
        point.append(dict(region=reg, tracer=y, block=blk,
                          b_point=sum(w[r.station] * r.b for r in g.itertuples()) / sum(w.values()),
                          se_DL=dl_pool(g.b, g.se)[1]))
    PT = pd.DataFrame(point)
    S = BT.groupby(["region", "tracer", "block"]).b.agg(
        se_boot="std", lo_b=lambda x: np.percentile(x, 2.5), hi_b=lambda x: np.percentile(x, 97.5)).reset_index()
    S = PT.merge(S, on=["region", "tracer", "block"])
    S["pct"] = 100 * (np.exp(S.b_point) - 1)
    S["lo_pct"] = 100 * (np.exp(S.lo_b) - 1)
    S["hi_pct"] = 100 * (np.exp(S.hi_b) - 1)
    S["z_boot"] = S.b_point / S.se_boot
    S["se_inflation"] = S.se_boot / S.se_DL
    S.to_csv(os.path.join(OUT, "robust_bootstrap.csv"), index=False)
    print("\nJoint-date block bootstrap, Friday (B=%d, %d-day blocks)" % (B, BLOCK_DAYS))
    print(S[["region", "tracer", "block", "pct", "lo_pct", "hi_pct", "z_boot", "se_inflation"]]
          .round(2).to_string(index=False))

    # ---------------------------------------------------------------- hourly profile
    rows = []
    for st, h in allh.groupby("station"):
        reg = "Dhaka" if st in DHAKA else "national"
        if not any(k[0] == st and k[2] == "PM25" for k in frames):
            continue
        h = h.copy()
        h["date"] = h["time"].dt.normalize()
        h["hour"] = h["time"].dt.hour
        h = h.drop_duplicates(["date", "hour"])
        tmpl = frames[(st, reg, "PM25", "daily")]
        for hh in range(24):
            blk = [b for b, hs in BLOCKS.items() if hh in hs][0]
            x = h[h.hour == hh].set_index("date")[["PM25"]]
            mb = metb.xs(blk, level="block")
            s = x.join(mb, how="inner")
            s = s.join(tmpl[["dow", "ym", "event"]], how="inner")
            s = s[(~s.event) & (s.PM25 > 0)].copy()
            s["ly"] = np.log(s.PM25)
            s["lvc2"] = s.lvc ** 2
            s["lpr"] = np.log1p(s.pr)
            s = s.dropna(subset=["ly"] + METC).sort_index()
            for dw in (4, 5, 6, 0, 1, 2):
                r = fit(s, dw)
                if r:
                    rows.append(dict(station=st, region=reg, hour=hh, day=DAYN[dw], b=r[0], se=r[1]))
    H = pd.DataFrame(rows)
    hp = []
    for (reg, hh, day), g in H.groupby(["region", "hour", "day"]):
        b, se, tau2, i2, k = dl_pool(g.b, g.se)
        hp.append(dict(region=reg, hour=hh, day=day, pct=100 * (np.exp(b) - 1),
                       lo=100 * (np.exp(b - 1.96 * se) - 1), hi=100 * (np.exp(b + 1.96 * se) - 1), k=k))
    HP = pd.DataFrame(hp)
    band = HP[HP.day.isin(["Sun", "Mon", "Tue", "Wed"])].groupby(["region", "hour"]).pct.agg(
        placebo_min="min", placebo_max="max").reset_index()
    HP = HP.merge(band, on=["region", "hour"])
    HP.to_csv(os.path.join(OUT, "robust_hourly.csv"), index=False)
    print("\nPooled hourly Friday PM2.5 percent")
    print(HP[HP.day == "Fri"].pivot(index="hour", columns="region", values="pct").round(1).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
