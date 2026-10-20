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
  no_imputed  without the gap-filled boundary-layer months (Jan-Jun 2024)
  no_pandemic without 26 March 2020 - 31 December 2021, the lockdown and the later pandemic
              restrictions (including the nationwide lockdowns of 2021)
  balanced    only dates on which >= 70% of the region's stations report
  screen_none, screen_strict   the same fits with no station screens, and with stricter
              thresholds (PM2.5 r >= 0.85; gas rho >= 0.3; any analyzer with CV < 0.2 dropped)
  screen_behaviour  the gas screen judged on instrument behaviour alone: an analyzer is
              dropped when its daily means barely vary (CV < 0.15), without asking whether it
              co-varies with PM2.5
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
from screens import gas_allowed, is_fault, screens  # noqa: E402

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


# Station screens as rule sets.  "main" is the screen every other result uses (screens.py);
# the other two bound how much the result depends on thresholds chosen after inspecting
# the diagnostics.  r: daily PM2.5 correlation with the Embassy and lag: best hourly lag
# (Dhaka-region PM2.5); rho: correlation of a gas analyzer with TROPOMI NO2 (Dhaka region);
# cv: flat-line threshold on the 30-day coefficient of variation (all gas analyzers).
SCREEN_RULES = {
    "main": dict(r=0.6, lag=1, rho=0.2, cv=0.15, r_fault=0.1),
    "screen_none": dict(r=None, lag=None, rho=None, cv=None, r_fault=None),
    "screen_strict": dict(r=0.85, lag=1, rho=0.3, cv=0.2, r_fault=None),
    "screen_behaviour": dict(r=0.6, lag=1, rho=0.2, cv=0.15, r_fault=None),
}


def passes(rule, st, reg, y, val, fl):
    """True if this station's analyzer for y passes the screen `rule`."""
    if st == "US Embassy":
        return y == "PM25"            # the reference monitor; it measures no gases
    if y == "PM25":
        if reg != "Dhaka" or rule["r"] is None:
            return True
        v = val.loc[st]
        return v.daily_r_embassy >= rule["r"] and abs(v.best_lag_h) <= rule["lag"]
    if rule["cv"] is not None and (st, y) in fl and is_fault(fl[(st, y)][0], fl[(st, y)][1],
                                                             rule["cv"], rule["r_fault"]):
        return False
    if reg == "Dhaka" and rule["rho"] is not None:
        return st in val.index and val.loc[st, y + "_rho_tropomi"] >= rule["rho"]
    return True


def load(root):
    metd, metb = met_daily(root)
    hart, eid, eid_days = calendars(root)
    ok_pm, gas_ok, flat = screens()
    val = pd.read_csv(os.path.join(OUT, "cams_validation.csv")).set_index("station")
    flt = pd.read_csv(os.path.join(OUT, "cams_flatline_check.csv"))
    fl = {(r.station, r.tracer): (r.daily_cv30, r.r_pm25) for r in flt.itertuples()}
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[(cams.time >= "2019-01-01") & (cams.time < "2025-01-01")]
    emb = embassy_hourly(root)
    emb = emb[(emb.time >= "2019-01-01") & (emb.time < "2026-01-01")]
    allh = pd.concat([cams, emb], ignore_index=True)
    frames, frames_all, rules_of = {}, {}, {}
    for st, h in allh.groupby("station"):
        reg = "Dhaka" if st in DHAKA else "national"
        db = frame(h, metd, hart, eid_days, how="blocks", metb=metb)
        dd = frame(h, metd, hart, eid_days)
        for y in ("PM25", "NOX", "CO"):
            ok = {name for name, rule in SCREEN_RULES.items() if passes(rule, st, reg, y, val, fl)}
            # the main rule must reproduce screens.py exactly
            main_ok = (st == "US Embassy" or reg != "Dhaka" or st in ok_pm) if y == "PM25" \
                else gas_allowed(st, y, reg, gas_ok, flat)
            assert ("main" in ok) == main_ok, (st, y)
            for blk in ORDER:
                src = dd if blk == "daily" else db[db.block == blk]
                if y not in src or src[y].notna().sum() < 300:
                    continue
                key = (st, reg, y, blk)
                frames_all[key] = prep(src, y)
                rules_of[key] = ok
                if "main" in ok:
                    frames[key] = frames_all[key]
    return frames, allh, metb, (hart, eid_days), frames_all, rules_of


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
    frames, allh, metb, (hart, eid_days), frames_all, rules_of = load(root)
    print("station-block-tracer frames:", len(frames), "(unscreened:", len(frames_all), ")")

    # ---------------------------------------------------------------- variants
    variants = {
        "main": dict(),
        "cluster": dict(cov="cluster"),
        "nomet": dict(met=False),
        "y2019_21": dict(sel=lambda s: s[s.index.year <= 2021]),
        "y2022_25": dict(sel=lambda s: s[s.index.year >= 2022]),
        "dry": dict(sel=lambda s: s[s.index.month.isin([11, 12, 1, 2, 3, 4])]),
        "wet": dict(sel=lambda s: s[s.index.month.isin([5, 6, 7, 8, 9, 10])]),
        # without the 4,368 gap-filled boundary-layer hours (1 January - 30 June 2024)
        "no_imputed": dict(sel=lambda s: s[(s.index < "2024-01-01") | (s.index > "2024-06-30")]),
        # without the lockdown and every later pandemic restriction (2021 had nationwide lockdowns)
        "no_pandemic": dict(sel=lambda s: s[(s.index < "2020-03-26") | (s.index > "2021-12-31")]),
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
    # balanced sample: only dates on which at least 70 percent of the region's stations with
    # this tracer and block report, so the pooled contrast does not mix station sets by date
    present = {}
    for (st, reg, y, blk), s in frames.items():
        present.setdefault((reg, y, blk), []).append(set(s.index))
    keep = {}
    for k, sets in present.items():
        cnt = pd.Series([d for ss in sets for d in ss]).value_counts()
        keep[k] = set(cnt[cnt >= np.ceil(0.7 * len(sets))].index)
    rows = []
    for (st, reg, y, blk), s in frames.items():
        ss = s[s.index.isin(keep[(reg, y, blk)])]
        for dw in (4, 5):
            r = fit(ss, dw)
            if r:
                rows.append(dict(station=st, region=reg, tracer=y, block=blk,
                                 day=DAYN[dw], b=r[0], se=r[1], n=r[2]))
    est = pd.DataFrame(rows)
    est["variant"] = "balanced"
    est_all.append(est)
    print("variant balanced      %d station estimates" % len(est))
    # screening thresholds: the same Friday and Saturday fits on other station sets
    for name in ("screen_none", "screen_strict", "screen_behaviour"):
        rows = []
        for key, s in frames_all.items():
            if name not in rules_of[key]:
                continue
            st, reg, y, blk = key
            for dw in (4, 5):
                r = fit(s, dw)
                if r:
                    rows.append(dict(station=st, region=reg, tracer=y, block=blk,
                                     day=DAYN[dw], b=r[0], se=r[1], n=r[2]))
        est = pd.DataFrame(rows)
        est["variant"] = name
        est_all.append(est)
        print("variant %-13s %d station estimates" % (name, len(est)))
    EST = pd.concat(est_all, ignore_index=True)
    EST.to_csv(os.path.join(OUT, "robust_station_estimates.csv"), index=False)

    tabs = []
    for name in list(variants) + ["balanced", "screen_none", "screen_strict", "screen_behaviour"]:
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
    # Joint-date moving-block bootstrap.  Each draw resamples the same 14-day calendar blocks
    # for every station, day of week and outcome, so the dependence between stations (same
    # dates, same weather) and between Thursday, Friday and Saturday is kept.  Friday rows are
    # computed exactly as before; Thursday, Saturday and the daily PM2.5 AQI sub-index come
    # from the same draws.  Pooling uses fixed inverse-variance weights from the main fit.
    from events_hourly import aqi_pm25
    aqi_frames = {}
    for (st, reg, y, blk), s in frames.items():
        if y == "PM25" and blk == "daily":
            a = s.copy()
            a["ly"] = aqi_pm25(a["PM25"].values)
            aqi_frames[(st, reg, "AQI", "daily")] = a.dropna(subset=["ly"])
    rows = []
    for (st, reg, y, blk), s in aqi_frames.items():
        for dw in (3, 4, 5):
            r = fit(s, dw)
            if r:
                rows.append(dict(station=st, region=reg, tracer=y, block=blk, day=DAYN[dw],
                                 b=r[0], se=r[1], n=r[2]))
    main_all = pd.concat([main, pd.DataFrame(rows)], ignore_index=True)
    allf = dict(frames)
    allf.update(aqi_frames)
    BOOT_DAYS = (4, 3, 5)                      # Friday first: its draws are the published ones
    w_main = {}
    for (reg, y, blk, day), g in main_all[main_all.day.isin([DAYN[d] for d in BOOT_DAYS])].groupby(
            ["region", "tracer", "block", "day"]):
        _, _, tau2, _, _ = dl_pool(g.b, g.se)
        w_main[(reg, y, blk, day)] = {r.station: 1 / (r.se ** 2 + tau2) for r in g.itertuples()}

    def make_cache(block_days, days, keys):
        cache = {}
        for key in keys:
            st, reg, y, blk = key
            for dw in days:
                k = (reg, y, blk, DAYN[dw])
                if k not in w_main or st not in w_main[k]:
                    continue
                s = allf[key]
                s = s[s.dow.isin(WORK + [dw])]
                bid = ((s.index - T0).days // block_days).values
                cache[(key, dw)] = (s, {b: np.where(bid == b)[0] for b in np.unique(bid)})
        return cache

    def run_boot(cache, nb, block_days, rng):
        nblk = int(np.ceil((pd.Timestamp("2025-12-31") - T0).days / block_days)) + 1
        out = []
        for it in range(nb):
            draw = rng.integers(0, nblk, nblk)
            res = {}
            for ((st, reg, y, blk), dw), (s, rows_by) in cache.items():
                idx = np.concatenate([rows_by[b] for b in draw if b in rows_by] or [np.array([], int)])
                if len(idx) < 150:
                    continue
                b = fit(s.iloc[idx], dw, point_only=True)
                if b is not None:
                    res.setdefault((reg, y, blk, DAYN[dw]), []).append((st, b))
            for k, lst in res.items():
                w = w_main[k]
                out.append(dict(it=it, region=k[0], tracer=k[1], block=k[2], day=k[3],
                                b=sum(w[st] * b for st, b in lst) / sum(w[st] for st, b in lst)))
            if (it + 1) % 100 == 0:
                print("bootstrap", block_days, "days", it + 1)
        return pd.DataFrame(out)

    def summarise(BT):
        point = []
        for (reg, y, blk, day), g in main_all.groupby(["region", "tracer", "block", "day"]):
            k = (reg, y, blk, day)
            if k not in w_main:
                continue
            w = w_main[k]
            point.append(dict(region=reg, tracer=y, block=blk, day=day,
                              b_point=sum(w[r.station] * r.b for r in g.itertuples()) / sum(w.values()),
                              se_DL=dl_pool(g.b, g.se)[1], k=len(g)))
        PT = pd.DataFrame(point)
        S = BT.groupby(["region", "tracer", "block", "day"]).b.agg(
            se_boot="std", lo_b=lambda x: np.percentile(x, 2.5),
            hi_b=lambda x: np.percentile(x, 97.5)).reset_index()
        S = PT.merge(S, on=["region", "tracer", "block", "day"])
        S["pct"] = 100 * (np.exp(S.b_point) - 1)
        S["lo_pct"] = 100 * (np.exp(S.lo_b) - 1)
        S["hi_pct"] = 100 * (np.exp(S.hi_b) - 1)
        # the AQI sub-index is modelled in index points, not logs: no percentage
        S.loc[S.tracer == "AQI", ["pct", "lo_pct", "hi_pct"]] = np.nan
        S["z_boot"] = S.b_point / S.se_boot
        S["se_inflation"] = S.se_boot / S.se_DL
        return S

    cache = make_cache(BLOCK_DAYS, BOOT_DAYS, list(allf))
    BT = run_boot(cache, B, BLOCK_DAYS, RNG)
    BT.to_csv(os.path.join(OUT, "robust_bootstrap_draws.csv.gz"), index=False)
    S = summarise(BT)
    # the Friday log-outcome rows keep their published file, unchanged in layout
    S[(S.day == "Fri") & (S.tracer != "AQI")].drop(columns=["day", "k"]).to_csv(
        os.path.join(OUT, "robust_bootstrap.csv"), index=False)
    S.to_csv(os.path.join(OUT, "robust_bootstrap_days.csv"), index=False)
    print("\nJoint-date block bootstrap (B=%d, %d-day blocks)" % (B, BLOCK_DAYS))
    print(S[["region", "tracer", "block", "day", "pct", "lo_pct", "hi_pct", "z_boot", "se_inflation"]]
          .round(2).to_string(index=False))

    # Benjamini-Hochberg on the bootstrap p-values.  Family: every pooled Friday and Saturday
    # test of PM2.5, NOx and CO, both regions, four blocks and the daily mean (60 tests).
    fam = S[S.day.isin(["Fri", "Sat"]) & S.tracer.isin(["PM25", "NOX", "CO"])].copy()
    fam["p_boot"] = 2 * norm.sf(np.abs(fam.z_boot))
    fam = fam.sort_values("p_boot")
    nt = len(fam)
    qb = (fam.p_boot * nt / np.arange(1, nt + 1)).values
    fam["q_boot"] = np.minimum(np.minimum.accumulate(qb[::-1])[::-1], 1)
    fam.to_csv(os.path.join(OUT, "robust_bh_bootstrap.csv"), index=False)
    print("\nBH on bootstrap p-values (%d tests)" % nt)
    print(fam[["region", "tracer", "block", "day", "pct", "z_boot", "p_boot", "q_boot"]].round(4)
          .to_string(index=False))

    # block-length sensitivity, Friday PM2.5
    bl = []
    pm_keys = [k for k in frames if k[2] == "PM25"]
    for blen in (7, 28):
        BTb = run_boot(make_cache(blen, (4,), pm_keys), B, blen, np.random.default_rng(20261016 + blen))
        Sb = summarise(BTb)
        Sb["block_days"] = blen
        bl.append(Sb)
    S14 = S[(S.day == "Fri") & (S.tracer == "PM25")].copy()
    S14["block_days"] = BLOCK_DAYS
    BL = pd.concat([S14] + bl, ignore_index=True)
    BL = BL[BL.day == "Fri"]
    BL.to_csv(os.path.join(OUT, "robust_bootstrap_blocklen.csv"), index=False)
    print("\nBlock-length sensitivity, Friday PM2.5")
    print(BL[["block_days", "region", "block", "pct", "lo_pct", "hi_pct"]].round(2).to_string(index=False))

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
