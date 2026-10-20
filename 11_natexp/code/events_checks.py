"""Checks on the two whole-city shocks.

Eid  (the Dhaka-region PM2.5 stations of the main Eid estimate)
  main          the stacked design of events_hourly.py: each Eid window (-15..+18 days)
                with window and day-of-week fixed effects and ERA5 meteorology, treatment
                days -1..+3, lockdown and strike days dropped
  event_boot    the same estimate re-fitted on resamples of the events (B = 1000).  The
                events, not the stations, are the independent units: every station sees
                the same Eid.  Station weights are fixed from the main fit.
  loo_<date>    leaving each event out
  fitr, adha    Eid-ul-Fitr and Eid-ul-Adha separately
  excl_2020_2021  the events of 2019 and 2022-2024 only, outside the pandemic years
  windows       treatment 0..+4 and 0..+8; comparison days restricted to -30..-8 and
                +12..+30, away from the exodus and return; for Eid-ul-Fitr also the
                comparison after Eid only (+12..+30), outside Ramadan
  trajectory    relative days -10..+12 against days -30..-11 and +13..+30, for the city
                sites (US Embassy, BARC, Darussalam) and the exit-corridor sites
                (Narsingdi, Savar), with event-bootstrap intervals.  Savar enters only
                here: its two-hour timing offset does not matter for daily means.
Lockdown  (the Dhaka-region stations with 2020 data)
  main          year and calendar-month fixed effects on the whole record, day of week and
                meteorology (events_hourly.lockdown_effect)
  woy           week-of-year harmonics (three pairs) instead of calendar months
  excl_later    without 31 May 2020 - 31 December 2021, so that later pandemic
                restrictions (a nationwide lockdown from 14 April 2021) do not enter
  nomet         without meteorology
  placebo_<Y>   the same design with a pseudo-lockdown on 26 March - 30 May of 2019, 2022,
                2023 and 2024 (2021 excluded: lockdown from 14 April 2021), the true
                lockdown days dropped.  Their spread measures how far the design strays
                from zero when nothing happened; it is added to the sampling error to give
                an interval that also reflects counterfactual uncertainty.
  trajectory    weekly mean residual, February - July 2020, from the main model without
                the lockdown indicator fitted on all other data, against the same weeks of
                the other years
The AQI sub-index versions use the same designs with the daily PM2.5 sub-index as outcome.

Writes out/eid_checks.csv, out/eid_trajectory.csv, out/lockdown_checks.csv,
out/lockdown_trajectory.csv and out/events_dependence.csv (the event-bootstrap Eid and the
placebo-calibrated lockdown intervals for every diurnal block and the daily mean).
Usage: python -I events_checks.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cams_pooled import DHAKA, dl_pool  # noqa: E402
from cams_shocks import COVID, calendars, embassy_hourly, met_daily  # noqa: E402
from events_hourly import METC, aqi_pm25, design_frames, ols_fe  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
RNG = np.random.default_rng(20261020)
B_EVENTS = 1000
CITY = ["US Embassy", "BARC", "Darussalam"]
CORRIDOR = ["Narsingdi", "Savar"]
PLACEBO_YEARS = [2019, 2022, 2023, 2024]
LATER = (pd.Timestamp("2020-05-31"), pd.Timestamp("2021-12-31"))


def pct(b):
    return 100 * (np.exp(b) - 1)


def point_fe(y, treat, X, fe):
    """Point estimate of the treatment coefficient with fixed effects absorbed."""
    codes, uniq = pd.factorize(fe)
    ng = len(uniq)
    cnt = np.bincount(codes, minlength=ng)
    dm = lambda a: a - (np.bincount(codes, a, ng) / cnt)[codes]
    Z = np.column_stack([treat] + [X[:, j] for j in range(X.shape[1])]).astype(float)
    Zd = np.column_stack([dm(Z[:, j]) for j in range(Z.shape[1])])
    return float(np.linalg.lstsq(Zd, dm(np.asarray(y, float)), rcond=None)[0][0])


# ----------------------------------------------------------------------------- Eid
def eid_stack(s, events, treat=(-1, 3), comp=((-15, 18),), drop_treat_from=None):
    """Stack the windows of `events` (a list of dates, repeats allowed).  Treatment days are
    `treat`; comparison days are those in `comp` outside `treat`.  Returns the frame with a
    window id and a treatment flag."""
    lo = min([treat[0]] + [a for a, _ in comp])
    hi = max([treat[1]] + [b for _, b in comp])
    parts = []
    for i, t in enumerate(events):
        w = s.loc[t + pd.Timedelta(days=lo): t + pd.Timedelta(days=hi)].copy()
        rel = (w.index - t).days
        tr = (rel >= treat[0]) & (rel <= treat[1])
        cp = np.zeros(len(w), bool)
        for a, b in comp:
            cp |= (rel >= a) & (rel <= b)
        w = w[tr | (cp & ~tr)]
        w["treat"] = tr[tr | (cp & ~tr)].astype(float)
        w["win"] = i
        w["rel"] = rel[tr | (cp & ~tr)]
        parts.append(w)
    W = pd.concat(parts) if parts else s.iloc[:0]
    # date order, so that the Newey-West lags run through time as in events_hourly.eid_effect
    # (the events file lists every Eid-ul-Fitr before every Eid-ul-Adha)
    W = W.sort_index(kind="stable")
    return W[(W.covid == 0) & (W.hart == 0)]


def eid_X(W):
    return np.column_stack([W[METC].values] + [(W.dow == d).astype(float).values for d in range(1, 7)])


def eid_fit(s, events, yfun, point=False, **kw):
    W = eid_stack(s, events, **kw)
    if W.treat.sum() < 10 or len(W) < 60:
        return None
    if point:
        return point_fe(yfun(W), W.treat.values, eid_X(W), W.win.values)
    return ols_fe(yfun(W), W.treat.values, eid_X(W), W.win.values)


def pooled(fits):
    """fits: {station: (b, se, n)} -> DerSimonian-Laird pooled (b, se, k) and fixed weights."""
    st = sorted(fits)
    b = np.array([fits[x][0] for x in st])
    se = np.array([fits[x][1] for x in st])
    pb, pse, tau2, i2, k = dl_pool(pd.Series(b), pd.Series(se))
    w = {x: 1 / (fits[x][1] ** 2 + tau2) for x in st}
    return pb, pse, k, w, i2


def eid_trajectory(s, events, yfun, rels=range(-10, 13)):
    """Relative-day effects against days -30..-11 and +13..+30."""
    W = eid_stack(s, events, treat=(-10, 12), comp=((-30, -11), (13, 30)))
    if len(W) < 200:
        return None
    D = np.column_stack([(W.rel == r).astype(float).values for r in rels])
    X = np.column_stack([D, eid_X(W)])
    codes, uniq = pd.factorize(W.win.values)
    ng = len(uniq)
    cnt = np.bincount(codes, minlength=ng)
    dm = lambda a: a - (np.bincount(codes, a, ng) / cnt)[codes]
    Xd = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    coef = np.linalg.lstsq(Xd, dm(np.asarray(yfun(W), float)), rcond=None)[0]
    return dict(zip(rels, coef[:len(list(rels))]))


def eid_checks(daily, eid, outcomes):
    usable = [t for t in eid if not (COVID[0] <= t <= COVID[1])]
    stations = [x for x in daily if x in DHAKA and x != "Savar"]
    rows, traj = [], []
    for oname, yfun in outcomes.items():
        fits = {x: eid_fit(daily[x], eid, yfun) for x in stations}
        fits = {x: f for x, f in fits.items() if f}
        pb, pse, k, w, i2 = pooled(fits)
        rows.append(dict(analysis="main", outcome=oname, b=pb, lo=pb - 1.96 * pse, hi=pb + 1.96 * pse,
                         k=k, n_events=len(usable), I2=i2))
        # event bootstrap with fixed station weights
        draws = []
        for _ in range(B_EVENTS):
            ev = list(RNG.choice(np.array(usable, dtype="datetime64[ns]"), len(usable), replace=True))
            ev = [pd.Timestamp(e) for e in ev]
            bs = {x: eid_fit(daily[x], ev, yfun, point=True) for x in fits}
            bs = {x: v for x, v in bs.items() if v is not None}
            if bs:
                draws.append(sum(w[x] * v for x, v in bs.items()) / sum(w[x] for x in bs))
        draws = np.array(draws)
        rows.append(dict(analysis="event_boot", outcome=oname, b=pb, lo=np.percentile(draws, 2.5),
                         hi=np.percentile(draws, 97.5), se=draws.std(), k=k, n_events=len(usable)))
        # leave one event out
        for t in usable:
            ev = [e for e in eid if e != t]
            f = {x: eid_fit(daily[x], ev, yfun) for x in fits}
            f = {x: v for x, v in f.items() if v}
            if f:
                b2, se2, k2, _, _ = pooled(f)
                rows.append(dict(analysis="loo_" + t.strftime("%Y-%m-%d"), outcome=oname, b=b2,
                                 lo=b2 - 1.96 * se2, hi=b2 + 1.96 * se2, k=k2, n_events=len(usable) - 1))
        # Fitr and Adha
        for lab in ("fitr", "adha"):
            ev = EID_KIND[lab]
            f = {x: eid_fit(daily[x], ev, yfun) for x in fits}
            f = {x: v for x, v in f.items() if v}
            if f:
                b2, se2, k2, _, _ = pooled(f)
                rows.append(dict(analysis=lab, outcome=oname, b=b2, lo=b2 - 1.96 * se2, hi=b2 + 1.96 * se2,
                                 k=k2, n_events=len([e for e in ev if not (COVID[0] <= e <= COVID[1])])))
        # alternative windows
        alts = {"treat_0_4": dict(treat=(0, 4)), "treat_0_8": dict(treat=(0, 8), comp=((-15, 23),)),
                "comp_clean": dict(comp=((-30, -8), (12, 30))),
                "treat_0_4_comp_clean": dict(treat=(0, 4), comp=((-30, -8), (12, 30)))}
        for lab, kw in alts.items():
            f = {x: eid_fit(daily[x], eid, yfun, **kw) for x in fits}
            f = {x: v for x, v in f.items() if v}
            if f:
                b2, se2, k2, _, _ = pooled(f)
                rows.append(dict(analysis=lab, outcome=oname, b=b2, lo=b2 - 1.96 * se2, hi=b2 + 1.96 * se2,
                                 k=k2, n_events=len(usable)))
        # outside the pandemic years (2020 and 2021 Eids fell amid restrictions)
        ev = [e for e in eid if e.year not in (2020, 2021)]
        f = {x: eid_fit(daily[x], ev, yfun) for x in fits}
        f = {x: v for x, v in f.items() if v}
        if f:
            b2, se2, k2, _, _ = pooled(f)
            rows.append(dict(analysis="excl_2020_2021", outcome=oname, b=b2, lo=b2 - 1.96 * se2,
                             hi=b2 + 1.96 * se2, k=k2, n_events=len(ev)))
        f = {x: eid_fit(daily[x], EID_KIND["fitr"], yfun, comp=((12, 30),)) for x in fits}
        f = {x: v for x, v in f.items() if v}
        if f:
            b2, se2, k2, _, _ = pooled(f)
            rows.append(dict(analysis="fitr_post_only", outcome=oname, b=b2, lo=b2 - 1.96 * se2,
                             hi=b2 + 1.96 * se2, k=k2, n_events=len([e for e in EID_KIND["fitr"]
                                                                     if not (COVID[0] <= e <= COVID[1])])))
        if oname != "PM25":
            continue
        # event-time trajectories, PM2.5 only
        for grp, sts in (("city", CITY), ("corridor", CORRIDOR)):
            sts = [x for x in sts if x in daily]
            base = {x: eid_trajectory(daily[x], eid, yfun) for x in sts}
            base = {x: v for x, v in base.items() if v}
            pts = {r: np.mean([base[x][r] for x in base]) for r in range(-10, 13)}
            bd = {r: [] for r in pts}
            for _ in range(500):
                ev = [pd.Timestamp(e) for e in RNG.choice(np.array(usable, dtype="datetime64[ns]"),
                                                           len(usable), replace=True)]
                tb = [eid_trajectory(daily[x], ev, yfun) for x in base]
                tb = [v for v in tb if v]
                if not tb:      # no station has enough data in this resample
                    continue
                for r in pts:
                    bd[r].append(np.mean([v[r] for v in tb]))
            for r in pts:
                traj.append(dict(group=grp, rel=r, pct=pct(pts[r]), lo=pct(np.percentile(bd[r], 2.5)),
                                 hi=pct(np.percentile(bd[r], 97.5)), k=len(base), n_draws=len(bd[r])))
    return pd.DataFrame(rows), pd.DataFrame(traj)


# ----------------------------------------------------------------------------- lockdown
def lock_frame(s, year=2020, spec="main", placebo=False):
    w = s[(s.hart == 0) & ((s.eid == 0) | (s.covid == 1))].copy()
    if placebo:
        w = w[w.covid == 0]
    if spec == "excl_later":
        w = w[(w.index < LATER[0]) | (w.index > LATER[1])]
    ind = ((w.index >= pd.Timestamp("%d-03-26" % year)) & (w.index <= pd.Timestamp("%d-05-30" % year)))
    w["ind"] = ind.astype(float)
    return w


def lock_X(w, spec):
    cols = [] if spec == "nomet" else [w[METC].values]
    cols += [(w.dow == d).astype(float).values for d in range(1, 7)]
    if spec == "woy":
        doy = w.index.dayofyear.values
        cols += [f(2 * np.pi * k * doy / 365.25) for k in (1, 2, 3) for f in (np.sin, np.cos)]
    else:
        cols += [(w.index.month == mo).astype(float) for mo in range(2, 13)]
    return np.column_stack(cols)


def lock_fit(s, yfun, year=2020, spec="main", placebo=False, point=False):
    w = lock_frame(s, year, spec, placebo)
    inY = w.index.year == year
    if w.ind.sum() < 15 or (inY & (w.ind == 0)).sum() < 60 or (~inY).sum() < 300:
        return None
    X = lock_X(w, spec)
    if point:
        return point_fe(yfun(w), w.ind.values, X, w.year.values)
    return ols_fe(yfun(w), w.ind.values, X, w.year.values)


def lockdown_checks(daily, outcomes):
    stations = [x for x in daily if x in DHAKA and x != "Savar"]
    rows, traj = [], []
    for oname, yfun in outcomes.items():
        base = {x: lock_fit(daily[x], yfun) for x in stations}
        base = {x: f for x, f in base.items() if f}
        pb, pse, k, w, i2 = pooled(base)
        res = {"main": (pb, pse, k)}
        for spec in ("woy", "excl_later", "nomet"):
            f = {x: lock_fit(daily[x], yfun, spec=spec) for x in base}
            f = {x: v for x, v in f.items() if v}
            if f:
                b2, se2, k2, _, _ = pooled(f)
                res[spec] = (b2, se2, k2)
        plac = {}
        for yr in PLACEBO_YEARS:
            f = {x: lock_fit(daily[x], yfun, year=yr, placebo=True) for x in base}
            f = {x: v for x, v in f.items() if v}
            if len(f) >= 2:
                b2, se2, k2, _, _ = pooled(f)
                res["placebo_%d" % yr] = (b2, se2, k2)
                plac[yr] = b2
        sd = float(np.std(list(plac.values()), ddof=1)) if len(plac) >= 2 else np.nan
        for name, (b2, se2, k2) in res.items():
            rows.append(dict(analysis=name, outcome=oname, b=b2, lo=b2 - 1.96 * se2, hi=b2 + 1.96 * se2,
                             se=se2, k=k2, I2=i2 if name == "main" else np.nan))
        tot = np.sqrt(pse ** 2 + sd ** 2)
        rows.append(dict(analysis="main_with_placebo_spread", outcome=oname, b=pb, lo=pb - 1.96 * tot,
                         hi=pb + 1.96 * tot, se=tot, k=k, placebo_sd=sd,
                         placebo_mean=float(np.mean(list(plac.values())))))
        if oname != "PM25":
            continue
        # weekly residual trajectory: main model without the indicator, fitted outside
        # February - July 2020, applied to every year's February - July
        for x in base:
            s = daily[x]
            wfit = s[(s.hart == 0) & (s.eid == 0)].copy()
            hold = (wfit.index >= "2020-02-01") & (wfit.index <= "2020-07-31")
            fitset = wfit[~hold]
            X = lock_X(fitset, "main")
            codes, uniq = pd.factorize(fitset.year.values)
            ng = len(uniq)
            cnt = np.bincount(codes, minlength=ng)
            ym = np.asarray(yfun(fitset), float)
            means_y = np.bincount(codes, ym, ng) / cnt
            Xm = np.vstack([np.bincount(codes, X[:, j], ng) / cnt for j in range(X.shape[1])]).T
            beta = np.linalg.lstsq(X - Xm[codes], ym - means_y[codes], rcond=None)[0]
            yfe = dict(zip(uniq, means_y - Xm @ beta))
            allw = wfit[(wfit.index.month >= 2) & (wfit.index.month <= 7)]
            allw = allw[allw.year.isin(list(yfe))]
            pred = lock_X(allw, "main") @ beta + allw.year.map(yfe).values
            resid = np.asarray(yfun(allw), float) - pred
            t = pd.DataFrame({"station": x, "year": allw.year.values,
                              "week": allw.index.isocalendar().week.values.astype(int), "resid": resid})
            traj.append(t)
    T = pd.concat(traj)
    T = T.groupby(["year", "week", "station"]).resid.mean().reset_index()
    T = T.groupby(["year", "week"]).resid.mean().reset_index()
    T["pct"] = pct(T.resid)
    return pd.DataFrame(rows), T


def block_rows(blocks, eid, yfun):
    """Eid (event bootstrap) and lockdown (placebo-calibrated) intervals for each diurnal block."""
    usable = [t for t in eid if not (COVID[0] <= t <= COVID[1])]
    rows = []
    units = sorted({u for v in blocks.values() for u in v})
    for u in units:
        fr = {x: v[u] for x, v in blocks.items() if u in v and x in DHAKA and x != "Savar"}
        fits = {x: eid_fit(s, eid, yfun) for x, s in fr.items()}
        fits = {x: f for x, f in fits.items() if f}
        if len(fits) >= 2:
            pb, pse, k, w, _ = pooled(fits)
            draws = []
            for _ in range(B_EVENTS):
                ev = [pd.Timestamp(e) for e in RNG.choice(np.array(usable, dtype="datetime64[ns]"),
                                                           len(usable), replace=True)]
                bs = {x: eid_fit(fr[x], ev, yfun, point=True) for x in fits}
                bs = {x: v for x, v in bs.items() if v is not None}
                if bs:
                    draws.append(sum(w[x] * v for x, v in bs.items()) / sum(w[x] for x in bs))
            rows.append(dict(event="Eid", unit=u, b=pb, lo=np.percentile(draws, 2.5),
                             hi=np.percentile(draws, 97.5), k=k, interval="event bootstrap"))
        base = {x: lock_fit(s, yfun) for x, s in fr.items()}
        base = {x: f for x, f in base.items() if f}
        if len(base) >= 2:
            pb, pse, k, _, _ = pooled(base)
            plac = []
            for yr in PLACEBO_YEARS:
                f = {x: lock_fit(fr[x], yfun, year=yr, placebo=True) for x in base}
                f = {x: v for x, v in f.items() if v}
                if len(f) >= 2:
                    plac.append(pooled(f)[0])
            tot = np.sqrt(pse ** 2 + np.var(plac, ddof=1)) if len(plac) >= 2 else np.nan
            rows.append(dict(event="Lockdown", unit=u, b=pb, lo=pb - 1.96 * tot, hi=pb + 1.96 * tot, k=k,
                             interval="model + placebo spread"))
    return pd.DataFrame(rows)


EID_KIND = {}


def main(root):
    metd, metb = met_daily(root)
    hart, eid, eid_days = calendars(root)
    ev = pd.read_csv(os.path.join(root, "Traffic", "data_availability", "02_event_calendars", "eid_dates.csv"),
                     parse_dates=["date"])
    EID_KIND["fitr"] = ev[ev.festival.str.contains("Fitr", case=False)].date.tolist()
    EID_KIND["adha"] = ev[~ev.festival.str.contains("Fitr", case=False)].date.tolist()
    val = pd.read_csv(os.path.join(OUT, "cams_validation.csv"))
    ok_pm = set(val[(val.daily_r_embassy >= 0.6) & (val.best_lag_h.abs() <= 1)].station)
    cams = pd.read_csv(os.path.join(DATA, "cams_hourly_qc.csv.gz"), parse_dates=["time"])
    cams = cams[cams.station.isin(ok_pm | {"Savar"}) & (cams.time >= "2016-01-01")]
    emb = embassy_hourly(root)
    allh = pd.concat([cams[["station", "time", "PM25"]], emb], ignore_index=True)
    allh = allh[allh.time < "2026-01-01"]
    daily, blocks = {}, {}
    for st, h in allh.groupby("station"):
        if st not in DHAKA:
            continue
        for u, s in design_frames(h, metd, metb, hart, eid, eid_days):
            if u == "daily":
                daily[st] = s
            if u == "daily" or (isinstance(u, str) and " " in u):
                blocks.setdefault(st, {})[u] = s
    outcomes = {"PM25": lambda s: np.log(s.PM25.values), "AQI": lambda s: aqi_pm25(s.PM25.values)}
    E, ET = eid_checks(daily, eid, outcomes)
    for c in ("b", "lo", "hi"):
        E[c + "_pct"] = np.where(E.outcome == "PM25", pct(E[c]), np.nan)
    E.to_csv(os.path.join(OUT, "eid_checks.csv"), index=False)
    ET.to_csv(os.path.join(OUT, "eid_trajectory.csv"), index=False)
    L, LT = lockdown_checks(daily, outcomes)
    for c in ("b", "lo", "hi"):
        L[c + "_pct"] = np.where(L.outcome == "PM25", pct(L[c]), np.nan)
    L.to_csv(os.path.join(OUT, "lockdown_checks.csv"), index=False)
    LT.to_csv(os.path.join(OUT, "lockdown_trajectory.csv"), index=False)
    U = block_rows(blocks, eid, outcomes["PM25"])
    for c in ("b", "lo", "hi"):
        U[c + "_pct"] = pct(U[c])
    U.to_csv(os.path.join(OUT, "events_dependence.csv"), index=False)
    print(U.round(2).to_string(index=False))
    pd.set_option("display.width", 220)
    print(E.round(3).to_string(index=False))
    print(ET[ET.rel.isin([-5, -2, -1, 0, 1, 2, 4, 6, 8, 10, 12])].round(1).to_string(index=False))
    print(L.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
