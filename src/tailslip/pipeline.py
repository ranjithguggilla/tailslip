"""End-to-end run: features, model selection, calibration, studies, charts, bundle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tailslip import data, explain, models, plots, state
from tailslip import evaluate as E
from tailslip import features as F

SHIPPED_HORIZON = 2.0
HORIZONS = [0.0, 1.0, 2.0, 4.0, 8.0, 24.0]
BOOST_GRID = [
    dict(learning_rate=0.06, max_leaf_nodes=31, max_iter=300, min_samples_leaf=40),
    dict(learning_rate=0.05, max_leaf_nodes=15, max_iter=400, min_samples_leaf=100),
    dict(learning_rate=0.10, max_leaf_nodes=63, max_iter=200, min_samples_leaf=40),
]
# simplest first; the shipped model is the simplest one within one standard error of the best
COMPLEXITY_ORDER = ["base rate", "slot rate", "late-aircraft rule", "logistic", "boosting, no live state", "boosting, full"]


def _r(x, nd=4):
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _fit_predict(pipe, X, y, tr, others):
    pipe.fit(X.loc[tr], y[tr])
    return {k: pipe.predict_proba(X.loc[m])[:, 1] for k, m in others.items()}


def run(raw_dir="data/raw", out_dir="outputs", assets_dir="assets", quick=False, log=print):
    t0 = time.time()
    out, assets = Path(out_dir), Path(assets_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    tables = data.load_tables(raw_dir)
    flights = data.prepare_flights(tables["flights"])
    flights, n_unassigned = data.drop_unassigned(flights)
    split = data.split_by_month(flights)
    y = flights["delayed"].to_numpy()
    tr, va, te = (split == "train").to_numpy(), (split == "validation").to_numpy(), (split == "test").to_numpy()
    indexes, wx = F.build_origin_indexes(flights), F.weather_table(tables["weather"])
    reps = 40 if quick else 300
    M: dict = {}
    M["data"] = {
        "flights": len(flights),
        "cancelled": int(flights["cancelled"].sum()),
        "dropped_without_tail_number": n_unassigned,
        "delay_rate": float(y.mean()),
        "split_rows": {s: int((split == s).sum()) for s in ("train", "validation", "test")},
        "split_delay_rate": {s: float(y[(split == s).to_numpy()].mean()) for s in ("train", "validation", "test")},
        "horizon_hours": SHIPPED_HORIZON,
    }
    X = F.build_features(flights, tables, SHIPPED_HORIZON, indexes, wx)
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    bva = E.DayBlocks.from_dates(flights.loc[va, "sched_date"])
    bte = E.DayBlocks.from_dates(flights.loc[te, "sched_date"])

    # ---- model comparison on the shipped horizon ------------------------------------------
    log("fitting candidates")
    boost_cols = F.ALL_FEATURES
    sched_wx = F.CATEGORICAL + F.SCHEDULE + F.PLANE + F.WEATHER
    grid_scores = []
    best_cfg, best_ll, best_pipe = None, np.inf, None
    for cfg in BOOST_GRID[: (1 if quick else len(BOOST_GRID))]:
        pipe = models.make_boosting(boost_cols, **cfg).fit(Xtr, ytr)
        ll = E.score_block(yva, pipe.predict_proba(Xva)[:, 1])["log_loss"]
        grid_scores.append({**cfg, "val_log_loss": ll})
        if ll < best_ll:
            best_cfg, best_ll, best_pipe = cfg, ll, pipe
    cand = {
        "base rate": models.make_base_rate().fit(Xtr, ytr),
        "slot rate": models.SlotRate().fit(Xtr, ytr),
        "late-aircraft rule": models.RuleLate().fit(Xtr, ytr),
        "logistic": models.make_logistic(boost_cols).fit(Xtr, ytr),
        "boosting, no live state": models.make_boosting(sched_wx, **best_cfg).fit(Xtr, ytr),
        "boosting, full": best_pipe,
    }
    val_p = {k: m.predict_proba(Xva)[:, 1] for k, m in cand.items()}
    test_p = {k: m.predict_proba(Xte)[:, 1] for k, m in cand.items()}
    val_scores = {k: E.score_block(yva, p) for k, p in val_p.items()}
    best_name = min(val_scores, key=lambda k: val_scores[k]["log_loss"])
    chosen = best_name
    ses = {}
    for name in COMPLEXITY_ORDER:
        se = E.paired_logloss_se(bva, yva, val_p[name], val_p[best_name], reps=reps)
        ses[name] = se
        gap = val_scores[name]["log_loss"] - val_scores[best_name]["log_loss"]
        if gap <= se:
            chosen = name
            break
    model = cand[chosen]
    M["boost_grid"] = grid_scores
    M["selection"] = {
        "best_on_validation": best_name,
        "shipped": chosen,
        "rule": "simplest candidate within one bootstrap standard error of the best validation log loss",
        "boost_config": best_cfg,
        "paired_se": ses,
    }
    M["models"] = {}
    for name in COMPLEXITY_ORDER:
        tsc = E.score_block(yte, test_p[name])
        lo, hi = E.bootstrap_ci(bte, lambda i, p=test_p[name]: E.roc_auc(yte[i], p[i]), reps=reps)
        M["models"][name] = {"validation": val_scores[name], "test": tsc, "test_auc_ci": [lo, hi]}

    # ---- calibration -----------------------------------------------------------------------
    raw_va, raw_te = val_p[chosen], test_p[chosen]
    from sklearn.isotonic import IsotonicRegression

    sep = (flights.loc[va, "month"] == 9).to_numpy()
    iso_a = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(raw_va[sep], yva[sep])
    ece_raw_b = E.ece(yva[~sep], raw_va[~sep])
    ece_iso_b = E.ece(yva[~sep], iso_a.predict(raw_va[~sep]))
    iso_full = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(raw_va, yva)
    # The model ships uncalibrated. A calibrator fitted on the validation months passed the
    # September to October check but then made the November to December ECE worse, because
    # validation sits at a 16% delay rate and the test months at 24%. See the README.
    calibrator = None
    cal_va, cal_te = raw_va, raw_te
    M["calibration"] = {
        "shipped_calibrator": "none",
        "october_ece_raw": ece_raw_b,
        "october_ece_isotonic": ece_iso_b,
        "test_ece_raw": E.ece(yte, raw_te),
        "test_ece_isotonic_on_validation": E.ece(yte, iso_full.predict(raw_te)),
        "test_mean_pred": float(raw_te.mean()),
        "test_base_rate": float(yte.mean()),
        "validation_base_rate": float(yva.mean()),
        "validation_mean_pred": float(raw_va.mean()),
    }
    M["shipped_test"] = E.score_block(yte, cal_te)
    M["shipped_test"]["auc_ci"] = list(E.bootstrap_ci(bte, lambda i: E.roc_auc(yte[i], cal_te[i]), reps=reps))
    M["shipped_test"]["log_loss_ci"] = list(
        E.bootstrap_ci(bte, lambda i: E.score_block(yte[i], cal_te[i])["log_loss"], reps=max(reps // 3, 20))
    )

    # ---- alert economics ---------------------------------------------------------------------
    thr, val_best, grid, vals = E.best_threshold(yva, cal_va)
    shipped_alert = E.alert_table(yte, cal_te, thr)
    hind_thr, hind_val, tgrid, tvals = E.best_threshold(yte, cal_te)
    slot_thr, _, _, _ = E.best_threshold(yva, val_p["slot rate"])
    rule_flag = ((Xte["prev_dep_delay"].fillna(0) >= 15) | (Xte["prev_overdue_min"] >= 15)).to_numpy()
    econ = {
        "alert none": 0.0,
        "alert every flight": E.alert_value(yte, np.ones(len(yte), bool)),
        "late-aircraft rule": E.alert_value(yte, rule_flag),
        "slot rate": E.alert_value(yte, test_p["slot rate"] >= slot_thr),
        "tailslip": shipped_alert["net_per_flight"],
    }
    ci_fn = lambda i: E.alert_value(yte[i], cal_te[i] >= thr)  # noqa: E731
    diff_fn = lambda i: (
        E.alert_value(yte[i], cal_te[i] >= thr)
        - E.alert_value(  # noqa: E731
            yte[i], test_p["slot rate"][i] >= slot_thr
        )
    )
    M["economics"] = {
        "assumptions": {
            "value_per_caught_delay": E.VALUE_PER_CAUGHT_DELAY,
            "cost_per_false_alert": E.COST_PER_FALSE_ALERT,
            "flights_per_year": len(flights),
        },
        "threshold_from_validation": thr,
        "test": shipped_alert,
        "net_per_flight": econ,
        "net_ci": list(E.bootstrap_ci(bte, ci_fn, reps=reps)),
        "gain_over_slot_ci": list(E.bootstrap_ci(bte, diff_fn, reps=reps)),
        "hindsight_threshold": hind_thr,
        "hindsight_net_per_flight": hind_val,
        "sensitivity": _sensitivity(yte, cal_te, test_p["slot rate"], [(60, 25), (120, 15), (40, 40), (100, 50)]),
        "per_1000_flights": {k: v * 1000 for k, v in econ.items()},
        "per_year_at_nyc_volume": {k: v * len(flights) for k, v in econ.items()},
    }

    # ---- lead-time curve, hindsight weather, leakage demos -------------------------------------
    log("lead-time studies")
    lead = []
    horizons = [0.0, 2.0, 8.0] if quick else HORIZONS
    for h in horizons:
        Xh = X if h == SHIPPED_HORIZON else F.build_features(flights, tables, h, indexes, wx)
        m = models.make_boosting(boost_cols, **best_cfg).fit(Xh[tr], ytr)
        p = m.predict_proba(Xh[te])[:, 1]
        sc = E.score_block(yte, p)
        lead.append({"hours_before": h, "auc": sc["auc"], "log_loss": sc["log_loss"], "ece": sc["ece"]})
    M["lead_time"] = lead

    studies = {}
    Xhw = F.build_features(flights, tables, SHIPPED_HORIZON, indexes, wx, hindsight_weather=True)
    m = models.make_boosting(boost_cols, **best_cfg).fit(Xhw[tr], ytr)
    studies["hindsight_weather_at_departure"] = E.score_block(yte, m.predict_proba(Xhw[te])[:, 1])
    rng = np.random.default_rng(7)
    rand_tr = rng.random(len(flights)) < 0.7
    m = models.make_boosting(boost_cols, **best_cfg).fit(X[rand_tr], y[rand_tr])
    studies["random_row_split_all_year"] = E.score_block(y[~rand_tr], m.predict_proba(X[~rand_tr])[:, 1])
    M["leakage_studies"] = studies

    abl = {}
    if not quick:
        for gname, cols in F.GROUPS.items():
            keep = [c for c in F.ALL_FEATURES if c not in cols]
            m = models.make_boosting(keep, **best_cfg).fit(Xtr[keep], ytr)
            abl[gname] = E.score_block(yte, m.predict_proba(Xte[keep])[:, 1])
    M["group_ablation"] = abl

    # ---- how far the chain effect reaches ------------------------------------------------------
    d = X.assign(delayed=y)
    bins = [-1e9, -5, 5, 15, 30, 60, 1e9]
    labels = ["early", "on time", "5 to 15 late", "15 to 30 late", "30 to 60 late", "60+ late"]
    known = d[d["prev_dep_delay"].notna()].copy()
    known["band"] = pd.cut(known["prev_dep_delay"], bins, labels=labels)
    chain = known.groupby("band", observed=True)["delayed"].agg(["mean", "size"])
    grounded = d[d["prev_overdue_min"] > 0]
    chain_rows = [{"band": b, "delay_rate": float(r["mean"]), "n": int(r["size"])} for b, r in chain.iterrows()]
    chain_rows.append(
        {"band": "previous leg still on ground", "delay_rate": float(grounded["delayed"].mean()), "n": len(grounded)}
    )
    blind = d["prev_dep_delay"].isna() & (d["prev_overdue_min"] == 0)
    chain_rows.append(
        {"band": "no previous leg known", "delay_rate": float(d.loc[blind, "delayed"].mean()), "n": int(blind.sum())}
    )
    M["chain_effect"] = chain_rows
    M["rotation"] = {
        "median_gap_hours": float(d["rotation_gap_h"].median()),
        "share_previous_leg_same_day_under_12h": float((d["rotation_gap_h"] < 12).mean()),
        "prev_departure_known_share": float(d["prev_dep_delay"].notna().mean()),
    }

    # ---- quantile forecast of the departure delay -------------------------------------------------
    log("quantile forest")
    taus = (0.5, 0.8, 0.9)
    dep = ~flights["cancelled"].to_numpy()
    delay = flights["dep_delay"].to_numpy()
    tr_idx = np.flatnonzero(tr & dep)
    rs = np.random.default_rng(3)
    sub = rs.choice(tr_idx, size=min(len(tr_idx), 15000 if quick else 60000), replace=False)
    qf = models.make_quantile_forest(boost_cols, taus).fit(X.iloc[sub], delay[sub])
    te_dep = np.flatnonzero(te & dep)
    qpred = models.predict_quantiles(qf, X.iloc[te_dep], taus)
    ytd = delay[te_dep]
    slot_q = _slot_quantiles(flights[tr & dep], flights.iloc[te_dep], taus)
    qres = {}
    for j, tau in enumerate(taus):
        qres[str(tau)] = {
            "coverage_forest": float((ytd <= qpred[:, j]).mean()),
            "coverage_slot": float((ytd <= slot_q[:, j]).mean()),
            "pinball_forest": E.pinball(ytd, qpred[:, j], tau),
            "pinball_slot": E.pinball(ytd, slot_q[:, j], tau),
            "median_prediction_min": float(np.median(qpred[:, j])),
        }
    hrs = flights.iloc[te_dep]["hour"].to_numpy()
    cov_by_hour = pd.DataFrame({"hour": hrs, "hit": ytd <= qpred[:, 1]}).groupby("hour")["hit"].agg(["mean", "size"])
    qres["p80_coverage_by_hour"] = {int(h): float(r["mean"]) for h, r in cov_by_hour.iterrows() if r["size"] >= 100}
    qres["n_test_departed"] = int(len(te_dep))
    M["quantiles"] = qres

    # ---- group audit ------------------------------------------------------------------------------
    audit = []
    tf = flights[te].reset_index(drop=True)
    for col, vals_ in (("origin", ["EWR", "JFK", "LGA"]), ("carrier", None)):
        keys = vals_ or tf["carrier"].value_counts().head(8).index.tolist()
        for k in keys:
            mk = (tf[col] == k).to_numpy()
            flag = cal_te[mk] >= thr
            yy = yte[mk]
            audit.append(
                {
                    "group": f"{col}={k}",
                    "n": int(mk.sum()),
                    "base_rate": float(yy.mean()),
                    "mean_pred": float(cal_te[mk].mean()),
                    "auc": E.roc_auc(yy, cal_te[mk]),
                    "alert_rate": float(flag.mean()),
                    "recall": float((flag & (yy == 1)).sum() / max(yy.sum(), 1)),
                }
            )
    M["group_audit"] = audit

    # ---- streaming equals batch -------------------------------------------------------------------
    M["streaming_check"] = _streaming_check(flights, X, indexes, n=300 if quick else 2000)

    # ---- explainability, permutation importance by group ---------------------------------------------
    ref = explain.reference_row(Xtr)
    imp = {}
    rsamp = np.random.default_rng(11).choice(len(Xva), size=min(8000, len(Xva)), replace=False)
    Xs, ys = Xva.iloc[rsamp], yva[rsamp]
    base_ll = E.score_block(ys, model.predict_proba(Xs)[:, 1])["log_loss"]
    for gname, cols in F.GROUPS.items():
        Xp = Xs.copy()
        perm = np.random.default_rng(5).permutation(len(Xp))
        for c in cols:
            Xp[c] = Xs[c].to_numpy()[perm]
        imp[gname] = E.score_block(ys, model.predict_proba(Xp)[:, 1])["log_loss"] - base_ll
    M["group_importance_logloss_increase"] = imp

    # ---- artefacts ----------------------------------------------------------------------------------
    qf_bundle = qf
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "threshold": thr,
        "quantile_model": qf_bundle,
        "quantiles": taus,
        "reference": ref,
        "horizon": SHIPPED_HORIZON,
        "features": F.ALL_FEATURES,
        "shipped": chosen,
    }
    joblib.dump(bundle, out / "tailslip_model.joblib", compress=3)
    meta = flights.loc[
        te,
        [
            "flight_id",
            "sched_date",
            "carrier",
            "flight",
            "tailnum",
            "origin",
            "dest",
            "sched_utc",
            "hour",
            "minute",
            "dep_delay",
            "delayed",
        ],
    ].copy()
    replay = pd.concat([meta.reset_index(drop=True), Xte.reset_index(drop=True).add_prefix("f_")], axis=1)
    replay.to_parquet(out / "test_replay.parquet", index=False)

    # sample explanations
    samples = []
    dep_te = ~flights.loc[te, "cancelled"].to_numpy()
    idx = np.flatnonzero(dep_te)
    pick = [
        int(idx[np.argmax(cal_te[idx])]),
        int(idx[np.argmin(np.abs(cal_te[idx] - 0.5))]),
        int(idx[np.argmin(cal_te[idx])]),
    ]
    for i in pick:
        row = Xte.iloc[[i]]
        pred = lambda r: _score(bundle, r)
        samples.append(
            {
                "p": float(cal_te[i]),
                "actual_delay_min": float(flights.loc[te, "dep_delay"].iloc[i]),
                "reasons": explain.reasons(pred, row, ref),
            }
        )
    M["sample_explanations"] = samples

    plots.make_all(
        M,
        X,
        y,
        flights,
        split,
        test_arrays=dict(y=yte, raw=raw_te, cal=cal_te, grid=grid, vals=vals),
        qinfo=dict(y=ytd, pred=qpred, hours=hrs),
        assets=assets,
        quick=quick,
    )
    M["runtime_seconds"] = time.time() - t0
    (out / "metrics.json").write_text(json.dumps(_r(M), indent=2))
    log(f"done in {M['runtime_seconds']:.0f}s")
    return M


def _sensitivity(y, p, p_slot, scenarios):
    rows = []
    for value, cost in scenarios:
        brk = cost / (value + cost)
        rows.append(
            {
                "value_per_caught_delay": value,
                "cost_per_false_alert": cost,
                "break_even_probability": brk,
                "tailslip": E.alert_value(y, p >= brk, value, cost),
                "slot_rate": E.alert_value(y, p_slot >= brk, value, cost),
                "alert_every_flight": E.alert_value(y, np.ones(len(y), bool), value, cost),
            }
        )
    return rows


def _score(bundle, X):
    p = bundle["model"].predict_proba(X)[:, 1]
    if bundle["calibrator"] is not None:
        p = bundle["calibrator"].predict(p)
    return p


def _slot_quantiles(train: pd.DataFrame, target: pd.DataFrame, taus) -> np.ndarray:
    keys = ["carrier", "origin", "hour"]
    table = train.groupby(keys)["dep_delay"].quantile(list(taus)).unstack()
    table = table[train.groupby(keys).size() >= 30]
    glob = np.array([train["dep_delay"].quantile(t) for t in taus])
    idx = pd.MultiIndex.from_frame(target[keys])
    vals = table.reindex(idx).to_numpy().copy()
    vals[np.isnan(vals).any(axis=1)] = glob
    return vals


def _streaming_check(flights, X, indexes, n=2000):
    ok = flights[~flights["cancelled"]]
    books = {}
    for o, g in ok.groupby("origin"):
        b = state.AirportBook()
        for s, a, dly in zip(F.minutes(g["sched_utc"]), F.minutes(g["actual_utc"]), g["dep_delay"]):
            b.add(s, a, dly)
        books[o] = b
    ids = np.random.default_rng(0).choice(len(flights), n, replace=False)
    worst = 0.0
    cols = ["recent_n", "recent_mean_delay", "recent_late_share", "recent3h_mean_delay", "queue_ahead", "backlog", "sched_load"]
    sched = F.minutes(flights["sched_utc"])
    for i in ids:
        snap = books[flights["origin"].iloc[i]].snapshot(sched[i] - SHIPPED_HORIZON * 60, sched[i])
        for c in cols:
            a, b = snap[c], X.iloc[i][c]
            if not (np.isnan(a) and np.isnan(b)):
                worst = max(worst, abs(a - b))
    return {"flights_checked": int(n), "max_abs_difference": float(worst)}
