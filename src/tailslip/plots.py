"""Charts for the README and the overview document."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from plotnine import (  # noqa: E402
    aes,
    element_text,
    geom_line,
    geom_point,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient,
    scale_x_continuous,
    theme,
    theme_minimal,
)

INK, BLUE, ORANGE, GREY = "#1f2933", "#2b6cb0", "#dd6b20", "#a0aec0"
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 10,
    }
)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def lead_time(M, path):
    df = pd.DataFrame(M["lead_time"])
    slot_auc = M["models"]["slot rate"]["test"]["auc"]
    g = (
        ggplot(df, aes("hours_before", "auc"))
        + geom_line(color=BLUE, size=1)
        + geom_point(color=BLUE, size=3)
        + scale_x_continuous(breaks=df["hours_before"].tolist())
        + labs(
            x="hours before scheduled departure",
            y="test AUC",
            title="How much the forecast gains as departure gets closer",
            subtitle=f"Dashed baseline is the carrier and hour lookup, AUC {slot_auc:.3f}",
        )
        + theme_minimal()
        + theme(plot_title=element_text(weight="bold"))
    )
    from plotnine import geom_hline

    g = g + geom_hline(yintercept=slot_auc, linetype="dashed", color=GREY)
    g.save(path, width=7, height=4, dpi=150, verbose=False)


def heatmap(flights, path):
    d = flights.assign(dow=flights["sched_date"].dt.dayofweek, hour=flights["hour"])
    t = d.groupby(["dow", "hour"])["delayed"].mean().reset_index()
    t = t[t["hour"].between(5, 23)]
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    t["day"] = pd.Categorical([names[i] for i in t["dow"]], categories=names[::-1])
    g = (
        ggplot(t, aes("hour", "day", fill="delayed"))
        + geom_tile()
        + scale_fill_gradient(low="#ebf4fa", high="#c05621", name="share late")
        + labs(
            x="scheduled hour (local)",
            y="",
            title="Late departures build through the day",
            subtitle="Share of flights leaving 15+ minutes late or cancelled, all of 2013",
        )
        + theme_minimal()
        + theme(plot_title=element_text(weight="bold"))
    )
    g.save(path, width=8, height=3.6, dpi=150, verbose=False)


def model_comparison(M, path):
    names = list(M["models"])
    ll = [M["models"][n]["test"]["log_loss"] for n in names]
    auc = [M["models"][n]["test"]["auc"] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    colors = [ORANGE if n == M["selection"]["shipped"] else BLUE for n in names]
    ax[0].barh(names, ll, color=colors)
    ax[0].set_xlabel("test log loss (lower is better)")
    ax[1].barh(names, auc, color=colors)
    ax[1].set_xlim(0.45, max(auc) + 0.04)
    ax[1].set_xlabel("test AUC")
    ax[1].set_yticklabels([])
    for a, vals in zip(ax, (ll, auc)):
        for i, v in enumerate(vals):
            a.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    _save(fig, path)


def calibration(T, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    for key, label, col in (("raw", "shipped model", ORANGE),):
        p, y = T[key], T["y"]
        order = np.argsort(p)
        chunks = np.array_split(order, 10)
        ax.plot([p[c].mean() for c in chunks], [y[c].mean() for c in chunks], marker="o", color=col, label=label)
    ax.set_xlabel("predicted probability of a late departure")
    ax.set_ylabel("observed share late")
    ax.legend(frameon=False)
    _save(fig, path)


def chain_effect(M, path):
    rows = M["chain_effect"]
    labels = [r["band"] for r in rows]
    vals = [r["delay_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [BLUE] * len(rows)
    colors[-1] = GREY
    colors[-2] = ORANGE
    ax.bar(range(len(rows)), vals, color=colors)
    ax.set_xticks(range(len(rows)))
    wrapped = {
        "previous leg still on ground": "previous leg\nstill on ground",
        "no previous leg known": "no previous\nleg known",
    }
    ax.set_xticklabels([wrapped.get(b, b.replace(" late", "\nlate")) for b in labels], fontsize=8)
    ax.set_ylabel("share of flights delayed")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_title("A late aircraft hands its delay to the next flight", loc="left", fontweight="bold")
    _save(fig, path)


def economics(M, T, path):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax[0].plot(T["grid"], T["vals"], color=BLUE)
    thr = M["economics"]["threshold_from_validation"]
    ax[0].axvline(thr, color=ORANGE, ls="--")
    ax[0].set_xlabel("alert threshold")
    ax[0].set_ylabel("net dollars per flight (validation)")
    ax[0].set_title(f"Threshold chosen on validation: {thr:.2f}", loc="left", fontsize=10)
    e = M["economics"]["net_per_flight"]
    names = list(e)
    ax[1].barh(names, [e[n] for n in names], color=[ORANGE if n == "tailslip" else BLUE for n in names])
    for i, n in enumerate(names):
        ax[1].text(e[n], i, f" {e[n]:.2f}", va="center", fontsize=8)
    ax[1].set_xlabel("net dollars per flight (test)")
    _save(fig, path)


def ablation(M, path):
    ab = M["group_ablation"]
    if not ab:
        return
    full = M["models"]["boosting, full"]["test"]["log_loss"]
    names = list(ab)
    inc = [ab[n]["log_loss"] - full for n in names]
    order = np.argsort(inc)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh([names[i] for i in order], [inc[i] for i in order], color=BLUE)
    ax.set_xlabel("increase in test log loss when the block is removed and the model refit")
    _save(fig, path)


def quantile_coverage(M, path):
    cov = M["quantiles"]["p80_coverage_by_hour"]
    hours = sorted(cov)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(hours, [cov[h] for h in hours], marker="o", color=BLUE)
    ax.axhline(0.8, color=ORANGE, ls="--")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("scheduled hour (local)")
    ax.set_ylabel("share of flights leaving before the p80 time")
    _save(fig, path)


def make_all(M, X, y, flights, split, test_arrays, qinfo, assets, quick=False):
    assets = Path(assets)
    lead_time(M, assets / "lead_time.png")
    heatmap(flights, assets / "delay_heatmap.png")
    model_comparison(M, assets / "model_comparison.png")
    calibration(test_arrays, assets / "calibration.png")
    chain_effect(M, assets / "chain_effect.png")
    economics(M, test_arrays, assets / "economics.png")
    ablation(M, assets / "group_ablation.png")
    quantile_coverage(M, assets / "quantile_coverage.png")
