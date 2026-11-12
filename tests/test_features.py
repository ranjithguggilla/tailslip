import numpy as np
import pandas as pd
import pytest

from tailslip import features as F
from tailslip import state


@pytest.fixture(scope="module")
def X2(flights, tables):
    return F.build_features(flights, tables, 2.0)


def test_columns_and_row_alignment(X2, flights):
    assert list(X2.columns) == F.ALL_FEATURES
    assert len(X2) == len(flights) and X2.index.equals(flights.index)


def test_wind_glitch_is_clipped(flights, tables):
    t = {**tables, "weather": tables["weather"].assign(wind_speed=1048.0)}
    X = F.build_features(flights, t, 2.0)
    assert X["wind_speed"].max() <= 70.0


def test_prev_leg_matches_manual_lookup(flights, X2):
    tail = flights["tailnum"].value_counts().index[0]
    g = flights[flights["tailnum"] == tail].sort_values("sched_utc")
    cur, prev = g.iloc[5], g.iloc[4]
    T = cur["sched_utc"] - pd.Timedelta(hours=2)
    row = X2.loc[cur.name]
    if prev["cancelled"]:
        assert np.isnan(row["prev_dep_delay"])
    elif prev["actual_utc"] <= T:
        assert row["prev_dep_delay"] == prev["dep_delay"]
    else:
        assert np.isnan(row["prev_dep_delay"])
    assert row["rotation_gap_h"] == pytest.approx(min((cur["sched_utc"] - prev["sched_utc"]).total_seconds() / 3600, 72))


def test_overdue_only_when_previous_leg_still_on_ground(flights, X2):
    over = X2[X2["prev_overdue_min"] > 0]
    assert len(over) > 0
    assert over["prev_dep_delay"].isna().all()


def test_nothing_after_decision_time_leaks_into_prev_features(flights, tables):
    """Features at a longer horizon may only know less about the previous leg."""
    far = F.build_features(flights, tables, 24.0)
    near = F.build_features(flights, tables, 0.0)
    assert far["prev_dep_delay"].notna().mean() <= near["prev_dep_delay"].notna().mean()


def test_airport_features_do_not_include_the_flight_itself(flights, tables):
    far = F.build_features(flights, tables, 6.0)
    ok = flights[~flights["cancelled"]]
    i = ok.index[len(ok) // 2]
    t = F.minutes(flights.loc[[i], "sched_utc"])[0] - 6 * 60
    assert F.minutes(flights.loc[[i], "actual_utc"])[0] > t
    cnt = far.loc[i, "recent_n"]
    window = ok[
        (ok["origin"] == flights.loc[i, "origin"]) & (F.minutes(ok["actual_utc"]) > t - 60) & (F.minutes(ok["actual_utc"]) <= t)
    ]
    assert cnt == len(window)


def test_weather_uses_lagged_observation(flights, tables):
    wx = F.weather_table(tables["weather"])
    X = F.build_features(flights, tables, 2.0, wx=wx)
    i = flights.index[100]
    origin = flights.loc[i, "origin"]
    key = (flights.loc[i, "sched_utc"] - pd.Timedelta(hours=3)).floor("h")
    assert X.loc[i, "temp"] == pytest.approx(wx[origin].loc[key, "temp"])


def test_streaming_book_matches_batch(flights, X2):
    ok = flights[~flights["cancelled"]]
    books = {}
    for o, g in ok.groupby("origin"):
        b = state.AirportBook()
        for s, a, d in zip(F.minutes(g["sched_utc"]), F.minutes(g["actual_utc"]), g["dep_delay"]):
            b.add(s, a, d)
        books[o] = b
    sched = F.minutes(flights["sched_utc"])
    for i in np.random.default_rng(1).choice(len(flights), 300, replace=False):
        snap = books[flights["origin"].iloc[i]].snapshot(sched[i] - 120.0, sched[i])
        for k, v in snap.items():
            w = X2.iloc[i][k]
            assert (np.isnan(v) and np.isnan(w)) or v == pytest.approx(w)


def test_book_empty_window_is_nan():
    b = state.AirportBook()
    b.add(100.0, 110.0, 10.0)
    snap = b.snapshot(50.0, 120.0)
    assert snap["recent_n"] == 0 and np.isnan(snap["recent_mean_delay"])
    assert len(b) == 1
