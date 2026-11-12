import numpy as np

from tailslip import data


def test_prepare_marks_cancelled_as_delayed(tables):
    df = data.prepare_flights(tables["flights"])
    canc = df[df["dep_delay"].isna()]
    assert len(canc) > 0 and (canc["delayed"] == 1).all()
    ok = df[df["dep_delay"].notna()]
    assert ((ok["dep_delay"] >= 15).astype(int) == ok["delayed"]).all()


def test_actual_departure_is_schedule_plus_delay(tables):
    df = data.prepare_flights(tables["flights"])
    row = df[df["dep_delay"].notna()].iloc[0]
    assert (row["actual_utc"] - row["sched_utc"]).total_seconds() / 60 == row["dep_delay"]


def test_sorted_by_schedule_with_ids(tables):
    df = data.prepare_flights(tables["flights"])
    assert df["sched_utc"].is_monotonic_increasing
    assert (df["flight_id"].to_numpy() == np.arange(len(df))).all()


def test_drop_unassigned_removes_missing_tails(tables):
    df = data.prepare_flights(tables["flights"])
    kept, n = data.drop_unassigned(df)
    assert n == int((df["tailnum"] == "UNKNOWN").sum()) and n > 0
    assert (kept["tailnum"] != "UNKNOWN").all()


def test_split_is_forward_in_time(flights):
    s = data.split_by_month(flights)
    for a, b in (("train", "validation"), ("validation", "test")):
        assert flights.loc[s == a, "sched_utc"].max() < flights.loc[s == b, "sched_utc"].min()


def test_load_tables_roundtrip(raw_dir, tables):
    loaded = data.load_tables(raw_dir)
    assert len(loaded["flights"]) == len(tables["flights"])
    assert set(loaded) == {"flights", "weather", "planes", "airports", "airlines"}
