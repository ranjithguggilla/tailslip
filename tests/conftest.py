import io
import zipfile

import numpy as np
import pandas as pd
import pytest


def make_tables(seed: int = 0, days_per_month: int = 8, per_origin_day: int = 28):
    """A small synthetic set of tables shaped like nycflights13."""
    rng = np.random.default_rng(seed)
    carriers = ["AA", "DL", "UA", "B6"]
    dests = ["ATL", "ORD", "MIA", "BOS", "LAX", "DEN"]
    tails = [f"N{100 + i}AA" for i in range(40)]
    rows = []
    for month in range(1, 13):
        for day in range(1, days_per_month + 1):
            storm = rng.random() < 0.15
            for origin in ["EWR", "JFK", "LGA"]:
                for k in range(per_origin_day):
                    hour = int(rng.integers(6, 22))
                    minute = int(rng.integers(0, 60))
                    tail = tails[int(rng.integers(0, len(tails)))]
                    base = 4 + 1.2 * (hour - 6) + (25 if storm else 0)
                    delay = float(rng.gamma(1.4, max(base, 1.0)) - 8)
                    cancelled = rng.random() < 0.02
                    rows.append(
                        dict(
                            year=2013,
                            month=month,
                            day=day,
                            hour=hour,
                            minute=minute,
                            sched_dep_time=hour * 100 + minute,
                            dep_delay=np.nan if cancelled else round(delay),
                            arr_delay=np.nan if cancelled else round(delay - 3),
                            carrier=carriers[k % 4],
                            flight=int(rng.integers(1, 3000)),
                            tailnum=np.nan if cancelled and rng.random() < 0.8 else tail,
                            origin=origin,
                            dest=dests[int(rng.integers(0, len(dests)))],
                            air_time=np.nan if cancelled else float(rng.integers(60, 300)),
                            distance=int(rng.integers(200, 2500)),
                            time_hour=f"2013-{month:02d}-{day:02d}T{(hour + 5) % 24:02d}:00:00Z",
                        )
                    )
    flights = pd.DataFrame(rows)
    hours = pd.date_range("2013-01-01", "2014-01-02", freq="h")
    weather = pd.concat(
        [
            pd.DataFrame(
                dict(
                    origin=o,
                    year=2013,
                    month=h.month,
                    day=h.day,
                    hour=h.hour,
                    temp=50 + rng.normal(0, 10, len(h)),
                    dewp=40.0,
                    humid=60.0,
                    wind_dir=180.0,
                    wind_speed=rng.gamma(3, 3, len(h)),
                    wind_gust=np.where(rng.random(len(h)) < 0.2, 30.0, np.nan),
                    precip=rng.binomial(1, 0.05, len(h)) * 0.1,
                    pressure=1015.0,
                    visib=np.where(rng.random(len(h)) < 0.05, 2.0, 10.0),
                    time_hour=[t.strftime("%Y-%m-%dT%H:00:00Z") for t in hours],
                )
            )
            for o, h in (("EWR", hours), ("JFK", hours), ("LGA", hours))
        ]
    )
    planes = pd.DataFrame(
        dict(
            tailnum=tails[:30],
            year=rng.integers(1990, 2012, 30).astype(float),
            type="Fixed wing multi engine",
            manufacturer="BOEING",
            model="737",
            engines=2,
            seats=rng.integers(80, 200, 30),
            speed=np.nan,
            engine="Turbo-fan",
        )
    )
    airports = pd.DataFrame(dict(faa=["EWR"], name=["Newark"]))
    airlines = pd.DataFrame(dict(carrier=carriers, name=carriers))
    return dict(flights=flights, weather=weather, planes=planes, airports=airports, airlines=airlines)


@pytest.fixture(scope="session")
def tables():
    return make_tables()


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory, tables):
    d = tmp_path_factory.mktemp("raw")
    buf = io.StringIO()
    tables["flights"].to_csv(buf, index=False)
    with zipfile.ZipFile(d / "flights.csv.zip", "w") as zf:
        zf.writestr("flights.csv", buf.getvalue())
    for name in ("weather", "planes", "airports", "airlines"):
        tables[name].to_csv(d / f"{name}.csv", index=False)
    return d


@pytest.fixture(scope="session")
def flights(tables):
    from tailslip import data

    df = data.prepare_flights(tables["flights"])
    df, _ = data.drop_unassigned(df)
    return df
