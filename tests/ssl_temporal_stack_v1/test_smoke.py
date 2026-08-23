from __future__ import annotations

from datetime import date

import polars as pl

from src.ssl_temporal_stack_v1.smoke import _smoke_cohort


def test_smoke_cohort_selects_25_users_per_transition() -> None:
    users = list(range(1, 401))
    rows = []
    anchor = date(2025, 11, 10)
    for index, user in enumerate(users):
        previous = (index // 25) % 2
        future = (index // 50) % 2
        if previous:
            rows.append((user, anchor, 1.0))
        if future:
            rows.append((user, date(2025, 11, 11), 1.0))
    raw = pl.DataFrame(rows, schema=["user_id", "event_date", "gmv"], orient="row")
    selected, counts = _smoke_cohort(raw, users)
    assert len(selected) == 100
    assert counts == {"00": 25, "01": 25, "10": 25, "11": 25}
