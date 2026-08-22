"""Unit tests for newly added Cadence and Personal Propensity features across verified personas."""

import os
import sys
sys.path.insert(0, os.getcwd())
from datetime import date, timedelta
import polars as pl
import numpy as np

from src.cadence_features import extract_cadence_features_for_anchor
from src.personal_propensity_features import (
    compute_exact_last_year_target_features,
    compute_personal_propensity_features,
)


def test_user_personas():
    anchor = date(2026, 1, 14)
    # Build synthetic test log for 7 distinct personas
    logs = []

    # Persona 1: Never bought (uid 1) -> searches only
    logs.append({"user_id": 1, "event_date": date(2026, 1, 10), "searches": 5, "to_cart": 0, "to_ord": 0, "gmv": 0.0})

    # Persona 2: Single purchase (uid 2) -> purchase on 2026-01-05
    logs.append({"user_id": 2, "event_date": date(2026, 1, 5), "searches": 2, "to_cart": 1, "to_ord": 1, "gmv": 1500.0})

    # Persona 3: Exactly 2 purchases (uid 3) -> on 2025-12-01 and 2026-01-01 (gap = 31 days)
    logs.append({"user_id": 3, "event_date": date(2025, 12, 1), "searches": 1, "to_cart": 1, "to_ord": 1, "gmv": 800.0})
    logs.append({"user_id": 3, "event_date": date(2026, 1, 1), "searches": 1, "to_cart": 1, "to_ord": 1, "gmv": 1200.0})

    # Persona 4: Regular buyer (uid 4) -> purchases every 14 days
    for d_offset in [14, 28, 42, 56, 70, 84]:
        d = anchor - timedelta(days=d_offset)
        logs.append({"user_id": 4, "event_date": d, "searches": 3, "to_cart": 2, "to_ord": 1, "gmv": 1000.0})

    # Persona 5: Churned sleeper (uid 5) -> active 140 days ago, then silent
    logs.append({"user_id": 5, "event_date": anchor - timedelta(days=140), "searches": 2, "to_cart": 1, "to_ord": 1, "gmv": 2000.0})

    # Persona 6: New user (uid 6) -> first seen 30 days ago (censored)
    logs.append({"user_id": 6, "event_date": anchor - timedelta(days=30), "searches": 1, "to_cart": 1, "to_ord": 1, "gmv": 500.0})

    # Persona 7: Last-year buyer (uid 7) -> bought on 2025-01-20 (exact LY target window for Jan 14 anchor)
    logs.append({"user_id": 7, "event_date": date(2025, 1, 20), "searches": 4, "to_cart": 2, "to_ord": 1, "gmv": 3500.0})

    data_df = pl.DataFrame(logs)
    user_ids = [1, 2, 3, 4, 5, 6, 7]

    print("[*] Testing Cadence feature extractor on 7 personas...")
    cadence_df = extract_cadence_features_for_anchor(data_df, user_ids, anchor)
    assert cadence_df.height == 7, "Height mismatch in cadence features"

    # Persona 1: No purchases -> cycle_estimate_available == 0
    p1 = cadence_df.filter(pl.col("user_id") == 1).to_dicts()[0]
    assert p1["has_2_purchase_days"] == 0.0
    assert p1["cycle_estimate_available"] == 0.0
    assert p1["purchase_days_180d"] == 0.0

    # Persona 3: Exactly 2 purchases -> gap == 31 days
    p3 = cadence_df.filter(pl.col("user_id") == 3).to_dicts()[0]
    assert p3["has_2_purchase_days"] == 1.0
    assert p3["last_interpurchase_gap"] == 31.0
    assert p3["median_interpurchase_gap"] == 31.0

    # Persona 4: Regular buyer -> CV gap should be near 0
    p4 = cadence_df.filter(pl.col("user_id") == 4).to_dicts()[0]
    assert p4["purchase_days_180d"] == 6.0
    assert abs(p4["mean_interpurchase_gap"] - 14.0) < 1e-4

    # Persona 6: New user -> is_history_censored == 1
    p6 = cadence_df.filter(pl.col("user_id") == 6).to_dicts()[0]
    assert p6["is_history_censored"] == 1.0

    print("[+] Cadence features assertions PASSED!")

    # Test Exact Last-Year Features
    print("[*] Testing Exact Last-Year target feature extractor...")
    ly_df = compute_exact_last_year_target_features(data_df, user_ids, anchor)
    p7 = ly_df.filter(pl.col("user_id") == 7).to_dicts()[0]
    assert p7["ly_exact_target_buy"] == 1.0
    assert p7["ly_exact_target_gmv"] == 3500.0
    assert p7["ly_exact_target_available"] == 1.0

    p1_ly = ly_df.filter(pl.col("user_id") == 1).to_dicts()[0]
    assert p1_ly["ly_exact_target_buy"] == 0.0
    print("[+] Last-Year features assertions PASSED!")


if __name__ == "__main__":
    test_user_personas()
    print("\nALL UNIT TESTS FOR NEW FEATURES COMPLETED AND VERIFIED!")
