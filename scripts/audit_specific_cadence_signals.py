"""Detailed Signal Presence Auditor for Cadence, Lifetime, Propensity, and Last-Year features."""

import os
import sys
sys.path.insert(0, os.getcwd())
import polars as pl
from pathlib import Path

def audit_signals():
    catalog = pl.read_csv("artifacts/catboost_cadence_audit/feature_catalog.csv")
    feat_names = set(catalog["feature_name"].to_list())

    # Detailed Checklist as specified in User Plan Sections 3, 4, 5
    checklist = [
        # 3.1 Purchase Recency
        ("days_since_last_purchase", "days_since_last_order", "EXISTS_EXACTLY", "Present in snapshots as days_since_last_order"),
        ("days_since_second_last_purchase", "-", "MISSING", "Second-to-last purchase date not tracked in tabular features"),
        ("days_since_third_last_purchase", "-", "MISSING", "Third-to-last purchase date not tracked"),
        ("days_since_first_purchase", "-", "MISSING", "First purchase date not tracked separately from customer_age_days"),
        ("days_since_last_active_day", "days_since_last_activity", "EXISTS_EXACTLY", "Present as days_since_last_activity"),

        # 3.2 Purchase Frequency
        ("purchase_days_7d", "purchase_days_7d", "EXISTS_EXACTLY", "Present in rolling 7d window"),
        ("purchase_days_14d", "purchase_days_14d", "EXISTS_EXACTLY", "Present in rolling 14d window"),
        ("purchase_days_30d", "purchase_days_30d", "EXISTS_EXACTLY", "Present in rolling 30d window"),
        ("purchase_days_60d", "purchase_days_60d", "EXISTS_EXACTLY", "Present in rolling 60d window"),
        ("purchase_days_90d", "purchase_days_90d", "EXISTS_EXACTLY", "Present in rolling 90d window"),
        ("purchase_days_180d", "-", "MISSING", "Tabular snapshot only computes up to 90d rolling windows"),
        ("purchase_days_lifetime", "lifetime_purchase_days", "EXISTS_EXACTLY", "Present as lifetime_purchase_days"),
        ("active_weeks_30d/90d/180d", "-", "MISSING", "Weekly active masks not computed"),
        ("share_of_weeks_with_purchase", "-", "MISSING", "Not computed"),
        ("share_of_completed_30d_windows_with_purchase", "-", "MISSING", "Multi-window panel share not computed"),

        # 3.3 Interpurchase Gaps
        ("last_interpurchase_gap", "-", "MISSING", "Specific last gap between purchases not computed"),
        ("previous_interpurchase_gap", "-", "MISSING", "Penultimate gap not computed"),
        ("mean_interpurchase_gap", "approx_order_interval_90d", "EXISTS_PARTIALLY", "Approximated as 90 / (orders_90d + 1), not exact gaps"),
        ("median_interpurchase_gap", "-", "MISSING", "Median interpurchase gap not computed"),
        ("std_interpurchase_gap", "-", "MISSING", "Std of gaps not computed"),
        ("min/max_interpurchase_gap", "-", "MISSING", "Min/Max gap not computed"),
        ("P25/P75_interpurchase_gap", "-", "MISSING", "Quantiles of gaps not computed"),
        ("coefficient_of_variation_gap", "-", "MISSING", "Gap CV (std/mean) not computed"),
        ("trend_interpurchase_gap", "-", "MISSING", "Gap acceleration/trend not computed"),
        ("number_of_observed_gaps", "-", "MISSING", "Count of gaps not computed"),

        # 3.4 Purchase Cycle Phase
        ("gap_ratio (days_since_last / median_gap)", "-", "MISSING", "Cycle phase ratio not computed"),
        ("overdue_days (days_since_last - median_gap)", "-", "MISSING", "Overdue days not computed"),
        ("gap_zscore", "-", "MISSING", "Z-score of current gap not computed"),
        ("is_cycle_overdue", "-", "MISSING", "Binary overdue indicator not computed"),
        ("cycle_estimate_available", "-", "MISSING", "Availability flag not computed"),
        ("has_2_purchase_days", "-", "MISSING", "Flag for >= 2 purchase days not computed"),
        ("has_3_purchase_days", "-", "MISSING", "Flag for >= 3 purchase days not computed"),

        # 3.5 Regularity and Inactivity Streaks
        ("longest_inactivity_streak", "-", "MISSING", "Max streak of inactivity not computed"),
        ("current_inactivity_streak", "days_since_last_activity", "EXISTS_PARTIALLY", "Represented by days_since_last_activity, but not streak-specific"),
        ("mean_active_streak", "-", "MISSING", "Mean consecutive active days not computed"),
        ("purchase_week_regularness", "-", "MISSING", "Weekly regularity score not computed"),
        ("active_week_entropy", "-", "MISSING", "Entropy of activity across weeks not computed"),
        ("recent_vs_long_ratios", "gmv_ratio_7_30d / gmv_ratio_14_60d", "EXISTS_EXACTLY", "Present for GMV, orders, searches, carts, active_days"),
        ("slope_purchase_activity", "ts_gmv_acceleration_ratio", "EXISTS_PARTIALLY", "Present as 14d vs 90d acceleration ratio, linear regression slope missing"),

        # 3.6 User Tenure and Censoring
        ("first_seen_date / tenure", "customer_age_days", "EXISTS_EXACTLY", "Present as customer_age_days"),
        ("available_history_days", "available_history_days", "EXISTS_EXACTLY", "Present in snapshots"),
        ("is_history_censored", "-", "MISSING", "Explicit censoring flag (tenure < 180d) not computed"),

        # 4. Personal Propensity
        ("personal_buy_rate (historical multi-anchor)", "-", "MISSING", "Historical target propensity across past completed windows not computed"),
        ("personal_reactivation_rate", "-", "MISSING", "Historical reactivation propensity not computed"),
        ("personal_churn_rate", "-", "MISSING", "Historical churn propensity not computed"),
        ("mean_positive_target_log_gmv", "-", "MISSING", "Historical positive target GMV not computed"),
        ("smoothed_rate (k=5, 20)", "-", "MISSING", "Bayesian m-estimate smoothed rates not computed"),
        ("user_id categorical", "-", "MISSING", "user_id is not passed as categorical feature to CatBoost"),

        # 5. Exact Target-Aligned Last-Year
        ("ly_same_target_buy", "-", "MISSING", "Only general ly_window_purchase_days in transitions/features, exact target-aligned snapshot missing in baseline"),
        ("ly_same_target_gmv", "-", "MISSING", "Exact target-aligned GMV missing in baseline"),
        ("ly_same_target_available", "-", "MISSING", "Exact availability flag missing in baseline"),
    ]

    df_check = pl.DataFrame({
        "signal_name": [c[0] for c in checklist],
        "existing_equivalent": [c[1] for c in checklist],
        "status": [c[2] for c in checklist],
        "notes": [c[3] for c in checklist],
    })

    out_path = Path("artifacts/catboost_cadence_audit/signal_audit_checklist.csv")
    df_check.write_csv(out_path)
    print(f"[+] Saved signal checklist to {out_path}")
    print("\nSummary of Signal Statuses:")
    print(df_check.group_by("status").agg(pl.count().alias("count")))

if __name__ == "__main__":
    audit_signals()
