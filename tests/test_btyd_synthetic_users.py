import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
from lifetimes import BetaGeoFitter, GammaGammaFitter
from src.btyd_research_pipeline import compute_exact_btyd_predictions


def test_synthetic_users():
    # Fit BetaGeoFitter and GammaGammaFitter on small calibration population
    # Typical e-commerce distribution
    np.random.seed(42)
    N_cal = 5000
    syn_freq = np.random.geometric(p=0.3, size=N_cal) - 1
    syn_rec = np.random.uniform(0, 150, size=N_cal) * (syn_freq > 0)
    syn_T = syn_rec + np.random.uniform(5, 60, size=N_cal)
    syn_mon = np.random.gamma(shape=2.0, scale=30.0, size=N_cal)

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(syn_freq, syn_rec, syn_T)
    print("\n[+] BetaGeoFitter Fitted Parameters:")
    print(f"  r = {bgf.params_['r']:.4f}, alpha = {bgf.params_['alpha']:.4f}")
    print(f"  a = {bgf.params_['a']:.4f}, b = {bgf.params_['b']:.4f}")

    repeat_mask = (syn_freq > 0) & (syn_mon > 0)
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(syn_freq[repeat_mask], syn_mon[repeat_mask])
    print("\n[+] GammaGammaFitter Fitted Parameters:")
    print(f"  p = {ggf.params_['p']:.4f}, q = {ggf.params_['q']:.4f}, v = {ggf.params_['v']:.4f}")

    # 5 Synthetic User Personas:
    # U1: 1 purchase 150 days ago
    # U2: Regular customer (bought every 7 days, last 7 days ago)
    # U3: Regular customer, likely churned (bought every 7 days, but last purchase 80 days ago)
    # U4: Infrequent customer (2 purchases separated by 90 days)
    # U5: Frequent high-ticket customer
    rfm_df = pl.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "persona": [
            "U1_One_Old_Purchase",
            "U2_Regular_Active",
            "U3_Regular_Churned",
            "U4_Infrequent_2Orders",
            "U5_Frequent_Wholesaler",
        ],
        "btyd_available": [1.0, 1.0, 1.0, 1.0, 1.0],
        "btyd_frequency": [0.0, 19.0, 15.0, 1.0, 25.0],
        "btyd_recency": [0.0, 133.0, 105.0, 90.0, 140.0],
        "btyd_T": [150.0, 140.0, 185.0, 120.0, 145.0],
        "btyd_monetary_value": [50.0, 50.0, 50.0, 50.0, 500.0],
    })

    pred_df = compute_exact_btyd_predictions(bgf, ggf, rfm_df, t_horizons=[7, 14, 30])
    full_df = rfm_df.join(pred_df, on="user_id")

    print("\n=== SYNTHETIC USER PREDICTIONS (SECTION 8) ===")
    for row in full_df.iter_rows(named=True):
        print(f"User {row['user_id']} ({row['persona']}):")
        print(f"  RFM: freq={row['btyd_frequency']}, rec={row['btyd_recency']}, T={row['btyd_T']}, mon={row['btyd_monetary_value']}")
        print(f"  P(alive)={row['btyd_p_alive']:.4f}, logit(P_alive)={row['btyd_logit_p_alive']:.2f}")
        print(f"  Expected Purchases 30d={row['btyd_expected_purchases_30d']:.4f}")
        print(f"  P(buy 30d)={row['btyd_p_buy_30d']:.4f}, P(zero 30d)={row['btyd_p_zero_30d']:.4f}")
        print(f"  Expected Monetary={row['btyd_expected_monetary_value']:.2f}, Expected GMV 30d={row['btyd_expected_gmv_30d']:.2f}")
        print("-" * 50)

    # Mandatory Assertions
    u1 = full_df.filter(pl.col("user_id") == 1).to_dicts()[0]
    u2 = full_df.filter(pl.col("user_id") == 2).to_dicts()[0]
    u3 = full_df.filter(pl.col("user_id") == 3).to_dicts()[0]
    u4 = full_df.filter(pl.col("user_id") == 4).to_dicts()[0]
    u5 = full_df.filter(pl.col("user_id") == 5).to_dicts()[0]

    # U2 & U5 expected_count > U1 & U3
    assert u2["btyd_expected_purchases_30d"] > u1["btyd_expected_purchases_30d"], "U2 must have higher expected count than U1"
    assert u5["btyd_expected_purchases_30d"] > u3["btyd_expected_purchases_30d"], "U5 must have higher expected count than U3"
    
    # U3 (churned) P(alive) and expected count must be lower than U2 (active)
    assert u3["btyd_p_alive"] < u2["btyd_p_alive"], "U3 churned P(alive) must be strictly lower than U2"
    assert u3["btyd_expected_purchases_30d"] < u2["btyd_expected_purchases_30d"], "U3 expected purchases must be lower than U2"

    # U5 expected GMV must be much higher than U2
    assert u5["btyd_expected_gmv_30d"] > u2["btyd_expected_gmv_30d"] * 5, "U5 expected GMV must reflect higher monetary value"

    # Outputs not constant, not equal to 30.0
    exp_counts = [u["btyd_expected_purchases_30d"] for u in [u1, u2, u3, u4, u5]]
    assert len(set(exp_counts)) == 5, "Expected counts must be unique across all 5 distinct personas"
    assert not any(c == 30.0 for c in exp_counts), "Expected purchases must not equal 30.0"

    print("\n[+] ALL SANITY CHECKS PASSED PERFECTLY!")


if __name__ == "__main__":
    test_synthetic_users()
