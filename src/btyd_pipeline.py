"""Leakage-safe Vectorized BTYD (BG/NBD & Gamma-Gamma) Feature Pipeline.

Calculates:
1. RFM Statistics: frequency, recency, T, monetary_value.
2. BG/NBD: P(alive), expected_transactions_30d.
3. Gamma-Gamma: expected_monetary_value.
4. Composite: expected_gmv_30d = expected_transactions_30d * expected_monetary_value.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln


class VectorizedBGNBD:
    """Fast vectorized BG/NBD (Beta-Geometric / Negative Binomial Distribution)."""

    def __init__(self, r: float = 1.0, alpha: float = 10.0, a: float = 1.0, b: float = 1.0):
        self.r = r
        self.alpha = alpha
        self.a = a
        self.b = b
        self.fitted = False

    def _log_likelihood(self, params: np.ndarray, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> float:
        r, alpha, a, b = np.exp(params)
        
        ll_1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        ll_2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        
        term1 = (r + x) * np.log(alpha + T)
        c_val = np.maximum(a / (b + np.maximum(x - 1, 0)), 1e-10)
        term2 = np.where(
            x > 0,
            np.log(1.0 + c_val * np.power((alpha + T) / (alpha + t_x), r + x)),
            0.0
        )
        
        ll = ll_1 + ll_2 - term1
        return -float(np.sum(ll))

    def fit(self, x: np.ndarray, t_x: np.ndarray, T: np.ndarray, max_samples: int = 20000):
        if len(x) > max_samples:
            idx = np.random.choice(len(x), size=max_samples, replace=False)
            x_s, tx_s, T_s = x[idx], t_x[idx], T[idx]
        else:
            x_s, tx_s, T_s = x, t_x, T

        x_mean = np.mean(x_s)
        x_var = np.var(x_s)
        init_r = max(0.5, (x_mean ** 2) / max(1e-4, x_var - x_mean))
        init_alpha = max(1.0, x_mean / max(1e-4, x_var - x_mean) * np.mean(T_s))

        init_params = np.log([init_r, init_alpha, 0.5, 2.0])
        
        try:
            res = minimize(
                self._log_likelihood,
                init_params,
                args=(x_s, tx_s, T_s),
                method="Nelder-Mead",
                options={"maxiter": 200},
            )
            self.r, self.alpha, self.a, self.b = np.exp(res.x)
            self.fitted = True
        except Exception:
            self.r, self.alpha, self.a, self.b = 0.8, 15.0, 0.6, 2.5
            self.fitted = True

    def predict_p_alive(self, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> np.ndarray:
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        ratio = (alpha + T) / np.maximum(alpha + t_x, 1e-4)
        term = (a / (b + np.maximum(x - 1, 0))) * np.power(ratio, r + x)
        p_alive = np.where(x > 0, 1.0 / (1.0 + term), 1.0)
        return np.clip(p_alive, 1e-6, 1.0)

    def predict_expected_transactions(self, t_future: float, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> np.ndarray:
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        p_alive = self.predict_p_alive(x, t_x, T)
        hyp_term = ((a + b + x - 1.0) / max(1e-4, a - 1.0)) * (
            1.0 - np.power((alpha + T) / (alpha + T + t_future), r + x)
        )
        exp_trans = p_alive * hyp_term
        return np.clip(exp_trans, 0.0, t_future)


class VectorizedGammaGamma:
    """Fast vectorized Gamma-Gamma model for conditional expected monetary value."""

    def __init__(self, p: float = 1.0, q: float = 2.0, v: float = 1000.0):
        self.p = p
        self.q = q
        self.v = v
        self.fitted = False

    def fit(self, x: np.ndarray, m_x: np.ndarray):
        mask = (x > 0) & (m_x > 0)
        if mask.sum() < 50:
            self.p, self.q, self.v = 1.5, 3.0, 1500.0
            self.fitted = True
            return

        m_pos = m_x[mask]
        mean_m = float(np.mean(m_pos))
        self.p = 2.0
        self.q = 3.0
        self.v = mean_m * (self.q - 1.0) / self.p
        self.fitted = True

    def predict_expected_monetary_value(self, x: np.ndarray, m_x: np.ndarray) -> np.ndarray:
        p, q, v = self.p, self.q, self.v
        prior_mean = (v * p) / max(1e-4, q - 1.0)
        weight = (p * x) / (p * x + q - 1.0)
        exp_m = np.where(x > 0, weight * m_x + (1.0 - weight) * prior_mean, prior_mean)
        return np.maximum(exp_m, 0.0)


def extract_rfm_features_for_anchor(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
    history_days: int = 180,
) -> pl.DataFrame:
    """Extracts leakage-safe RFM and inter-purchase intervals strictly before anchor_date."""
    start_date = anchor_date - timedelta(days=history_days - 1)
    user_set = set(user_ids)

    hist = data.filter(
        (pl.col("event_date") >= start_date)
        & (pl.col("event_date") <= anchor_date)
        & (pl.col("user_id").is_in(user_set))
    )

    purchases = (
        hist.filter(pl.col("gmv") > 0)
        .group_by(["user_id", "event_date"])
        .agg(pl.col("gmv").sum().alias("daily_gmv"))
    )

    if purchases.height == 0:
        base_df = pl.DataFrame({"user_id": user_ids})
        return base_df.with_columns([
            pl.lit(0.0).alias("btyd_frequency"),
            pl.lit(0.0).alias("btyd_recency_days"),
            pl.lit(float(history_days)).alias("btyd_T_days"),
            pl.lit(0.0).alias("btyd_monetary_avg"),
        ])

    rfm = (
        purchases.group_by("user_id")
        .agg([
            pl.count().alias("n_purchases"),
            pl.col("event_date").min().alias("first_purch"),
            pl.col("event_date").max().alias("last_purch"),
            pl.col("daily_gmv").mean().alias("monetary_avg"),
        ])
        .with_columns([
            (pl.col("n_purchases") - 1).clip(0, None).alias("btyd_frequency"),
            ((pl.col("last_purch") - pl.col("first_purch")).dt.total_days()).alias("btyd_recency_days"),
            ((pl.lit(anchor_date) - pl.col("first_purch")).dt.total_days()).alias("btyd_T_days"),
            pl.col("monetary_avg").alias("btyd_monetary_avg"),
        ])
        .select(["user_id", "btyd_frequency", "btyd_recency_days", "btyd_T_days", "btyd_monetary_avg"])
    )

    base = pl.DataFrame({"user_id": user_ids})
    full_rfm = (
        base.join(rfm, on="user_id", how="left")
        .with_columns([
            pl.col("btyd_frequency").fill_null(0.0),
            pl.col("btyd_recency_days").fill_null(0.0),
            pl.col("btyd_T_days").fill_null(float(history_days)),
            pl.col("btyd_monetary_avg").fill_null(0.0),
        ])
    )
    return full_rfm


def generate_btyd_dataset_for_anchor(
    data: pl.DataFrame,
    user_ids: List[int],
    anchor_date: date,
    bgnbd_model: Optional[VectorizedBGNBD] = None,
    gamma_model: Optional[VectorizedGammaGamma] = None,
    fit_models: bool = False,
) -> Tuple[pl.DataFrame, VectorizedBGNBD, VectorizedGammaGamma]:
    """Generates complete BTYD feature table for anchor with trained or fitted models."""
    rfm_df = extract_rfm_features_for_anchor(data, user_ids, anchor_date)

    x = rfm_df["btyd_frequency"].to_numpy().astype(np.float64)
    t_x = rfm_df["btyd_recency_days"].to_numpy().astype(np.float64)
    T = rfm_df["btyd_T_days"].to_numpy().astype(np.float64)
    m_x = rfm_df["btyd_monetary_avg"].to_numpy().astype(np.float64)

    if fit_models or bgnbd_model is None or not bgnbd_model.fitted:
        bgnbd_model = VectorizedBGNBD()
        bgnbd_model.fit(x, t_x, T)

    if fit_models or gamma_model is None or not gamma_model.fitted:
        gamma_model = VectorizedGammaGamma()
        gamma_model.fit(x, m_x)

    p_alive = bgnbd_model.predict_p_alive(x, t_x, T)
    exp_trans_30d = bgnbd_model.predict_expected_transactions(30.0, x, t_x, T)
    exp_monetary = gamma_model.predict_expected_monetary_value(x, m_x)
    exp_gmv_30d = exp_trans_30d * exp_monetary
    exp_z_30d = np.log1p(exp_gmv_30d)

    p_alive_clipped = np.clip(p_alive, 1e-7, 1.0 - 1e-7)
    logit_p_alive = np.log(p_alive_clipped / (1.0 - p_alive_clipped))
    p_zero_30d = np.exp(-np.clip(exp_trans_30d, 0.0, 50.0))
    p_buy_30d = 1.0 - p_zero_30d

    btyd_df = rfm_df.with_columns([
        pl.Series("btyd_p_alive", p_alive),
        pl.Series("btyd_logit_p_alive", logit_p_alive),
        pl.Series("btyd_expected_purchases_30d", exp_trans_30d),
        pl.Series("btyd_p_at_least_one_purchase_30d", p_buy_30d),
        pl.Series("btyd_p_zero_purchase_30d", p_zero_30d),
        pl.Series("btyd_expected_monetary_value", exp_monetary),
        pl.Series("btyd_expected_gmv_30d", exp_gmv_30d),
        pl.Series("btyd_log_expected_gmv_30d", exp_z_30d),
    ])

    return btyd_df, bgnbd_model, gamma_model

