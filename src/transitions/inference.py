"""Factorized GMV Inference Engine combining Transition Probabilities with Conditional Value Regressor."""

from typing import Dict, Optional, Tuple, Union
import numpy as np


def assemble_factorized_probabilities(
    past_buyer_30d: np.ndarray,
    p_reactivation: np.ndarray,
    p_churn: np.ndarray,
) -> np.ndarray:
    """Assembles overall probability of purchase: p_buy = p_reactivation if past=0 else (1 - p_churn)."""
    past_buyer_30d = np.asarray(past_buyer_30d, dtype=np.int32)
    p_reactivation = np.asarray(p_reactivation, dtype=np.float32)
    p_churn = np.asarray(p_churn, dtype=np.float32)

    p_buy = np.where(
        past_buyer_30d == 0,
        p_reactivation,
        1.0 - p_churn,
    )
    return np.clip(p_buy, 0.0, 1.0)


def compute_factorized_gmv(
    p_buy: np.ndarray,
    conditional_z: np.ndarray,
    power_p: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates z_factorized = (p_buy ** power_p) * conditional_z and predicted GMV in rubles."""
    p_buy = np.asarray(p_buy, dtype=np.float32)
    conditional_z = np.asarray(conditional_z, dtype=np.float32)

    p_adj = np.power(p_buy, power_p) if power_p != 1.0 else p_buy
    z_factorized = np.maximum(p_adj * conditional_z, 0.0)
    pred_gmv = np.clip(np.expm1(z_factorized), 0.0, None)

    return z_factorized, pred_gmv
