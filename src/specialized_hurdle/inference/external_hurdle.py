"""Clean Single External Hurdle Assembler."""

from typing import Dict, Optional, Tuple
import numpy as np


def assemble_external_hurdle(
    p_react_stack: np.ndarray,
    p_churn_stack: np.ndarray,
    conditional_z_stack: np.ndarray,
    was_active: np.ndarray,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combines specialized stacks into the final Hurdle prediction without double-factorization.

    Returns: (p_buy, factorized_z, predicted_gmv)
    """
    is_act = was_active.astype(bool)

    # Clean Hurdle Probability:
    # If active in past 30 days -> P(will_buy) = 1.0 - P(churn)
    # If inactive in past 30 days -> P(will_buy) = P(reactivation)
    p_buy = np.where(is_act, 1.0 - p_churn_stack, p_react_stack)
    p_buy_clamped = np.clip(p_buy, eps, 1.0 - eps)

    # Hurdle factorization:
    # Under calibrated probabilities and log-space conditional MSE:
    # E[z | X] = P(Y > 0 | X) * E[z | Y > 0, X]
    factorized_z = (p_buy_clamped ** alpha) * conditional_z_stack

    predicted_gmv = np.expm1(np.maximum(0.0, factorized_z))

    return p_buy_clamped, factorized_z, predicted_gmv
