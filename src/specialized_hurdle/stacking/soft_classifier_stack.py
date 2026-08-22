"""Soft Classifier Stack with Temperature Scaling & Softmax Weights."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize as opt
from scipy.special import expit, logit


@dataclass
class SoftStackResult:
    weights: np.ndarray
    temperature: float
    bias: float
    best_logloss: float
    model_names: List[str]


def fit_soft_classifier_stack(
    prob_matrix: np.ndarray,
    y_true: np.ndarray,
    model_names: List[str],
    eps: float = 1e-6,
) -> SoftStackResult:
    """Optimizes softmax weights and temperature-scaled bias by minimizing BCE (LogLoss).

    prob_matrix: shape (N, M) with model predicted probabilities in [0, 1].
    y_true: shape (N,) binary targets in {0, 1}.
    """
    n_samples, n_models = prob_matrix.shape
    clipped_probs = np.clip(prob_matrix, eps, 1.0 - eps)
    logits_matrix = logit(clipped_probs)

    # Parameter vector: theta (n_models weights), raw_temp (scalar), bias (scalar)
    def loss_func(params):
        theta = params[:n_models]
        raw_temp = params[n_models]
        bias = params[n_models + 1]

        # Softmax weights (sum to 1, all >= 0)
        exp_theta = np.exp(theta - np.max(theta))
        w = exp_theta / np.sum(exp_theta)

        # Temperature > 0
        temp = np.log1p(np.exp(raw_temp)) + 1e-4

        combined_logits = np.dot(logits_matrix, w)
        scaled_logits = temp * combined_logits + bias
        p_stack = expit(scaled_logits)
        p_stack_clipped = np.clip(p_stack, eps, 1.0 - eps)

        bce = -np.mean(y_true * np.log(p_stack_clipped) + (1.0 - y_true) * np.log(1.0 - p_stack_clipped))
        return bce

    init_params = np.zeros(n_models + 2)
    init_params[n_models] = 0.5413  # softplus(0.5413) ~ 1.0
    init_params[n_models + 1] = 0.0

    res = opt.minimize(loss_func, init_params, method="L-BFGS-B")

    theta_opt = res.x[:n_models]
    raw_temp_opt = res.x[n_models]
    bias_opt = res.x[n_models + 1]

    exp_theta_opt = np.exp(theta_opt - np.max(theta_opt))
    w_opt = exp_theta_opt / np.sum(exp_theta_opt)
    temp_opt = float(np.log1p(np.exp(raw_temp_opt)) + 1e-4)

    return SoftStackResult(
        weights=w_opt,
        temperature=temp_opt,
        bias=float(bias_opt),
        best_logloss=float(res.fun),
        model_names=model_names,
    )


def predict_soft_classifier_stack(
    stack: SoftStackResult,
    prob_matrix: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Predicts calibrated probabilities given input model probabilities."""
    clipped_probs = np.clip(prob_matrix, eps, 1.0 - eps)
    logits_matrix = logit(clipped_probs)
    combined_logits = np.dot(logits_matrix, stack.weights)
    scaled_logits = stack.temperature * combined_logits + stack.bias
    return expit(scaled_logits)
