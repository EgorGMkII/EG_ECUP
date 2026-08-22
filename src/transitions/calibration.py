"""Probability Calibration Modules for Reactivation and Churn Classifiers."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Calibrator fitted strictly on training / OOF predictions."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None

    def fit(self, p_uncalibrated: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        p_uncalibrated = np.asarray(p_uncalibrated, dtype=np.float64).reshape(-1, 1)
        y_true = np.asarray(y_true, dtype=np.int32)

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(p_uncalibrated.ravel(), y_true)
        elif self.method == "platt":
            self.calibrator = LogisticRegression()
            self.calibrator.fit(p_uncalibrated, y_true)
        else:
            self.calibrator = None
        return self

    def transform(self, p_uncalibrated: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return np.clip(p_uncalibrated, 0.0, 1.0)

        p_arr = np.asarray(p_uncalibrated, dtype=np.float64)
        if self.method == "isotonic":
            p_cal = self.calibrator.predict(p_arr)
        elif self.method == "platt":
            p_cal = self.calibrator.predict_proba(p_arr.reshape(-1, 1))[:, 1]
        else:
            p_cal = p_arr

        return np.clip(p_cal, 0.0, 1.0).astype(np.float32)
