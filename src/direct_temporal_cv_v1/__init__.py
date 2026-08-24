"""Four-fold, direct-GMV temporal validation framework.

This package is deliberately separate from ``reference_framework_v1``.  It
implements the single-snapshot protocol used to audit the external CatBoost
baseline and is the extension point for direct ETT/TCN candidates.
"""

from .contracts import FOUR_FOLD_250K_V1, TemporalFold

__all__ = ["FOUR_FOLD_250K_V1", "TemporalFold"]
