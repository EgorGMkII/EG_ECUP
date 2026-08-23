"""Audited, leakage-safe BTYD features for CatBoost classifiers.

This mirrors the accepted B1 experiment in ``BTYD_LEAKAGE_AUDIT.md``: exact
full-history RFM, BG/NBD fitted only on RUN training anchors, and three
classifier-only outputs. Gamma-Gamma and monetary BTYD outputs are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from src.btyd_research_pipeline import compute_exact_btyd_predictions, extract_full_history_rfm_for_anchor


@dataclass(frozen=True)
class BTYDRecipe:
    feature_set_id: str = "btyd_audited_b1_classifier_only_v1"
    penalizer_coef: float = 0.001
    horizon_days: int = 30
    max_fit_users: int = 50_000


class AuditedBTYDClassifierProvider:
    """Fit and materialize the accepted B1 BTYD classifier features."""

    feature_set_id = "btyd_audited_b1_classifier_only_v1"
    feature_names = ("btyd_p_buy_30d", "btyd_expected_purchases_30d", "btyd_p_alive")

    def __init__(self, recipe: BTYDRecipe = BTYDRecipe(), *, root_seed: int = 42) -> None:
        self.recipe = recipe
        self.root_seed = int(root_seed)
        self._fit_rows = 0

    @property
    def fit_rows(self) -> int:
        return self._fit_rows

    def fit_transform_anchors(
        self,
        raw: pl.DataFrame,
        users: tuple[int, ...],
        train_anchors: tuple[str, ...],
        holdout_anchor: str,
    ) -> dict[str, pl.DataFrame]:
        """Fit on train-anchor RFM only and transform train plus holdout."""
        from lifetimes import BetaGeoFitter

        if not train_anchors:
            raise ValueError("BTYD requires at least one training anchor")
        rfm = {
            anchor: extract_full_history_rfm_for_anchor(raw, list(users), date.fromisoformat(anchor))
            for anchor in dict.fromkeys((*train_anchors, holdout_anchor))
        }
        pooled = pl.concat([rfm[anchor] for anchor in train_anchors])
        candidates = np.flatnonzero(pooled["btyd_available"].to_numpy() > 0)
        if candidates.size == 0:
            raise RuntimeError("BTYD has no purchasing users on training anchors")
        if candidates.size > self.recipe.max_fit_users:
            rng = np.random.default_rng(self.root_seed)
            fit_idx = np.sort(rng.choice(candidates, size=self.recipe.max_fit_users, replace=False))
        else:
            fit_idx = candidates
        frequency = pooled["btyd_frequency"].to_numpy().astype(np.float64)
        recency = pooled["btyd_recency"].fill_null(0.0).to_numpy().astype(np.float64)
        age = pooled["btyd_T"].fill_null(0.0).to_numpy().astype(np.float64)
        model = BetaGeoFitter(penalizer_coef=self.recipe.penalizer_coef)
        model.fit(frequency[fit_idx], recency[fit_idx], age[fit_idx], verbose=False)
        self._fit_rows = int(fit_idx.size)
        tables: dict[str, pl.DataFrame] = {}
        for anchor, anchor_rfm in rfm.items():
            predicted = compute_exact_btyd_predictions(model, None, anchor_rfm, t_horizons=[self.recipe.horizon_days])
            if self.recipe.horizon_days != 30:
                predicted = predicted.rename({
                    f"btyd_p_buy_{self.recipe.horizon_days}d": "btyd_p_buy_30d",
                    f"btyd_expected_purchases_{self.recipe.horizon_days}d": "btyd_expected_purchases_30d",
                })
            table = predicted.select(("user_id", *self.feature_names)).with_columns(
                pl.col(self.feature_names).fill_nan(0.0).fill_null(0.0).cast(pl.Float32)
            )
            if table["user_id"].to_list() != list(users):
                raise RuntimeError(f"BTYD user alignment failed at {anchor}")
            if not np.isfinite(table.select(self.feature_names).to_numpy()).all():
                raise RuntimeError(f"BTYD produced non-finite values at {anchor}")
            tables[anchor] = table
        return tables
