"""CatBoost Specialist Training for Reactivation, Churn, and Amount."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor


def train_fold_catboost_specialists(
    feature_store_dir: Path,
    train_anchors: List[str],
    val_anchor: str,
    out_dir: Path,
    iterations: int = 1500,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Trains 3 independent CatBoost models (React, Churn, Amount) for a given fold."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and concatenate training anchors
    train_dfs = []
    for a in train_anchors:
        p = feature_store_dir / f"anchor_{a}.parquet"
        if p.exists():
            train_dfs.append(pl.read_parquet(p))

    df_train = pl.concat(train_dfs)
    val_df = pl.read_parquet(feature_store_dir / f"anchor_{val_anchor}.parquet")

    # Feature columns (exclude user_id, target, lifetime_gmv)
    excluded = {"user_id", "target", "lifetime_gmv", "will_buy_30d"}
    feat_cols = [c for c in df_train.columns if c not in excluded]

    X_train_all = df_train.select(feat_cols).to_numpy().astype(np.float32)
    y_train_gmv = df_train["target"].to_numpy().astype(np.float32)
    past_gmv_train = df_train["lifetime_gmv"].to_numpy().astype(np.float32)
    was_act_train = (past_gmv_train > 0).astype(int)
    will_buy_train = (y_train_gmv > 0).astype(int)

    X_val_all = val_df.select(feat_cols).to_numpy().astype(np.float32)
    y_val_gmv = val_df["target"].to_numpy().astype(np.float32)
    past_gmv_val = val_df["lifetime_gmv"].to_numpy().astype(np.float32)
    was_act_val = (past_gmv_val > 0).astype(int)
    will_buy_val = (y_val_gmv > 0).astype(int)

    # A. CB_REACT: was_active == 0, target = will_buy
    mask_react_train = was_act_train == 0
    mask_react_val = was_act_val == 0
    cb_react = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=random_seed,
        verbose=False,
    )
    cb_react.fit(
        X_train_all[mask_react_train],
        will_buy_train[mask_react_train],
        eval_set=(X_val_all[mask_react_val], will_buy_val[mask_react_val]),
        early_stopping_rounds=100,
        verbose=False,
    )
    cb_react.save_model(out_dir / "catboost_react.cbm")
    # Output raw logits for stacking
    react_logits_val = cb_react.predict(X_val_all, prediction_type="RawFormulaVal")

    # B. CB_CHURN: was_active == 1, target = 1 - will_buy
    mask_churn_train = was_act_train == 1
    mask_churn_val = was_act_val == 1
    cb_churn = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=random_seed,
        verbose=False,
    )
    cb_churn.fit(
        X_train_all[mask_churn_train],
        (1 - will_buy_train[mask_churn_train]),
        eval_set=(X_val_all[mask_churn_val], (1 - will_buy_val[mask_churn_val])),
        early_stopping_rounds=100,
        verbose=False,
    )
    cb_churn.save_model(out_dir / "catboost_churn.cbm")
    churn_logits_val = cb_churn.predict(X_val_all, prediction_type="RawFormulaVal")

    # C. CB_AMOUNT: future_gmv > 0, target = log1p(future_gmv)
    mask_amt_train = y_train_gmv > 0
    mask_amt_val = y_val_gmv > 0
    cb_amount = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_seed,
        verbose=False,
    )
    cb_amount.fit(
        X_train_all[mask_amt_train],
        np.log1p(y_train_gmv[mask_amt_train]),
        eval_set=(X_val_all[mask_amt_val], np.log1p(y_val_gmv[mask_amt_val])),
        early_stopping_rounds=100,
        verbose=False,
    )
    cb_amount.save_model(out_dir / "catboost_amount.cbm")
    amount_z_val = np.maximum(0.0, cb_amount.predict(X_val_all))

    return {
        "react_logits": react_logits_val,
        "churn_logits": churn_logits_val,
        "amount_z": amount_z_val,
        "was_active": was_act_val,
        "will_buy": will_buy_val,
        "y_rub": y_val_gmv,
    }
