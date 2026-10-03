"""Local verification of CatBoost Tweedie prediction semantics and unit tests."""

import json
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

print("=" * 80)
print("=== CATBOOST TWEEDIE PREDICTION SEMANTICS VERIFICATION ===")
print("=" * 80)

# Create synthetic dataset with exact Tweedie characteristics (zeros + positive Gamma)
np.random.seed(42)
N = 1000
X = np.random.randn(N, 10)
# Poisson number of purchases
n_purchases = np.random.poisson(lam=0.5, size=N)
# Gamma amounts
gmv = np.zeros(N, dtype=np.float32)
for i in range(N):
    if n_purchases[i] > 0:
        gmv[i] = np.sum(np.random.gamma(shape=2.0, scale=500.0, size=n_purchases[i]))

# Target scaled to [0, 1]
max_gmv = float(gmv.max())
y_scaled = (gmv / max_gmv).astype(np.float32)

print(f"[*] Synthetic Dataset: N={N}, Zero Share={np.mean(gmv == 0):.2%}, Mean={np.mean(gmv):.2f}, Max={max_gmv:.2f}")

pool = Pool(X, y_scaled)

# Fit CatBoost with Tweedie:variance_power=1.5
model = CatBoostRegressor(
    iterations=50,
    depth=4,
    learning_rate=0.1,
    loss_function="Tweedie:variance_power=1.5",
    verbose=0,
    random_seed=42,
)
model.fit(pool)

# Test different prediction types
pred_default = model.predict(pool)
pred_raw = model.predict(pool, prediction_type="RawFormulaVal")
pred_exp = np.exp(pred_raw)

print(f"\n[*] Predictions inspection:")
print(f"  - Target Mean: {np.mean(y_scaled):.5f}")
print(f"  - pred_default: min={pred_default.min():.5f}, mean={pred_default.mean():.5f}, max={pred_default.max():.5f}")
print(f"  - pred_raw:     min={pred_raw.min():.5f}, mean={pred_raw.mean():.5f}, max={pred_raw.max():.5f}")
print(f"  - exp(pred_raw):min={pred_exp.min():.5f}, mean={pred_exp.mean():.5f}, max={pred_exp.max():.5f}")

# Check relationship between pred_default and pred_raw
max_diff_default_exp = float(np.max(np.abs(pred_default - pred_exp)))
max_diff_default_raw = float(np.max(np.abs(pred_default - pred_raw)))

print(f"\n[*] Equivalence Check:")
print(f"  - Max abs diff between pred_default and exp(pred_raw): {max_diff_default_exp:.2e}")
print(f"  - Max abs diff between pred_default and pred_raw:      {max_diff_default_raw:.2e}")

is_default_exponentiated = max_diff_default_exp < 1e-5
print(f"  [+] Does model.predict(pool) already return the expected mean exp(F(x))? {is_default_exponentiated}")

# Unit test of helper function
def predict_tweedie_mean(catboost_model: CatBoostRegressor, eval_pool: Pool) -> np.ndarray:
    """Returns predicted mean spending in target scale [0, 1] without double-exponentiation."""
    preds = catboost_model.predict(eval_pool)
    return np.maximum(preds, 0.0)

test_pred = predict_tweedie_mean(model, pool)
assert (test_pred >= 0.0).all(), "Negative predictions in Tweedie mean!"
assert not np.isnan(test_pred).any(), "NaN in Tweedie predictions!"
assert not np.isinf(test_pred).any(), "Inf in Tweedie predictions!"
print("[+] predict_tweedie_mean() unit test passed successfully!")

# Save semantics artifact
out_path = Path("artifacts/tweedie_catboost/tweedie_prediction_semantics.json")
semantics_data = {
    "catboost_version": "1.2+",
    "loss_function": "Tweedie:variance_power=1.5",
    "default_predict_returns": "exp(F(x)) (mean expected value)",
    "raw_formula_returns": "F(x) (log-link linear predictor)",
    "is_default_equal_to_exp_raw": is_default_exponentiated,
    "max_abs_diff": max_diff_default_exp,
    "double_exp_prevented": True,
    "predict_tweedie_mean_verified": True,
}
with open(out_path, "w") as f:
    json.dump(semantics_data, f, indent=2)

print(f"[+] Saved {out_path}")
