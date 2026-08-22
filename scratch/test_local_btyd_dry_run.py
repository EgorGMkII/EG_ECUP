import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import polars as pl
from pathlib import Path
from datetime import date, timedelta
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.metrics import roc_auc_score, brier_score_loss, root_mean_squared_error
from catboost import CatBoostClassifier, CatBoostRegressor

from src.btyd_research_pipeline import extract_full_history_rfm_for_anchor, compute_exact_btyd_predictions
from src.cadence_features import extract_cadence_features_for_anchor
from src.personal_propensity_features import compute_personal_propensity_features
from src.snapshots import generate_panel_anchors
from src.validation import get_snapshot_path
from scripts.validate_experiment_report import validate_report_invariants

DATA_DIR = Path("data") if Path("data").exists() else Path(".")
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
TRAIN_PARQUET = DATA_DIR / "train.parquet"
USERS_PARQUET = Path("artifacts/selected_users_100k.parquet")

data = pl.read_parquet(TRAIN_PARQUET)
# Take 300 users for quick local test
user_ids = pl.read_parquet(USERS_PARQUET)["user_id"].head(300).to_list()
VAL_ANCHOR = date(2026, 1, 14)
anchors = generate_panel_anchors()
train_anchors = [a for a in anchors if a <= VAL_ANCHOR - timedelta(days=30)][-2:]

print("[+] Running local micro dry-run on 300 users...")
rfm_tables = {}
for a in train_anchors + [VAL_ANCHOR]:
    rfm_df = extract_full_history_rfm_for_anchor(data, user_ids, a)
    rfm_tables[a] = rfm_df

# Fit BetaGeoFitter and GammaGammaFitter
tr_rfm = pl.concat([rfm_tables[a] for a in train_anchors])
tr_avail = tr_rfm["btyd_available"].to_numpy() > 0
tr_freq = tr_rfm["btyd_frequency"].to_numpy().astype(np.float64)
tr_rec = tr_rfm["btyd_recency"].fill_null(0.0).to_numpy().astype(np.float64)
tr_T = tr_rfm["btyd_T"].to_numpy().astype(np.float64)
tr_mon = tr_rfm["btyd_monetary_value"].fill_null(0.0).to_numpy().astype(np.float64)

bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(tr_freq[tr_avail], tr_rec[tr_avail], tr_T[tr_avail])

rep_mask = (tr_freq > 0) & (tr_mon > 0)
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(tr_freq[rep_mask], tr_mon[rep_mask])

btyd_val = compute_exact_btyd_predictions(bgf, ggf, rfm_tables[VAL_ANCHOR])

# Check predictions with fill_nan/fill_null
p_buy_val = btyd_val["btyd_p_buy_30d"].fill_nan(0.0).fill_null(0.0).to_numpy()
fut_buyer_val = np.random.randint(0, 2, size=len(p_buy_val))
auc_score = roc_auc_score(fut_buyer_val, p_buy_val)
brier_score = brier_score_loss(fut_buyer_val, p_buy_val)
print(f"[+] Local ROC-AUC calculation verified: {auc_score:.4f}, Brier: {brier_score:.4f}")

# Test fast CatBoost fit
X_tr = tr_rfm.select(["btyd_frequency", "btyd_T"]).fill_nan(0.0).fill_null(0.0).to_numpy()
y_tr = np.random.randint(0, 2, size=len(X_tr))
clf = CatBoostClassifier(iterations=5, verbose=False)
clf.fit(X_tr, y_tr)
print(f"[+] Local CatBoost fit verified!")
print("[+] LOCAL DRY-RUN PASSED 100% WITHOUT ERROR!")
