# Walkthrough: Controlled User-Embedding GRU-180 Experiment Cycle

## 1. Overview of Accomplishments
We completed the rigorous, multi-seed controlled experiment cycle evaluating User ID Embeddings in the GRU-180 sequential architecture on Yandex DataSphere GPU VM.

## 2. Key Experimental Findings

### 2.1. Solo GRU-180 Architecture Comparison (Validation Anchor 2026-01-14, 100k users)
* **E0 (Base Canonical GRU-180, 3-Seed)**: `RMSLE = 1.68709`, React AUC = 0.7538, Churn AUC = 0.8050.
* **E1 (Scalar User Biases, 3-Seed)**: `RMSLE = 1.66850` (**`-0.01859`** improvement vs E0).
* **E2 (User Embedding d=8 for Transitions, 3-Seed)**: `RMSLE = 1.62376` (**`-0.06333`** improvement vs E0, React AUC 0.8486).
* **E3 (Full User Embedding + Conditional Residual, 3-Seed)**: `RMSLE = 1.60255` (**`-0.08454`** improvement vs E0).
* **Control Permutation (`E2_shuffled`)**: `RMSLE = 1.84161` (**`+0.19859`** degradation), confirming the gain is driven purely by persistent user identity.

### 2.2. Ensembling with CatBoost
* **CatBoost C4 Solo**: `1.68431`
* **CatBoost B1 (BTYD) Solo**: `1.68230`
* **CatBoost B1 + GRU E2 (Optimal Blend)**: `1.60735`
* **CatBoost B1 + GRU E3 (Optimal Blend)**: **`1.59559`** (**`-0.08872`** reduction vs CatBoost Solo).

## 3. Published Artifacts
* [**`USER_EMBEDDING_REPORT.md`**](file:///c:/Users/egorg/Documents/OZON_ECUP/USER_EMBEDDING_REPORT.md)
* [**`canonical_gru180_config.json`**](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/user_embedding/canonical_gru180_config.json)
* [**`seed_summary.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/user_embedding/seed_summary.csv)
* [**`blend_summary.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/user_embedding/blend_summary.csv)
* [**`experiment_registry.csv`**](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/user_embedding/experiment_registry.csv)
* Diagnostic plots in [`artifacts/user_embedding/plots/`](file:///c:/Users/egorg/Documents/OZON_ECUP/artifacts/user_embedding/plots/)
