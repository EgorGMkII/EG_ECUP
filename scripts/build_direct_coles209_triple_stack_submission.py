"""Full 250k Submission: Triple Diversity Stack (CatBoost 209f + LightGBM 209f + Multi-Task ETT)."""
from __future__ import annotations
import argparse, gc, hashlib, json, shutil, tempfile
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
import torch

from src.direct_temporal_cv_v1.adapters.catboost_direct import DirectCatBoostAdapter
from src.direct_temporal_cv_v1.adapters.lightgbm_direct import DirectLightGBMAdapter
from src.direct_temporal_cv_v1.adapters.direct_ett import DirectETTAdapter
from src.direct_temporal_cv_v1.base import FoldContext
from src.direct_temporal_cv_v1.coles import train_coles_embeddings
from src.direct_temporal_cv_v1.contracts import TemporalFold
from src.direct_temporal_cv_v1.datasets import build_target_z
from src.direct_temporal_cv_v1.features import SparseAggregateFeatureProvider
from src.ssl_temporal_stack_v1.stores import build_event_memmap_store

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=Path, default=Path('data/train.parquet'))
    ap.add_argument('--sample-submit', type=Path, default=Path('sample_submit.csv'))
    ap.add_argument('--output-root', type=Path, default=Path('artifacts/direct_coles209_triple_stack_submission_v1'))
    ap.add_argument('--pre-run-sha', required=True)
    ap.add_argument('--job-id')
    a = ap.parse_args()

    out = a.output_root
    out.mkdir(parents=True, exist_ok=False)

    users = pl.read_csv(a.sample_submit)['user_id'].to_numpy().astype(np.int64)
    raw = pl.read_parquet(a.train)
    fold = TemporalFold('FINAL', date(2026, 2, 13))

    print('[*] Building 177 Catalog & Funnel features for train=2026-01-14 and inference=2026-02-13...', flush=True)
    provider = SparseAggregateFeatureProvider()
    snaps = provider.build_pair(raw, users.tolist(), fold.train_anchor, fold.inference_anchor)
    target = build_target_z(raw, users.tolist(), fold.train_target_start, fold.train_target_end)

    print('[*] Building event sequences store on SSD...', flush=True)
    store_root = Path(tempfile.mkdtemp(prefix='direct_final_'))
    events = build_event_memmap_store(raw, users.tolist(), (fold.train_anchor.isoformat(), fold.inference_anchor.isoformat()), store_root / 'events')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('[*] Training CoLES self-supervised representations on GPU (32 dense features)...', flush=True)
    train_events_arr = events.get(fold.train_anchor.isoformat())
    infer_events_arr = events.get(fold.inference_anchor.isoformat())
    coles_train, coles_infer = train_coles_embeddings(
        train_events_arr,
        infer_events_arr,
        users.tolist(),
        dev,
        epochs=2,
        out_dim=32
    )

    snaps_train_209 = snaps.train.join(coles_train, on="user_id")
    snaps_infer_209 = snaps.validation.join(coles_infer, on="user_id")
    print(f'[+] Tabular matrices ready: {len(snaps_train_209.columns)-1} features', flush=True)

    ctx_209 = FoldContext(fold, users, target, np.zeros_like(target), snaps_train_209, snaps_infer_209, None, None, train_events_arr, infer_events_arr, dev, out, 42)

    print('[1/3] Training CatBoost with 209 Features on Full 250k...', flush=True)
    cb = DirectCatBoostAdapter().fit_predict_fold(
        ctx_209,
        DirectCatBoostAdapter().validate_config({
            'iterations': 400,
            'depth': 8,
            'learning_rate': 0.04,
            'l2_leaf_reg': 5.0,
            'loss_function': 'RMSE',
            'thread_count': 8,
            'random_seed': 42
        })
    )

    print('[2/3] Training LightGBM (leaf-wise) with 209 Features on Full 250k...', flush=True)
    lgb = DirectLightGBMAdapter().fit_predict_fold(
        ctx_209,
        DirectLightGBMAdapter().validate_config({
            'n_estimators': 400,
            'num_leaves': 127,
            'max_depth': 12,
            'learning_rate': 0.04,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'min_child_samples': 50,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'n_jobs': 8,
            'random_state': 42
        })
    )

    print('[3/3] Training Multi-Task Direct ETT Transformer on GPU on Full 250k...', flush=True)
    ett = DirectETTAdapter().fit_predict_fold(
        ctx_209,
        DirectETTAdapter().validate_config({
            'epochs': 2,
            'batch_size': 512,
            'learning_rate': 0.0003,
            'scheduler': 'cosine',
            'warmup_fraction': 0.1,
            'weight_decay': 0.0001,
            'dropout': 0.1,
            'history_days': 180,
            'gradient_accumulation': 1,
            'multitask': True,
            'aux_weight': 0.2
        })
    )

    print('[*] Computing Triple Diversity Blend (CatBoost: 0.38, LightGBM: 0.32, ETT: 0.30)...', flush=True)
    w_cb = 0.38
    w_lgb = 0.32
    w_ett = 0.30
    z = w_cb * cb.prediction_z + w_lgb * lgb.prediction_z + w_ett * ett.prediction_z
    pred = np.maximum(np.expm1(z), 0.0)

    sub_file = out / 'submission.csv'
    pl.DataFrame({'user_id': users, 'predict': pred}).write_csv(sub_file)
    print(f'[+] Submission saved to {sub_file}', flush=True)

    report = {
        'experiment_id': 'direct_coles209_triple_stack_submission_v1',
        'pre_run_sha': a.pre_run_sha,
        'job_id': a.job_id,
        'train_sha256': sha256(a.train),
        'sample_submit_sha256': sha256(a.sample_submit),
        'submission_sha256': sha256(sub_file),
        'inference_anchor': '2026-02-13',
        'weights': {'catboost_direct_209f': w_cb, 'lightgbm_direct_209f': w_lgb, 'ett_direct_multitask': w_ett},
        'models': {
            'catboost_direct_209f': cb.training_report,
            'lightgbm_direct_209f': lgb.training_report,
            'ett_direct_multitask': ett.training_report
        },
        'rows': len(users),
        'prediction_min': float(pred.min()),
        'prediction_median': float(np.median(pred)),
        'prediction_mean': float(pred.mean()),
        'prediction_max': float(pred.max())
    }

    manifest_file = out / 'manifest.json'
    manifest_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)

    events.close()
    shutil.rmtree(store_root, ignore_errors=True)
    gc.collect()

if __name__ == '__main__':
    main()
