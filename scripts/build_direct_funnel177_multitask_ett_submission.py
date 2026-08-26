"""Full 250k Submission: Direct CatBoost (177 Funnel Features) + Multi-Task Direct ETT Transformer."""
from __future__ import annotations
import argparse, gc, hashlib, json, shutil, tempfile
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
import torch

from src.direct_temporal_cv_v1.adapters.catboost_direct import DirectCatBoostAdapter
from src.direct_temporal_cv_v1.adapters.direct_ett import DirectETTAdapter
from src.direct_temporal_cv_v1.base import FoldContext
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
    ap.add_argument('--output-root', type=Path, default=Path('artifacts/direct_funnel177_multitask_ett_submission_v1'))
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
    ctx = FoldContext(fold, users, target, np.zeros_like(target), snaps.train, snaps.validation, None, None, events.get(fold.train_anchor.isoformat()), events.get(fold.inference_anchor.isoformat()), dev, out, 42)
    
    print('[1/2] Training Direct CatBoost with 177 Funnel Features on Full 250k...', flush=True)
    cb = DirectCatBoostAdapter().fit_predict_fold(
        ctx,
        DirectCatBoostAdapter().validate_config({
            'iterations': 400,
            'depth': 8,
            'learning_rate': 0.04,
            'l2_leaf_reg': 5,
            'loss_function': 'RMSE',
            'thread_count': 8,
            'random_seed': 42
        })
    )
    
    print('[2/2] Training Multi-Task Direct ETT Transformer on GPU on Full 250k...', flush=True)
    ett = DirectETTAdapter().fit_predict_fold(
        ctx,
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
    
    print('[*] Blending Direct CatBoost (0.58) + Multi-Task ETT (0.42)...', flush=True)
    w_cb = 0.58
    w_ett = 0.42
    z = w_cb * cb.prediction_z + w_ett * ett.prediction_z
    pred = np.maximum(np.expm1(z), 0.0)
    
    sub_file = out / 'submission.csv'
    pl.DataFrame({'user_id': users, 'predict': pred}).write_csv(sub_file)
    print(f'[+] Submission saved to {sub_file}', flush=True)
    
    report = {
        'experiment_id': 'direct_funnel177_multitask_ett_submission_v1',
        'pre_run_sha': a.pre_run_sha,
        'job_id': a.job_id,
        'train_sha256': sha256(a.train),
        'sample_submit_sha256': sha256(a.sample_submit),
        'submission_sha256': sha256(sub_file),
        'inference_anchor': '2026-02-13',
        'weights': {'catboost_direct_177f': w_cb, 'ett_direct_multitask': w_ett},
        'models': {
            'catboost_direct_177f': cb.training_report,
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
