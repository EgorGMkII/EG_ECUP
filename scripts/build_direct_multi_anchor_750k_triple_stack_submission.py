"""Full 250k Submission: Multi-Anchor Pooled Training (750,000 samples) on Triple Stack (CatBoost + LightGBM + Multi-Task ETT)."""
from __future__ import annotations
import argparse, gc, hashlib, json, shutil, tempfile
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from catboost import CatBoostRegressor
import lightgbm as lgb

from src.direct_temporal_cv_v1.coles import CoLESEncoder, EventSequenceDataset, FullEventSequenceDataset, info_nce_loss
from src.direct_temporal_cv_v1.contracts import TemporalFold
from src.direct_temporal_cv_v1.datasets import build_target_z
from src.direct_temporal_cv_v1.features import SparseAggregateFeatureProvider
from src.ssl_temporal_stack_v1.models import EventTimeTransformer
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
    ap.add_argument('--output-root', type=Path, default=Path('artifacts/direct_multi_anchor_750k_triple_stack_v1'))
    ap.add_argument('--pre-run-sha', required=True)
    ap.add_argument('--job-id')
    a = ap.parse_args()

    out = a.output_root
    out.mkdir(parents=True, exist_ok=False)

    users = pl.read_csv(a.sample_submit)['user_id'].to_numpy().astype(np.int64)
    raw = pl.read_parquet(a.train)

    anchors = [
        date(2025, 11, 15),
        date(2025, 12, 15),
        date(2026, 1, 14),
        date(2026, 2, 13),  # Inference anchor
    ]
    anchor_strs = tuple(d.isoformat() for d in anchors)

    print(f'[*] Extracting 177 Funnel features across 4 anchor dates: {anchor_strs}...', flush=True)
    provider = SparseAggregateFeatureProvider()
    snapshots = [provider.build_snapshot(raw, users.tolist(), d) for d in anchors]

    print('[*] Extracting target Z for 3 training anchor dates (750,000 total samples)...', flush=True)
    targets_z = [
        build_target_z(raw, users.tolist(), date(2025, 11, 16), date(2025, 12, 15)),
        build_target_z(raw, users.tolist(), date(2025, 12, 16), date(2026, 1, 14)),
        build_target_z(raw, users.tolist(), date(2026, 1, 15), date(2026, 2, 13)),
    ]

    print('[*] Building event sequences store on SSD for all 4 anchors...', flush=True)
    store_root = Path(tempfile.mkdtemp(prefix='multi_anchor_store_'))
    events = build_event_memmap_store(raw, users.tolist(), anchor_strs, store_root / 'events')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('[*] Training CoLES self-supervised representations on GPU (32 dense features)...', flush=True)
    # Stack memmaps of training anchors for CoLES pretraining
    train_content_stacked = np.concatenate([events.get(anchor_strs[i])[0] for i in range(3)], axis=0)
    dataset = EventSequenceDataset(train_content_stacked)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)
    
    coles_model = CoLESEncoder(in_features=train_content_stacked.shape[-1], hidden_dim=64, out_dim=32).to(dev)
    opt = torch.optim.AdamW(coles_model.parameters(), lr=1e-3, weight_decay=1e-4)
    coles_model.train()
    for ep in range(2):
        for w1, w2 in loader:
            w1, w2 = w1.to(dev), w2.to(dev)
            opt.zero_grad()
            loss = info_nce_loss(coles_model(w1), coles_model(w2))
            loss.backward()
            opt.step()

    # Extract CoLES embeddings for all 4 anchors
    coles_model.eval()
    def extract_coles(memmap_tuple):
        ds = FullEventSequenceDataset(memmap_tuple[0])
        dl = DataLoader(ds, batch_size=512, shuffle=False)
        res = []
        with torch.no_grad():
            for b in dl:
                res.append(coles_model(b.to(dev)).cpu().numpy())
        mat = np.concatenate(res, axis=0)
        cols = {f'coles_{i}': mat[:, i].astype(np.float32) for i in range(32)}
        cols['user_id'] = users
        return pl.DataFrame(cols)

    coles_dfs = [extract_coles(events.get(a_str)) for a_str in anchor_strs]
    joined_snaps = [s.join(c, on='user_id') for s, c in zip(snapshots, coles_dfs)]

    feature_order = tuple(c for c in joined_snaps[0].columns if c != 'user_id')
    print(f'[+] Tabular matrices ready with {len(feature_order)} features.', flush=True)

    # Prepare 750k training matrix and 250k test matrix
    x_train_parts = [joined_snaps[i].select(feature_order).to_numpy().astype(np.float32) for i in range(3)]
    x_train = np.concatenate(x_train_parts, axis=0)
    z_train = np.concatenate(targets_z, axis=0)

    x_test = joined_snaps[3].select(feature_order).to_numpy().astype(np.float32)
    print(f'[+] Training Matrix Shape: {x_train.shape} (750,000 samples), Test: {x_test.shape}', flush=True)

    print('[1/3] Training CatBoost on 750,000 samples...', flush=True)
    cb = CatBoostRegressor(
        iterations=600,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=5.0,
        loss_function='RMSE',
        thread_count=8,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    cb.fit(x_train, z_train)
    z_pred_cb = cb.predict(x_test)
    print('[+] CatBoost training done.', flush=True)

    print('[2/3] Training LightGBM on 750,000 samples...', flush=True)
    lgb_model = lgb.LGBMRegressor(
        n_estimators=600,
        num_leaves=127,
        max_depth=12,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=5.0,
        n_jobs=8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(x_train, z_train)
    z_pred_lgb = lgb_model.predict(x_test)
    print('[+] LightGBM training done.', flush=True)

    print('[3/3] Training Multi-Task Direct ETT on GPU on 750,000 samples...', flush=True)
    # Event store data for 3 training anchors
    train_content = np.concatenate([events.get(anchor_strs[i])[0] for i in range(3)], axis=0)
    train_time = np.concatenate([events.get(anchor_strs[i])[1] for i in range(3)], axis=0)
    train_ranks = np.concatenate([events.get(anchor_strs[i])[2] for i in range(3)], axis=0)
    train_mask = np.concatenate([events.get(anchor_strs[i])[3] for i in range(3)], axis=0)

    class MultiAnchorEventDataset(Dataset):
        def __len__(self): return len(z_train)
        def __getitem__(self, idx):
            return (
                torch.from_numpy(train_content[idx].astype(np.float32)),
                torch.from_numpy(train_time[idx].astype(np.float32)),
                torch.from_numpy(train_ranks[idx].astype(np.int64)),
                torch.from_numpy(train_mask[idx]),
                torch.tensor(z_train[idx], dtype=torch.float32),
            )

    ett_ds = MultiAnchorEventDataset()
    ett_loader = DataLoader(ett_ds, batch_size=512, shuffle=True, drop_last=True)
    ett = EventTimeTransformer(transformer_dropout=0.1, head_dropout=0.1).to(dev)
    ett_opt = torch.optim.AdamW(ett.parameters(), lr=3e-4, weight_decay=1e-4)

    ett.train()
    for ep in range(2):
        for c, t, r, m, target in ett_loader:
            c, t, r, m, target = c.to(dev), t.to(dev), r.to(dev), m.to(dev), target.to(dev)
            ett_opt.zero_grad()
            out_logits = ett(c, t, r, m)
            loss_direct = torch.nn.functional.mse_loss(out_logits["direct_z"], target)
            churn_lbl = (target <= 0.0).float()
            loss_churn = torch.nn.functional.binary_cross_entropy_with_logits(out_logits["churn_logit"], churn_lbl)
            loss = loss_direct + 0.2 * loss_churn
            loss.backward()
            ett_opt.step()

    # Predict ETT on inference anchor
    test_events = events.get(anchor_strs[3])
    class TestEventDataset(Dataset):
        def __len__(self): return len(users)
        def __getitem__(self, idx):
            return (
                torch.from_numpy(test_events[0][idx].astype(np.float32)),
                torch.from_numpy(test_events[1][idx].astype(np.float32)),
                torch.from_numpy(test_events[2][idx].astype(np.int64)),
                torch.from_numpy(test_events[3][idx]),
            )
    test_loader = DataLoader(TestEventDataset(), batch_size=512, shuffle=False)
    ett.eval()
    ett_preds = []
    with torch.no_grad():
        for c, t, r, m in test_loader:
            c, t, r, m = c.to(dev), t.to(dev), r.to(dev), m.to(dev)
            ett_preds.append(ett(c, t, r, m)["direct_z"].cpu().numpy())
    z_pred_ett = np.concatenate(ett_preds, axis=0)
    print('[+] Multi-Task ETT training done.', flush=True)

    print('[*] Computing Multi-Anchor Triple Diversity Blend (CatBoost: 0.38, LightGBM: 0.32, ETT: 0.30)...', flush=True)
    w_cb, w_lgb, w_ett = 0.38, 0.32, 0.30
    z_blend = w_cb * z_pred_cb + w_lgb * z_pred_lgb + w_ett * z_pred_ett
    pred = np.maximum(np.expm1(z_blend), 0.0)

    sub_file = out / 'submission.csv'
    pl.DataFrame({'user_id': users, 'predict': pred}).write_csv(sub_file)
    print(f'[+] Submission saved to {sub_file}', flush=True)

    report = {
        'experiment_id': 'direct_multi_anchor_750k_triple_stack_v1',
        'pre_run_sha': a.pre_run_sha,
        'job_id': a.job_id,
        'train_sha256': sha256(a.train),
        'sample_submit_sha256': sha256(a.sample_submit),
        'submission_sha256': sha256(sub_file),
        'training_samples': len(z_train),
        'inference_anchor': '2026-02-13',
        'weights': {'catboost_direct_209f': w_cb, 'lightgbm_direct_209f': w_lgb, 'ett_direct_multitask': w_ett},
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
