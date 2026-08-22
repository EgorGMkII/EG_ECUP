# Git recovery plan

## Current audit

`main` and `origin/main` point to `aa3364f`; history has only that commit and
`7e6224e`.  Git tracks ten files.  The modern pipeline is primarily untracked;
large data, Parquet, models and CSV files are ignored by `.gitignore`.  Existing
modified/deleted tracked files and all untracked files must be preserved.

## What belongs in normal Git

Source, configs, small manifests, Markdown reports, tests, DataSphere YAML, the
experiment registry, artifact hashes, Public-LB records and environment specs.

Do not put raw train data, feature stores, sequence tensors, prediction banks,
checkpoints, large submissions, temporary datasets or large logs in ordinary Git.
External storage must record logical artifact ID, location, SHA-256, size, schema,
producing run ID, producing commit SHA and DataSphere job ID.  Do not adopt LFS or
DVC without a separate decision.

## Safe recovery sequence

1. Preserve `main` and its two genuine commits unchanged.
2. Before any recovery work, create a backup branch/tag.
3. Create a separate recovery branch; never delete untracked files.
4. Create only honest commits marked `[reconstructed snapshot]`; do not alter
   author/commit dates, force-push, rebase, reset, clean or claim historical origin.
5. Prefer one snapshot commit if dependency boundaries cannot be verified.

Proposed groups, only after a file-by-file staging review:

| Proposed message | Scope/dependencies | Large artifacts represented only by |
| --- | --- | --- |
| `chore: [reconstructed snapshot] core data and snapshots` | `src/data.py`, `src/features.py`, `src/snapshots.py`, docs/configs they require | hashes/manifests |
| `feat: [reconstructed snapshot] specialized hurdle baseline` | `src/specialized_hurdle`, baseline scripts, configs | dataset/checkpoint hashes |
| `feat: [reconstructed snapshot] neural S1 S2 ETT research` | sequential code, SSL/ETT scripts, configs | checkpoint/prediction hashes |
| `feat: [reconstructed snapshot] joint RMSLE meta` | joint assembler, JSON contracts, reports | prediction-bank/submission hashes |
| `docs: [reconstructed snapshot] 250k postmortem` | 250k scripts, configs, reports | submission/artifact hashes |
| `docs: [reconstructed snapshot] record reproduction and validation design` | new docs/tests/config examples | reproduction-output hashes |

The grouping is plausible but not yet safe to execute: several scripts duplicate
model definitions, reports refer to missing artifacts, and current modified tracked
files overlap untracked successor code.  Recommendation: first make **one honest
reconstructed snapshot commit** of code/config/docs only, then split later work
only when dependencies are proven.

## Rule for future experiments

Create a pre-run commit before launch:

```text
exp(<experiment_id>): freeze runnable configuration
```

It includes source, configs, protocol, implementation IDs, seeds, environment,
DataSphere manifest and `PLANNED` run manifest.  The job must launch from that SHA.

After actual completion create:

```text
result(<experiment_id>): record validation and Public LB results
```

It records pre-run SHA, job IDs, artifact/checkpoint/prediction hashes, metrics,
submission hash, report and registry update.  Use `PUBLIC_LB_NOT_EVALUATED` when
no public submission exists.
