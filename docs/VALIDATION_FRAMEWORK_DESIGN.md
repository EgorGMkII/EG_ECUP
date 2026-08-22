# Validation framework design for REFERENCE_PIPELINE_V1

This is an orchestration design, not an implementation or a new training framework.
It reuses existing model/dataset entrypoints behind immutable stage artifacts.

| Stage | Inputs | Outputs | Cache key / integrity gate | Compute |
| --- | --- | --- | --- | --- |
| 01 cohort_manifest | cohort path | cohort manifest | cohort SHA, unique IDs, expected 100k | CPU |
| 02 anchor_manifest | temporal config, raw date bounds | anchors/windows manifest | config SHA; target end checks/no overlap | CPU |
| 03 feature_or_sequence_dataset | cohort, anchors, semantics | feature/sequence manifests | cohort, anchor, schema, preprocessing hashes | CPU/GPU storage |
| 04 base_training | dataset, implementation IDs, seed | base checkpoints + manifests | all upstream hashes, source snapshot, config, seed | GPU |
| 05 specialist_training | bases, task subsets | specialist checkpoints | base hashes, definitions, schedule, seed | CPU/GPU |
| 06 meta_inference | RUN A specialists, M dataset | immutable M prediction bank | checkpoint/schema/order hashes | CPU/GPU |
| 07 meta_fit | M bank and labels | frozen meta package | bank hash, M labels hash, method/alpha/constraints | CPU |
| 08 final_training | RUN B dataset | fresh final checkpoints | RUN B inputs only; never RUN A checkpoint | CPU/GPU |
| 09 validation_inference | RUN B checkpoints, V dataset | immutable V prediction bank | checkpoint/schema/order hashes | CPU/GPU |
| 10 stack_assembly | V bank, frozen meta package | final V predictions | bank/meta hash, fixed formula contract | CPU |
| 11 metrics | predictions, V labels | metrics JSON/Parquet | prediction/label hash, metric version | CPU |
| 12 report | manifests and metrics | run report | all report inputs + template version | CPU |

Stage reuse requires equality of input artifact hashes, cohort and anchor-manifest
hashes, feature/sequence schemas, model implementation IDs, relevant config hash,
seed and upstream checkpoint hashes.  File existence alone is never a cache hit.

Changing only meta method may reuse stages 01–06 and rerun 07/10–12.  Changing only
report metrics may reuse through 10.  Changing activity semantics, history window,
feature schema, sequence preprocessing, model code/implementation ID, training
anchors, schedule, or seed invalidates the relevant downstream chain.

Every run has immutable `experiment_id` and `run_id`, plus a manifest recording
parent/reference run, git commit, dirty-worktree diff/source-snapshot hash, config,
cohort/anchor/feature/sequence hashes, implementation IDs, environment, job IDs,
inputs/outputs, checkpoints/predictions hashes, status and timestamps.  A heavy run
must refuse to start if any implementation ID or schema hash is absent.

The report layer must state comparison/baseline, changed parameters, reused versus
retrained stages, temporal windows/overlap audit, headline and state metrics,
seasonal profile result, Public-LB ranking check, and next-experiment recommendation.

## Decisions requiring explicit confirmation

| Decision | Options | Recommendation now | Consequence / invalidation |
| --- | --- | --- | --- |
| Reference implementation | exact-record claim; versioned new V1 | establish V1, do not claim exact reconstruction | code/preprocessing ID changes invalidate 03–12 |
| S1 vs S2 | retain current shared class; forensic distinct implementations | defer until forensic evidence | any split changes sequence/base/specialist stages |
| ETT canonical form | RUN 1 FF256/pooling; builder FF512/last-token; new design | defer; no silent choice | changes sequence/base/specialist through report |
| Neural anchors | current eight; all 23; another fixed subset | retain only after V1 contract is approved | changes dataset/training/inference downstream |
| 100k cohort | immutable existing file; resample/stratify | keep immutable file despite unknown original stratification | cohort change invalidates all stages |
| Sealed profile | Jan-14 proxy; another late pair | recommend Jan-14 proxy | profile change changes RUN A/B and all downstream results |
| 250k calibration | include; exclude pending lineage | exclude until lineage is adequate | no invalidation unless a new calibration branch is added |
| Compute budget | one profile; primary+stress; all three | authorize profiles explicitly before jobs | determines permitted run IDs/cost, not metric semantics |
