# REFERENCE_PIPELINE_V1 — PRE_NY_PRIMARY

This is a new deterministic reference baseline for temporal validation. It is not an exact historical reproduction and does not claim to recreate the Public RMSLE record.

The isolated entrypoint is `scripts/run_reference_pipeline_v1.py`. RUN A trains only on its 12 anchors and fits the frozen meta package on M = 2025-10-13. RUN B starts all base and specialist models from scratch on its 15 anchors, evaluates V = 2025-11-24, and may receive only the frozen RUN A meta package.

Data joins use `user_id`. The state label is GMV over `[anchor - 89, anchor]`; the model target is GMV over `[anchor + 1, anchor + 30]`. Model histories are 180 daily steps for S1/S2 and 365 days / 180 trailing events for ETT. No submission is generated.

Runtime implementation rules:

- build the causal feature store once on the VM for the ordered union of RUN A,
  M, RUN B and V anchors; each run receives only its declared slices;
- cache S1/S2 daily tensors before any repeated raw-log filtering and cache the
  7/14/30-day S2 labels once per anchor;
- store ETT content/time in a pooled FP16 memmap, ranks in int16 and masks in
  bool; compressed per-anchor NPZ files are forbidden;
- release RUN A frames/models before loading RUN B frames; RUN B models still
  start from fresh initialization and receive only the frozen RUN A meta package;
- emit flushed JSON progress records for every major stage and every 250 neural
  steps, including elapsed time and GPU diagnostics.
