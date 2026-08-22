# Record-recipe training attempt

This is a controlled training-level attempt for the historical 100k joint
submission. It does not claim that the historic training lineage is already
reproduced.

## Contract

- `scripts/build_joint_rmsle_submission.py` is retained unchanged as the
  historical first-level training core.
- `scripts/record_recipe_attempt.py` is the only entrypoint for this attempt.
  It applies the immutable joint meta JSON; it never performs RUN A or fits
  new meta-weights.
- The entrypoint checks the immutable template, joint meta, reference raw
  prediction bank and record submission against their known SHA-256 hashes
  before and after a full run.
- CatBoost is forced to `task_type=GPU`, CUDA is mandatory, and the existing
  neural FP16 memmap implementation remains the sequence store.
- Candidate outputs live only under `artifacts/record_recipe_attempt_v1/`.
  The record CSV and record prediction bank are never writable destinations.

## DataSphere order

1. Run the local `--local-dry-run` check.
2. Commit and push the changed code and manifests as one PRE-RUN commit.
3. Run `datasphere.record_recipe_attempt_smoke.yaml` through
   `scripts/datasphere_runner.py`; it tests CUDA, CatBoost GPU and FP16
   memmap only.
4. If smoke passes, run `datasphere.record_recipe_attempt_full.yaml` through
   the same runner. The runner substitutes the committed SHA into its
   runtime-only YAML copy, since `/job` has no Git metadata.
5. Validate downloaded outputs and create a RESULT commit. Upload the
   candidate CSV manually to Public LB; the returned score is the only test
   of whether the new training attempt reaches the historical `1.6640779122`.

The smoke and full manifests declare only final files as `outputs`; generated
feature stores and memmaps remain VM-local work data.
