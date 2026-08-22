# RESULT: record-recipe training attempt

- DataSphere job: `bt15ofimtpbt1m8a1ell`
- PRE-RUN commit: `e33b28885c76d83c099357aa524cf0e72244d8d8`
- Status: `SUCCESS`
- Runtime: 1842.227 seconds
- GPU: Tesla V100-PCIE-32GB, CUDA 12.1
- Meta protocol: fixed immutable joint meta; no RUN A/meta fitting
- Public LB: not submitted

## Input hashes

| Input | SHA-256 |
| --- | --- |
| `sample_submit.csv` | `06a433b0ac32f7c0292ce3cb994c1684b4156b392f30fe537ea6a44d0bc4c1b1` |
| joint meta JSON | `e9077605f9b438311c46fa7151a099b617ff457eb5f87d972f465502c873961b` |
| reference prediction bank | `ddb0e882d80f002752f95d10388df40f09a7bebb3d3e61f92153a1a99fdab0d0` |
| historical record submission | `3300512c94579fc6692efb3a6d51a160f0ae5f2375c1476c3aaa54ff775aedcd` |

## Output validation

`artifacts/record_recipe_attempt_v1/candidate_submission.csv` has 250,000
rows, exact template user order, schema `user_id,predict`, and finite
non-negative predictions. Prediction range is 0.1015477454 .. 2908.7703446,
with mean 42.8718371232.

| Output | SHA-256 |
| --- | --- |
| candidate submission | `946ed6b15119fb3c6ee496e75a7d6eb6d944686906fb1e8d1ea565ab0628d264` |
| raw specialist predictions | `34931007b290d470d50e65bc7e04255e2df3806e8f037021e18f2bc65975549d` |
| diagnostics | `d560312761681bfa89083627510cbdf7fd807e4e700912ad1a3d3a32685a12ef` |

This is a valid candidate for manual Public LB upload. Its Public RMSLE is
not knowable locally because test targets are unavailable; the historical
`1.6640779122` remains the comparison target.
