# RESULT — CatBoost specialists, 1000 trees

- Individual DataSphere job: `bt11snmo4g41pm7l56fa`
- PRE-RUN: `785b49781fc53542f89b5a98d9989729bbc25187`
- Incremental gate RMSLE: `1.6875206364` vs baseline `1.6875360759`.
- Nominal delta: `-0.0000154394` RMSLE.

Paired bootstrap over all 25,000 ordered V users (5,000 repeats, seed 42):

- delta MSE: `-0.0000521090`;
- 95% CI: `[-0.0009085090, 0.0007587623]`;
- probability candidate better: `0.5564`.

The tiny apparent gain is not statistically established.  The 1500-tree
baseline CatBoost remains the incumbent; no CatBoost recipe is promoted.
