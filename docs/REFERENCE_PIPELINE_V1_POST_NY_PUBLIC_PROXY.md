# REFERENCE_PIPELINE_V1 — POST_NY_PUBLIC_PROXY

This is the active deterministic temporal baseline. It is not an exact
reconstruction of the historical Public-LB record.

RUN A trains on 17 anchors ending `2025-11-10`, fits a frozen meta package on
M=`2025-12-15`, and uses labels from `2025-12-16` through `2026-01-14`.
RUN B starts all base and specialist models from scratch on 20 anchors ending
`2025-12-15`, then evaluates V=`2026-01-14` on the final available labelled
30-day window: `2026-01-15` through `2026-02-13`.

The final RUN B training anchor has target end equal to V. This is allowed by
the temporal contract: a labelled training target may end on the final state
date available at inference; targets after V are forbidden.

All outputs are isolated under
`artifacts/reference_pipeline_v1/post_ny_public_proxy/`; the existing PRE_NY
artifacts remain unchanged.
