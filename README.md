# 4-16-pointcloud

This repository is based on GeoTransformer and is being reorganized for a paper-oriented pipeline.

## Branch roles

- `legacy-hogr-crm-rcl`:
  preserves the mixed experimental state containing HOGR + CRM + RCL + GPR stub.

- `paper-hogr-ucot`:
  the paper-oriented main line.
  Target story:
  GeoTransformer + HOGR as the main contribution,
  with UCOT as the secondary extension.

## Current status

- Baseline reproduction completed.
- HOGR v1 has been connected into the training pipeline.
- CRM/RCL code exists from earlier experiments.
- GPR is currently only a stub.
- 1-epoch smoke test passed.
- 3-epoch pilot still has OOM in the frontend geometric embedding stage.

## Current priority

1. stabilize HOGR v1 training
2. finish short pilot runs
3. then move to HOGR v2 / UCOT
