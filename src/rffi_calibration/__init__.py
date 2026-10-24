"""Low-budget receiver calibration for closed-set RFFI (experiment/rffi-budget-calibration-v1).

Task: the same registered transmitters are observed on a new receiver; each TX may supply
m in {0, 1, 2, 5, 10} trusted labelled packets on that receiver. Query packets are scored
inductively: their labels never enter any statistic, head or selection.

Specification: docs/rffi_calibration/EXPERIMENT_PROTOCOL.md. Attribution of every head to its
original (often non-RF) source: docs/rffi_calibration/NOVELTY_AND_BASELINE_AUDIT.md.
"""
