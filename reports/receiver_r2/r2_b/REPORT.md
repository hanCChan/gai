# R2-B minimal controller implementation

Concrete definition: [R2_B_DEFINITION](../../../docs/receiver_r2/R2_B_DEFINITION.md). Source code: [controller.py](../../../src/receiver_r2/controller.py) and [measurement.py](../../../src/receiver_r2/measurement.py).

Deterministic tests validate complete Adam rollback including the next update, nonzero-scaling moment retention, measurement RNG isolation and positive individual RX risk tolerance. Definition sealed before aggregate C2 analysis. Source-error noise constants are fixed; C2/target outcomes never choose a multiplier. No controlled fitting.

Offline source behavior and all scale/phase tables are in [R2-A evidence](../r2_a/EVIDENCE_INDEX.md); they do not form a controlled training trajectory. Initial five-arm45-fit fair matrix included matched SWAD. The current [six-arm54-fit fair proposal](../../../docs/receiver_r2/R2_C_FAIR_MATRIX.md) also includes matched CE, following the frozen convergence assessment. Horizon and new budget require separate approval.
