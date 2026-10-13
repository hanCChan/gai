# R2-A and R2-B evidence

Complete derived-only archives:

- evidence.zip: 15ce8e9f6c0e80397cd854f1b171a5bf426cbacb682359a47056face6854185b
- evidence_part02.zip: 981351aa475f2a06be7a2cae20a84bfc0cc620d6027898be8e73db017fca16b1

120 members verified individually against evidence_members.json, which maps every member to its archive. Includes every job configuration/status/selection/learning curve, risk table, Adam norm/hash statistics and complete process log. All large CSVs are delivered under tables/, with exact archive path/SHA256/bytes in PUBLIC_ARTIFACT_SHA256.json. Read-only offline choices are explicitly separated from controlled trajectories.

Local parameter/Adam snapshots, raw signals, complete features, predictions and weights remain private; only their hashes and derived aggregate losses/counters are public. Independent numerical/packet/count validation is in verification.json. Extract each archive using `python -m zipfile -e ARCHIVE.zip extracted`.

No R2-C fitting; final8 RX/date3/equalized values remain sealed. Publication is complete only after remote commit-tree and all changed-byte SHA256 readback, including REPORT.md and DECISION.md.
