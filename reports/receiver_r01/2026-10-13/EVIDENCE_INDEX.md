# R0–R1 evidence index

Full source/input and public file SHA256: PUBLIC_ARTIFACT_SHA256.json.

Derived-only evidence.zip: cb2a2523f282c382f1b64f58cbdec2ddab6f1e55845ee5e7402cd73048747c69

515 archive members verified against their SHA256 in evidence_members.json. All 85 job config/status/selection/curves/update-probes and full process logs are included. Complete large confusion_counts, per_RX_TX_date, paired_RX_TX_changes, update_diagnostics and update_RX_risks CSVs are available under tables/ in this ZIP, indexed by archived_complete_tables. Raw IQ, feature arrays, logits and model weights are excluded. Checkpoint hashes are public; checkpoint bytes stay local.

ZIP uses lossless LZMA compression. Extract with Python: `python -m zipfile -e evidence.zip extracted_evidence`. Original member bytes are unchanged; publication_packaging.json records the connector timeout and compression change.

Count-level recomputation and immutable history/seal checks: verification.json. Source selection and paired sampling hashes are public CSV/JSON. Remote publication requires independent commit-tree match plus report/decision readback.
