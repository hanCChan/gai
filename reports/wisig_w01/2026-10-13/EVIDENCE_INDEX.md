# W0/W1 evidence index

- [evidence.zip](evidence.zip): 52 CRC/SHA256 verified members. Full8736 prediction rows,34944 identity rows,2 run configurations/status/metrics,155 learning-curve epochs,logs,coverage-preflight abort snapshots and sources. No IQ,features,weights or credentials.
- [evidence_members.json](evidence_members.json): member bytes/hashes.
- [verification.json](verification.json),[pre_execution_seal.json](pre_execution_seal.json),[preflight.json](preflight.json): independent recomputation,pre-fit seal and cache/input hashes.
- [data_audit.csv](data_audit.csv),[invalid_samples.csv](invalid_samples.csv),[rx_hardware_and_split.csv](rx_hardware_and_split.csv):139852 development packets,0 numeric-invalid packets,14/10/8 RX assignment and three hardware models.
- [w0_pilot_summary.csv](w0_pilot_summary.csv),[w0_learning_curves.csv](w0_learning_curves.csv),[w0_receiver_date_metrics.csv](w0_receiver_date_metrics.csv): all W0 results;source-only checkpoint choice.
- [factorial_components.csv](factorial_components.csv),[sampling_uncertainty.csv](sampling_uncertainty.csv),[interaction_cross_date_repeatability.csv](interaction_cross_date_repeatability.csv): descriptive structure/repeatability.
- [strict_TX_date_predictions.csv](strict_TX_date_predictions.csv),[transformed_known_TX_identity.csv](transformed_known_TX_identity.csv),[prediction_summary.csv](prediction_summary.csv): all predictions and identity checks;no per-cell oracle selector.
- [shen_failure_review.json](shen_failure_review.json): historical posthoc count recomputation,no new Shen fitting.
- [w0_w1_results.pdf](w0_w1_results.pdf),[w0_w1_results.png](w0_w1_results.png): standalone figures.

Final8 RX/final date/equalized values were not converted to numeric arrays or encoded;opaque ZIP structure parsing is disclosed.
