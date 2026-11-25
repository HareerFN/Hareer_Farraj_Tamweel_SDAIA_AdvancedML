# قرار التجميع

KEEP SINGLE — Logistic

I kept the single Logistic Regression (gate decision: KEEP SINGLE). On three forward folds it had the highest mean AP, 0.392 (fold SD 0.030), with Brier 0.0633 and ECE 0.0188. The best ensemble, Weighted, reached AP 0.389 (SD 0.029, Brier 0.0633, ECE 0.0177), which is 0.002 lower than Logistic. Stack (0.383) and Equal (0.372) were also lower, while XGBoost (0.353) and LightGBM (0.345) were the weakest. No ensemble passed the gate, and the differences are smaller than one fold SD. An ensemble would add three models' complexity for no measured gain, so the simpler, more interpretable model is the better choice.

The comparison used nested forward OOF on the 6,576-row fit and selection pool. Each outer fold trains only on earlier, label-mature data, with customers separated in both levels. This gave 2,155 OOF rows across three forward periods. Limits: only 3 folds with about 180 defaults in total, so the SD is descriptive, not a significance test. Warm-up rows have no OOF score. The same OOF rows were used to choose both the model and the threshold, so the results are somewhat optimistic. FAST_MODE used fewer trees.

الدليل: artifacts/ensemble_comparison.csv وday5_ensemble_gate.json. SD وصفي، وليس اختبار دلالة.
