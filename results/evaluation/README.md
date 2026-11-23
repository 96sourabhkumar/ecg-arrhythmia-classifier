# results/evaluation/

| File | Contents |
|---|---|
| `weights_PA_test.json` | `software/evaluate.py hardware/weights PA_test`: patient-adaptive model on minutes 5-30 of the 22 DS2 records. Accuracy 97.05 %, strict 95.48 %, macro-F1 0.821, per-class sensitivity and F1, confusion matrix, per-record accuracy |
| `inter_patient_int8_DS2.json` | `evaluate.py results/models/inter_patient_int8 DS2`: inter-patient model on all of DS2. Accuracy 94.60 %, strict 92.95 %, macro-F1 0.619 |
| `qrs_detector_report.csv` | Per-record sensitivity / positive predictivity of the hardware-equivalent Pan-Tompkins detector alone, all 44 records (before PS post-processing) |
| `logs/` | Console logs of training, both evaluations and the system-level RTL simulation |

"Accuracy" counts beats that were detected and matched to an annotation within 150 ms; "strict"
also counts every annotated beat the detector missed as an error. Because the INT8 model is
bit-exact with the hardware (49,027 / 49,027 beats on the board), these files describe the
board's behaviour; the patient-adaptive figure was also measured directly on the board.
