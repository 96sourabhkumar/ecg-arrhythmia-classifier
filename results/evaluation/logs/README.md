# results/evaluation/logs/

| File | Produced by |
|---|---|
| `training_patient_adaptive.log` | `software/train.py` (20 epochs, epoch 16 selected, validation 98.88 %, float test 97.11 %) |
| `evaluation_patient_adaptive.log` | `software/evaluate.py hardware/weights PA_test` |
| `evaluation_inter_patient.log` | `software/evaluate.py results/models/inter_patient_int8 DS2` |
| `rtl_simulation_ecg_accel_top.log` | `hardware/sim/tb/tb_ecg_accel_top.v` with the deployed weights (64/64) |

These logs were written during development and use the development script names.
