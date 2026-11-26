# results/models/

| File | Contents |
|---|---|
| `ecgnet_patient_adaptive.pt` | PyTorch `state_dict` of the deployed model: trained on DS1 + minutes 0-4 of each DS2 record (adaptation beats repeated 4×), epoch selected on minute 4-5. Quantised into `hardware/weights/` |
| `training_patient_adaptive.json` | Arguments, per-epoch validation accuracy and macro-F1, selected epoch, float test accuracy (97.11 %) and confusion matrix |
| `ecgnet_inter_patient.pt` | The inter-patient model (trained on DS1 only; development name "W5") |
| `inter_patient_int8/` | INT8 export of `ecgnet_inter_patient.pt` in the same format as `hardware/weights/`, used by `evaluate.py ... DS2` |

Load with `ECGNet().load_state_dict(torch.load(path))` (`software/ecgnet.py`).
