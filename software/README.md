# software/

Python code for every step outside the FPGA fabric: data preparation, training, quantisation, the
bit-exact reference models, evaluation, test-vector generation and the runtime that drives the
board. Run the scripts from this folder (they add it to `sys.path` and use paths relative to the
repository root).

## Pipeline (in order)

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `mitbih.py --download` | PhysioNet | `data/mitdb/` (48 records); also `data/processed/` legacy 128-sample arrays and `qrs_report.csv` |
| 2 | `beat_dataset.py` | `data/mitdb/` | `data/beats/{DS1,DS2,PA_train,PA_train4,PA_val,PA_test}.npz` |
| 3 | `train.py` | `data/beats/` | `results/models/ecgnet_patient_adaptive.pt`, `training_patient_adaptive.json` |
| 4 | `quantize.py` | model + `data/beats/PA_train.npz` (calibration) | `hardware/weights/*` |
| 5 | `evaluate.py <weights dir> <PA_test or DS2>` | `data/mitdb/`, weights | `results/evaluation/<dir>_<split>.json` |
| 6 | `make_board_vectors.py [records]` | `data/mitdb/`, `hardware/weights/` | `pynq/test_vectors/record_<N>.npz` |
| 7 | `board_test.py`, `board_summary.py` | on the PYNQ-Z2 | `pynq/results/board_test_results.json` |

## Modules

| File | Purpose |
|---|---|
| `config.py` | Record lists (DS1/DS2 inter-patient split, paced records excluded), AAMI symbol→class map, sampling rates |
| `qrs/pan_tompkins.py` | Bit-exact integer Pan-Tompkins (360→180 Hz halfband decimation, filters, peak detector): the golden model of `hardware/rtl/qrs/` |
| `qrs/rr_features.py` | Integer RR-feature specification `rr_int8()`: golden model of `rr_feature_engine.v`, also used by the driver |
| `mitbih.py` | MIT-BIH download and `load_record()`, float RR features for training |
| `beat_matching.py` | Greedy one-to-one matching of detected peaks to annotations (150 ms tolerance) |
| `peak_postprocessing.py` | PS-side peak correction: position refinement (±24), duplicate removal (200 ms), T-wave rejection, searchback |
| `beat_dataset.py` | Builds training/validation/test beat sets at the post-processed peak positions |
| `ecgnet.py` | `ECGNet` (PyTorch, 8,101 parameters) and batched `predict()` |
| `classification_metrics.py` | Accuracy, macro-F1 (N, S, V, F), confusion matrix (NumPy only) |
| `train.py` | Training with class weighting and validation-only epoch selection |
| `quantize.py` | BatchNorm folding, INT8 scales, requant multipliers/shifts, hex memory images, golden vectors |
| `int8_model.py` | Bit-exact INT8 reference of the hardware CNN; reads the files in `hardware/weights/` |
| `evaluate.py` | End-to-end bit-exact evaluation: detector → post-processing → INT8 CNN, accuracy and strict accuracy |
| `make_board_vectors.py` | Golden expectations for the board test (peaks, RR words, classes per beat) |
| `pynq_mmio.py` | Register map, MMIO wrapper (`EcgAccelMMIO`) and beat normalisation/quantisation |
| `pynq_driver.py` | Board runtime: streams samples, post-processes peaks, computes RR, classifies each beat (`EcgAccel`, `run()`) |
| `board_test.py`, `board_summary.py` | On-board verification against the golden vectors and the summary of the 22-record run |
| `gen_filter_vectors.py`, `gen_peak_detector_vectors.py`, `gen_rr_vectors.py` | Test vectors for the unit testbenches in `hardware/sim/vectors/` |
| `tests/mock_board_test.py` | Runs the board runtime against a software model of the PL (no board needed) |

Only `config.py`, `qrs/`, `peak_postprocessing.py`, `pynq_mmio.py`, `pynq_driver.py`,
`board_test.py` and `board_summary.py` are needed on the board (see `pynq/make_pynq_package.py`).
